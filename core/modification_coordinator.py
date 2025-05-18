# core/modification_coordinator.py
import logging
import ast # For parsing FILES_TO_MODIFY list
import re  # For parsing plan segments
import os  # For file operations

from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

# Assuming these imports are correct from your project structure
try:
    from .modification_handler import ModificationHandler
    from .backend_coordinator import BackendCoordinator
    from .project_context_manager import ProjectContextManager
    from .models import ChatMessage, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from utils import constants
except ImportError as e:
    logging.critical(f"ModificationCoordinator: Failed to import core components: {e}")
    # Define fallback types for type hinting and basic script operation
    ModificationHandler = type("ModificationHandler", (object,), {}) # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {}) # type: ignore
    ProjectContextManager = type("ProjectContextManager", (object,), {}) # type: ignore
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    constants = type("constants", (object,), {}) # type: ignore
    USER_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "system", "error" # type: ignore

logger = logging.getLogger(__name__)

# Define Backend IDs - ensure these match your constants or actual IDs
PLANNER_BACKEND_ID = getattr(constants, "PLANNER_BACKEND_ID", "gemini_planner")
GENERATOR_BACKEND_ID = getattr(constants, "GENERATOR_BACKEND_ID", "ollama_generator")

class ModPhase: # State machine phases
    IDLE = "IDLE"
    AWAITING_PLAN = "AWAITING_PLAN" # Waiting for the overall plan from Planner AI
    AWAITING_GENERATOR_PROMPT_REFINEMENT = "AWAITING_GENERATOR_PROMPT_REFINEMENT" # Waiting for Planner AI to create/refine prompt for Generator AI
    AWAITING_CODE_GENERATION = "AWAITING_CODE_GENERATION" # Waiting for Generator AI to produce code
    # REMOVED: AWAITING_USER_CONFIRMATION_FOR_CURRENT_FILE
    ALL_FILES_GENERATED_AWAITING_USER_ACTION = "ALL_FILES_GENERATED_AWAITING_USER_ACTION" # All files generated, user can give overall feedback or accept

class ModificationCoordinator(QObject):
    # Signals to ChatManager/MainWindow
    request_llm_call = pyqtSignal(str, list)  # backend_id, history_to_send
    file_ready_for_display = pyqtSignal(str, str) # filename, content (AI-generated full content)
    modification_sequence_complete = pyqtSignal(str, str) # reason, original_query_summary
    modification_error = pyqtSignal(str)      # error_message (for UI error display)
    status_update = pyqtSignal(str)           # General status message for chat UI

    MAX_LINES_BEFORE_SPLIT = 400 # Constant for splitting large files

    def __init__(self,
                 modification_handler: ModificationHandler,
                 backend_coordinator: BackendCoordinator,
                 project_context_manager: ProjectContextManager,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([modification_handler, backend_coordinator, project_context_manager]):
            logger.critical("MC Init: Missing critical dependencies.")
            raise ValueError("ModificationCoordinator requires valid handler, backend_coord, and project_manager.")

        self._handler = modification_handler
        self._backend_coordinator = backend_coordinator
        self._project_context_manager = project_context_manager

        self._is_active: bool = False
        self._is_awaiting_llm: bool = False
        self._current_phase: str = ModPhase.IDLE

        # Information about the overall modification request
        self._original_query: Optional[str] = None # Holds current user intent (initial or overall refinement feedback)
        self._original_query_at_start: Optional[str] = None # Holds the very first user query for context
        self._original_context_from_rag: Optional[str] = None
        self._original_focus_prefix: Optional[str] = None

        # Plan-related state
        self._full_planner_proto_specification: Optional[str] = None
        self._planned_files_list: List[str] = []

        # Per-file iteration state
        self._current_file_index: int = -1
        self._current_target_filename: Optional[str] = None
        self._current_original_file_content_for_ai: Optional[str] = None
        self._current_plan_segment_for_file: Optional[str] = None
        self._last_generated_prompt_for_coder: Optional[str] = None
        # self._user_feedback_for_current_file: Optional[str] = None # REMOVED - No per-file feedback cycle

        # For handling split files
        self._current_file_parts_queue: List[Tuple[str, str]] = [] # Queue of (filename_with_part_label, content_part)

        self._connect_handler_signals()
        logger.info("ModificationCoordinator initialized.")

    def _connect_handler_signals(self):
        if self._handler:
            try:
                self._handler.modification_parsing_error.connect(self._handle_mh_parsing_error)
                logger.debug("MC connected to ModificationHandler's parsing error signal.")
            except Exception as e:
                logger.exception(f"Error connecting ModificationHandler signals in MC: {e}")

    def _reset_state(self):
        logger.debug("MC: Resetting internal state.")
        self._is_active = False
        self._is_awaiting_llm = False
        self._current_phase = ModPhase.IDLE
        self._original_query = None
        self._original_query_at_start = None
        self._original_context_from_rag = None
        self._original_focus_prefix = None
        self._full_planner_proto_specification = None
        self._planned_files_list = []
        self._current_file_index = -1
        self._current_target_filename = None
        self._current_original_file_content_for_ai = None
        self._current_plan_segment_for_file = None
        self._last_generated_prompt_for_coder = None
        self._current_file_parts_queue = []
        if self._handler:
            self._handler.cancel_modification()

    def is_active(self) -> bool:
        return self._is_active

    def is_awaiting_llm_response(self) -> bool:
        return self._is_active and self._is_awaiting_llm

    def start_sequence(self, query: str, context_from_rag: str, focus_prefix: str):
        if self._is_active:
            logger.warning("MC: start_sequence called while active. Resetting previous sequence.")
            self._reset_state()

        logger.info(f"MC: Starting new modification sequence. Query: '{query[:50]}...'")
        self._is_active = True
        self._original_query = query
        self._original_query_at_start = query
        self._original_context_from_rag = context_from_rag
        self._original_focus_prefix = focus_prefix
        self._is_awaiting_llm = False

        self._request_initial_plan_from_planner()

    def _request_initial_plan_from_planner(self):
        logger.debug("MC: Requesting plan (or re-plan) from Planner AI.")
        prompt_text_parts = [
            "You are an expert AI system planner. Your task is to create a detailed step-by-step plan.\n"
        ]

        if self._current_phase == ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION and self._original_query != self._original_query_at_start:
            # This is an overall refinement after all files were processed once and user provided feedback
            prompt_text_parts.append(
                f"This is a REFINEMENT of a previous overall plan based on user feedback after all files were generated. "
                f"The original goal was: \"{self._original_query_at_start}\".\n"
                f"The user's latest feedback on the generated files (potentially affecting multiple files) is: \"{self._original_query}\".\n" # _original_query now holds the overall feedback
                f"Re-evaluate the entire plan. Identify ALL files that need to be changed (added, modified, or even if a previously modified file now needs no changes or different changes) based on this feedback.\n"
            )
        else: # Initial plan request
            prompt_text_parts.append(f"Implement the following user request:\nUSER REQUEST: \"{self._original_query_at_start}\"\n\n")

        prompt_text_parts.extend([
            f"ASSOCIATED PROJECT CONTEXT (from RAG):\n{self._original_context_from_rag or 'N/A'}\n\n",
            f"PROJECT ROOT FOCUS (base path for relative file paths): {self._original_focus_prefix or 'N/A'}\n\n",
            "Your response MUST include two main parts:\n",
            "1. A machine-parseable line identifying all files to be modified or created. This line MUST start "
            "EXACTLY with 'FILES_TO_MODIFY: ' followed by a Python-style list of relative file paths "
            "(e.g., FILES_TO_MODIFY: ['src/file1.py', 'app/new_module/file2.py']). "
            "If no files need changes, use FILES_TO_MODIFY: []. This line should appear first or very early.\n",
            "2. After that line, provide a DETAILED natural language proto-specification. For EACH file listed in FILES_TO_MODIFY, "
            "you MUST include a dedicated section clearly marked like this:\n"
            "--- FILE_PLAN_START: path/to/filename.ext ---\n"
            "[Detailed logic, changes, new functions/classes for this specific file. If the plan is to effectively remove the file's utility or make it empty, state that clearly.]\n"
            "--- FILE_PLAN_END: path/to/filename.ext ---\n"
            "The proto-specification should also cover:\n"
            "- The refined user goal based on the request.\n"
            "- Key RAG insights derived from the RAG context that informed your plan.\n"
            "- Descriptions of any new data structures or interfaces if applicable.\n"
            "- Potential constraints, edge cases, or important considerations for the implementation.\n"
            "Ensure the plan is comprehensive enough for another AI to generate the code for each file based on its segment of the plan and original file content (if modifying).\n"
            "Generate the plan now."
        ])
        prompt_text = "".join(prompt_text_parts)

        self._full_planner_proto_specification = None # Reset for new plan

        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[prompt_text])]
        self._is_awaiting_llm = True
        self._current_phase = ModPhase.AWAITING_PLAN
        self.status_update.emit("[System: Asking Planner AI to create/refine modification plan...]")
        self.request_llm_call.emit(PLANNER_BACKEND_ID, history_for_llm)

    def _handle_plan_response(self, planner_response_text: str):
        logger.info("MC: Received response from Planner AI for the detailed plan.")
        self._is_awaiting_llm = False
        self._full_planner_proto_specification = planner_response_text.strip()

        parsed_list, error_msg_parse = self._parse_files_to_modify_list(self._full_planner_proto_specification)
        if error_msg_parse or parsed_list is None:
            err_msg_ui = f"Failed to parse FILES_TO_MODIFY list from Planner AI: {error_msg_parse}. Response preview: '{planner_response_text[:300]}...'"
            self.modification_error.emit(err_msg_ui)
            self._handle_sequence_end("error_plan_parsing", err_msg_ui)
            return

        self._planned_files_list = parsed_list
        if not self._planned_files_list:
            self.status_update.emit("[System: Planner AI indicates no file modifications are needed based on the current request/feedback.]")
            self._handle_sequence_end("completed_no_files_in_plan", "Planner found no files to modify in the current plan.")
            return

        logger.info(f"MC: Plan successfully parsed. Files to modify/create: {self._planned_files_list}")
        files_str_display = ", ".join([f"`{f}`" for f in self._planned_files_list])
        self.status_update.emit(f"[System: Planner AI will process: {files_str_display}]")

        self._current_file_index = -1 # Reset for the start of new file processing
        self._current_file_parts_queue = [] # Clear any old parts
        QTimer.singleShot(0, self._proceed_to_next_file_or_part) # Start processing

    def _proceed_to_next_file_or_part(self):
        self._is_awaiting_llm = False # Ensure this is reset

        # First, check if there are pending parts of a split file to display
        if self._current_file_parts_queue:
            filename_with_part_label, content_part = self._current_file_parts_queue.pop(0)
            self.file_ready_for_display.emit(filename_with_part_label, content_part)
            # If there are more parts for THIS file, stay in a state that will call this method again
            # Otherwise, after the last part, this method will be called again and move to the next file.
            if self._current_file_parts_queue: # More parts of the current file remain
                 self.status_update.emit(f"[System: Displaying next part of `{self._current_target_filename}`...]")
                 QTimer.singleShot(10, self._proceed_to_next_file_or_part) # Tiny delay to allow UI update
                 return
            else: # Last part of current file was just displayed
                 self.status_update.emit(f"[System: All parts of `{self._current_target_filename}` displayed. Processing next file...]")
                 # Fall through to increment _current_file_index and process next file

        # If no pending parts, move to the next file in the plan
        self._current_file_index += 1

        if not (0 <= self._current_file_index < len(self._planned_files_list)):
            # All files in the _planned_files_list have been processed
            logger.info("MC: All planned files and their parts processed.")
            self.status_update.emit(
                "[System: All planned files generated. Review in Code Viewer. "
                "Provide overall feedback for refinement or type 'accept' to finalize this modification task.]"
            )
            self._current_phase = ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION
            self._is_awaiting_llm = False # Ensure user can provide feedback
            return

        # Set up for the next file
        self._current_target_filename = self._planned_files_list[self._current_file_index]
        self.status_update.emit(
            f"[System: Preparing to process file {self._current_file_index + 1}/{len(self._planned_files_list)}: `{self._current_target_filename}`...]"
        )

        self._current_original_file_content_for_ai = self._read_original_file_content(self._current_target_filename)
        self._current_plan_segment_for_file = self._extract_plan_segment_for_file(
            self._current_target_filename,
            self._full_planner_proto_specification or ""
        )
        if not self._current_plan_segment_for_file:
            logger.warning(f"MC: Could not extract plan segment for '{self._current_target_filename}'. Using generic instructions.")
            self._current_plan_segment_for_file = f"Implement changes for '{self._current_target_filename}' based on the overall user request: \"{self._original_query_at_start}\"."
            self.status_update.emit(f"[System Warning: No detailed plan segment found for `{self._current_target_filename}`. Coder AI will use general plan.]")

        self._request_generator_prompt_from_planner(
            target_filename_for_generator=self._current_target_filename,
            original_file_content_for_context=self._current_original_file_content_for_ai,
            plan_segment_for_this_file=self._current_plan_segment_for_file
        )

    def _read_original_file_content(self, relative_filename: str) -> Optional[str]:
        content: Optional[str] = None
        full_path: Optional[str] = None

        if self._original_focus_prefix and os.path.isdir(self._original_focus_prefix):
            norm_relative_path = relative_filename.lstrip('/').lstrip('\\')
            full_path = os.path.normpath(os.path.join(self._original_focus_prefix, norm_relative_path))
        elif os.path.isabs(relative_filename):
            full_path = relative_filename
            logger.warning(f"MC: Planner provided an absolute path '{relative_filename}'. Prefer relative paths from focus_prefix.")

        if full_path and os.path.exists(full_path) and os.path.isfile(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"MC: Internally read original content for '{relative_filename}' from '{full_path}'.")
            except Exception as e:
                logger.error(f"MC: Failed to read original content of '{relative_filename}' from '{full_path}': {e}")
                self.status_update.emit(f"[System Warning: Could not read content of `{relative_filename}`. Assuming new file or using plan only.]")
        else:
            logger.info(f"MC: Original file '{relative_filename}' not found (resolved to '{full_path}'). Treating as new.")
            self.status_update.emit(f"[System Note: File `{relative_filename}` appears to be new.]")
        return content

    def _extract_plan_segment_for_file(self, filename: str, full_plan_text: str) -> Optional[str]:
        if not full_plan_text or not filename:
            return None
        normalized_filename_for_marker = filename.replace("\\", "/")
        start_marker = r"--- FILE_PLAN_START: " + re.escape(normalized_filename_for_marker) + r" ---"
        end_marker = r"--- FILE_PLAN_END: " + re.escape(normalized_filename_for_marker) + r" ---"
        pattern = re.compile(f"{start_marker}(.*?){end_marker}", re.DOTALL | re.IGNORECASE)
        match = pattern.search(full_plan_text)
        if match:
            segment = match.group(1).strip()
            logger.info(f"MC: Successfully extracted plan segment for '{filename}'. Length: {len(segment)}")
            return segment
        else:
            logger.warning(f"MC: Could not find plan segment for '{filename}' using markers in the proto-specification.")
            return None

    def _request_generator_prompt_from_planner(self,
                                               target_filename_for_generator: str,
                                               original_file_content_for_context: Optional[str],
                                               plan_segment_for_this_file: str):
        logger.debug(f"MC: Asking Planner AI to craft prompt for Generator AI regarding '{target_filename_for_generator}'.")
        self.status_update.emit(f"[System: Planner AI is formulating instructions for Coder AI regarding `{target_filename_for_generator}`...]")

        file_op_type = "UPDATE the existing file" if original_file_content_for_context else "CREATE a NEW file"

        prompt_for_planner = (
            f"You are an expert AI assistant responsible for creating precise instructions for a specialized code generation LLM (e.g., CodeLlama, StarCoder2).\n"
            f"The overall user request is: \"{self._original_query_at_start}\"\n"
            f"The current task is to {file_op_type}: `{target_filename_for_generator}`.\n\n"
            f"The high-level plan for this specific file is:\n"
            f"--- PLAN SEGMENT FOR `{target_filename_for_generator}` ---\n"
            f"{plan_segment_for_this_file}\n"
            f"--- END PLAN SEGMENT ---\n\n"
        )

        if original_file_content_for_context:
            max_orig_context_for_planner_prompt = 4000
            display_orig_content = original_file_content_for_context
            if len(original_file_content_for_context) > max_orig_context_for_planner_prompt:
                display_orig_content = original_file_content_for_context[:max_orig_context_for_planner_prompt] + \
                                       "\n... [Original Content Truncated for this Planner Instruction] ..."
            prompt_for_planner += (
                f"CONTEXT: The original content of `{target_filename_for_generator}` (if it exists) is:\n"
                f"```{'python' if target_filename_for_generator.endswith('.py') else ''}\n{display_orig_content}\n```\n\n"
            )
        else:
            prompt_for_planner += f"CONTEXT: The file `{target_filename_for_generator}` is new and should be created from scratch.\n\n"

        prompt_for_planner += (
            f"YOUR TASK: Generate the complete and precise instruction text that will be sent *directly* to the code generation LLM. "
            f"This instruction MUST guide the Coder AI to produce the FULL and correct code for the *entire* file `{target_filename_for_generator}`.\n"
            f"The instruction you generate should:\n"
            f"1. Clearly state the target filename: `{target_filename_for_generator}`.\n"
            f"2. Incorporate the user's original request context: \"{self._original_query_at_start}\" and this file's PLAN SEGMENT.\n"
            f"3. If updating, explicitly include the *complete original content* of `{target_filename_for_generator}` within the instruction for the Coder AI, clearly marked (e.g., inside a specific code block like ```python_original\\n...\\n```). Emphasize preserving unchanged parts perfectly.\n"
            f"4. If creating a new file, state this clearly.\n"
            f"5. Remind the Coder AI that its response MUST be ONLY a single standard Markdown fenced code block. This block MUST be labeled with the programming language followed by the exact filename `{target_filename_for_generator}` (e.g., ```python {target_filename_for_generator}\\n...\\n``` or ```javascript {target_filename_for_generator}\\n...\\n```). "
            f"**CRITICAL EMPHASIS: The Coder AI's response must contain NO other text, no explanations, no summaries, no greetings, no apologies, no conversational elements whatsoever, neither before nor after this single required code block.** "
            f"The Coder AI's entire response must be ONLY the code block itself, starting with ```<language> {target_filename_for_generator} and ending with ```.\n\n"
            f"OUTPUT ONLY THE INSTRUCTION TEXT FOR THE CODER AI. Do not add any conversational preamble or explanation of your own."
        )

        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[prompt_for_planner])]
        self._is_awaiting_llm = True
        self._current_phase = ModPhase.AWAITING_GENERATOR_PROMPT_REFINEMENT
        self.request_llm_call.emit(PLANNER_BACKEND_ID, history_for_llm)

    def _handle_planner_refined_generator_prompt(self, refined_generator_prompt_text: str):
        logger.info(f"MC: Received refined/created prompt for Generator AI (for file '{self._current_target_filename}').")
        self._is_awaiting_llm = False
        self._last_generated_prompt_for_coder = refined_generator_prompt_text.strip()

        if not self._last_generated_prompt_for_coder:
            err_msg = f"Planner AI returned an empty instruction for the Coder AI for file '{self._current_target_filename}'."
            self.modification_error.emit(err_msg)
            self.status_update.emit(f"[System Error: Planner failed to create instructions for Coder AI for `{self._current_target_filename}`. Skipping this file.]")
            QTimer.singleShot(0, self._proceed_to_next_file_or_part) # Skip to next file
            return

        self._request_code_generation_from_coder(self._last_generated_prompt_for_coder)

    def _request_code_generation_from_coder(self, generator_instruction: str):
        logger.debug(f"MC: Requesting code generation from Coder AI for '{self._current_target_filename}'.")
        self.status_update.emit(f"[System: Sending instructions to Coder AI for `{self._current_target_filename}`...]")

        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[generator_instruction])]
        self._is_awaiting_llm = True
        self._current_phase = ModPhase.AWAITING_CODE_GENERATION
        self.request_llm_call.emit(GENERATOR_BACKEND_ID, history_for_llm)

    def process_llm_response(self, backend_id: str, response_message: ChatMessage):
        if not self._is_active or not self._is_awaiting_llm:
            logger.warning(f"MC: process_llm_response from '{backend_id}' called when not active/awaiting. Phase: {self._current_phase}")
            return
        logger.info(f"MC: Processing LLM response from '{backend_id}' during phase '{self._current_phase}'. ID: {response_message.id}")
        response_text = response_message.text.strip()

        try:
            if self._current_phase == ModPhase.AWAITING_PLAN and backend_id == PLANNER_BACKEND_ID:
                self._handle_plan_response(response_text)
            elif self._current_phase == ModPhase.AWAITING_GENERATOR_PROMPT_REFINEMENT and backend_id == PLANNER_BACKEND_ID:
                self._handle_planner_refined_generator_prompt(response_text)
            elif self._current_phase == ModPhase.AWAITING_CODE_GENERATION and backend_id == GENERATOR_BACKEND_ID:
                self._handle_coder_ai_code_response(response_text, response_message)
            else:
                self._is_awaiting_llm = False
                err_msg = f"Unexpected LLM response from '{backend_id}' for phase {self._current_phase}."
                self.modification_error.emit(err_msg)
                self._handle_sequence_end("error_unexpected_response_phase", err_msg)
        except Exception as e:
            self._is_awaiting_llm = False
            logger.exception(f"MC: Error processing LLM response from '{backend_id}': {e}")
            self._handle_sequence_end("error_processing_llm_response", f"Error processing {backend_id} response: {e}")

    def _handle_coder_ai_code_response(self, generated_code_text: str, coder_response_message: ChatMessage):
        logger.info(f"MC: Handling Coder AI's response for code file '{self._current_target_filename}'. Message ID: {coder_response_message.id}")
        self._is_awaiting_llm = False
        if not self._current_target_filename:
            self._handle_sequence_end("error_internal_state", "MC Error: Missing target filename during code generation response.")
            return

        if self._handler.process_llm_code_generation_response(generated_code_text, self._current_target_filename):
            parsed_filename_content_tuple = self._handler.get_last_emitted_filename_and_content()
            if parsed_filename_content_tuple and parsed_filename_content_tuple[0] == self._current_target_filename:
                actual_filename, actual_content = parsed_filename_content_tuple

                if coder_response_message.metadata is None:
                    coder_response_message.metadata = {}
                coder_response_message.metadata["code_block_processed_by_mc"] = True
                coder_response_message.metadata["original_filename_for_viewer"] = actual_filename
                logger.info(f"MC: Flagged ChatMessage ID {coder_response_message.id} as processed by MC for '{actual_filename}'")

                is_new_file = self._current_original_file_content_for_ai is None
                if not is_new_file and actual_content.strip() == (self._current_original_file_content_for_ai or "").strip():
                    self.status_update.emit(f"[System: No effective changes were applied by AI to '{actual_filename}'. Skipping display, proceeding to next.]")
                    QTimer.singleShot(0, self._proceed_to_next_file_or_part)
                    return
                if is_new_file and not actual_content.strip():
                    self.status_update.emit(f"[System: File '{actual_filename}' was planned as new, but AI generated no content. Assuming not needed. Skipping display, proceeding to next.]")
                    QTimer.singleShot(0, self._proceed_to_next_file_or_part)
                    return

                lines = actual_content.splitlines()
                if len(lines) > self.MAX_LINES_BEFORE_SPLIT:
                    logger.info(f"File '{actual_filename}' has {len(lines)} lines, will be queued for split display.")
                    split_point = len(lines) // 2
                    part1_content = "\n".join(lines[:split_point])
                    part2_content = "\n".join(lines[split_point:])
                    self._current_file_parts_queue.append((actual_filename + " (Part 1/2)", part1_content))
                    self._current_file_parts_queue.append((actual_filename + " (Part 2/2)", part2_content))
                else:
                    self._current_file_parts_queue.append((actual_filename, actual_content))

                self.status_update.emit(f"[System: Code for `{actual_filename}` generated. Queued for display.]")
                QTimer.singleShot(0, self._proceed_to_next_file_or_part) # This will now pick up from the queue or move to next file
            else:
                mismatch_info = f"Expected '{self._current_target_filename}', MH provided '{parsed_filename_content_tuple[0] if parsed_filename_content_tuple else 'None'}'."
                logger.error(f"MC: Internal Mismatch after MH parsing. {mismatch_info}")
                self.status_update.emit(f"[System Warning: Issue processing Coder AI output for `{self._current_target_filename}`. {mismatch_info}. Skipping this file.]")
                QTimer.singleShot(0, self._proceed_to_next_file_or_part) # Skip to next file on parsing mismatch
        # If process_llm_code_generation_response returns False, _handle_mh_parsing_error is triggered by MH signal

    def process_user_input(self, user_command: str):
        if not self._is_active: return
        if self._is_awaiting_llm: # This means it's in AWAITING_PLAN, AWAITING_GENERATOR_PROMPT_REFINEMENT, or AWAITING_CODE_GENERATION
            self.status_update.emit("[System: Please wait for the current AI processing to complete.]")
            return

        logger.info(f"MC: User input during active sequence: '{user_command[:30]}...' (Phase: {self._current_phase})")
        command_lower = user_command.lower().strip()

        if command_lower in ["cancel", "stop", "abort"]:
            self._handle_sequence_end("cancelled_by_user", "User cancelled the modification process.")
            return

        if self._current_phase == ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION:
            if command_lower in ["accept", "done", "looks good", "ok", "okay", "proceed", "complete", "finalize"]:
                self._handle_sequence_end("completed_by_user_acceptance", "User accepted all generated files.")
            else: # Overall feedback for a new refinement cycle
                self.status_update.emit(f"[System: Received overall feedback: \"{user_command[:50]}...\". Requesting full re-plan...]")
                self._original_query = f"The initial request was: '{self._original_query_at_start}'. Based on the generated files, the user now provides this overall feedback for refinement: '{user_command}'"
                self._is_awaiting_llm = False
                self._planned_files_list = []
                self._current_file_index = -1
                self._current_file_parts_queue = [] # Clear queue for re-plan
                self._request_initial_plan_from_planner() # Request a new plan based on the overall feedback
        else:
            # If not in the final feedback phase, and not awaiting LLM, it means file generation is ongoing but paused (e.g., error).
            # Or it's an unexpected state.
            self.status_update.emit(f"[System: Currently processing files. Please wait until all files are generated to provide overall feedback or type 'cancel'. If stuck, 'cancel' might be needed.]")

    @pyqtSlot(str)
    def _handle_mh_parsing_error(self, error_message: str):
        if not self._is_active: return
        logger.error(f"MC: Received parsing error from MH for file '{self._current_target_filename}': {error_message}")
        self.status_update.emit(f"[System Error: Coder AI output for `{self._current_target_filename}` was not in the expected format. This file may be incomplete or incorrect. Skipping this file and proceeding.]")
        self._is_awaiting_llm = False # Ensure user can interact if needed, though it auto-proceeds
        QTimer.singleShot(0, self._proceed_to_next_file_or_part) # Proceed to next file/part

    def _parse_files_to_modify_list(self, response_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
        marker = "FILES_TO_MODIFY:"
        marker_pos = response_text.find(marker)
        if marker_pos == -1: return None, f"Marker '{marker}' not found in Planner response."
        list_str_start = marker_pos + len(marker)
        potential_list_str = response_text[list_str_start:].strip()
        first_line_after_marker = potential_list_str.split('\n', 1)[0].strip()
        list_match = re.search(r"(\[.*?\])", first_line_after_marker)
        if list_match:
            list_str_for_eval = list_match.group(1)
        elif first_line_after_marker.startswith('[') and first_line_after_marker.endswith(']'):
            list_str_for_eval = first_line_after_marker
        else:
            return None, "FILES_TO_MODIFY list not found or not correctly formatted with brackets on the first line after the marker."
        try:
            parsed_list = ast.literal_eval(list_str_for_eval)
            if not isinstance(parsed_list, list):
                return None, "Parsed data for FILES_TO_MODIFY is not a list."
            cleaned_list = [str(f).strip().replace("\\", "/") for f in parsed_list if isinstance(f, (str, int, float))]
            return [f_item for f_item in cleaned_list if f_item], None
        except (ValueError, SyntaxError, TypeError) as e:
            return None, f"Error parsing FILES_TO_MODIFY list string '{list_str_for_eval}': {e}"

    def cancel_sequence(self, reason: str = "cancelled_externally"):
        if not self._is_active: return
        self._handle_sequence_end(reason, f"Sequence cancelled: {reason}")

    def _handle_sequence_end(self, reason: str, details: Optional[str] = None):
        if not self._is_active and reason != "error_processing_llm_response":
             logger.debug(f"MC: _handle_sequence_end called but not active. Reason: {reason}.")
             return

        log_message = f"MC: Ending sequence. Reason: {reason}."
        if details: log_message += f" Details: {details}"
        logger.info(log_message)

        original_query_summary = self._original_query_at_start[:75] + '...' if self._original_query_at_start and len(self._original_query_at_start) > 75 else self._original_query_at_start or "User's request"
        self.modification_sequence_complete.emit(reason, original_query_summary)
        self._reset_state()
