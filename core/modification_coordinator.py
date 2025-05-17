# core/modification_coordinator.py
import logging
import ast # For parsing FILES_TO_MODIFY list
import re  # For parsing plan segments
import os  # For file operations

from typing import List, Optional, Dict, Any, Tuple # Ensure Tuple is imported

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

# Assuming these imports are correct from your project structure
try:
    from .modification_handler import ModificationHandler
    from .backend_coordinator import BackendCoordinator
    from .project_context_manager import ProjectContextManager # Might be needed for project root path if not passed directly
    from .models import ChatMessage, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from utils import constants # For backend IDs if used, and other constants
except ImportError as e:
    logging.critical(f"ModificationCoordinator: Failed to import core components: {e}")
    # Define fallback types for type hinting and basic script operation
    ModificationHandler = type("ModificationHandler", (object,), {})
    BackendCoordinator = type("BackendCoordinator", (object,), {})
    ProjectContextManager = type("ProjectContextManager", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    constants = type("constants", (object,), {}) # Add a fallback for constants # type: ignore
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
    AWAITING_USER_CONFIRMATION = "AWAITING_USER_CONFIRMATION" # Waiting for user after code is shown

class ModificationCoordinator(QObject):
    # Signals to ChatManager/MainWindow
    request_llm_call = pyqtSignal(str, list)  # backend_id, history_to_send
    file_ready_for_display = pyqtSignal(str, str) # filename, content (AI-generated full content)
    modification_sequence_complete = pyqtSignal(str, str) # reason, original_query_summary
    modification_error = pyqtSignal(str)      # error_message (for UI error display)
    status_update = pyqtSignal(str)           # General status message for chat UI

    def __init__(self,
                 modification_handler: ModificationHandler,
                 backend_coordinator: BackendCoordinator,
                 project_context_manager: ProjectContextManager, # Keep for project root path
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([modification_handler, backend_coordinator, project_context_manager]):
            logger.critical("MC Init: Missing critical dependencies.")
            raise ValueError("ModificationCoordinator requires valid handler, backend_coord, and project_manager.")

        self._handler = modification_handler
        self._backend_coordinator = backend_coordinator
        self._project_context_manager = project_context_manager # Store it

        self._is_active: bool = False
        self._is_awaiting_llm: bool = False
        self._current_phase: str = ModPhase.IDLE

        # Information about the overall modification request
        self._original_query: Optional[str] = None
        self._original_context_from_rag: Optional[str] = None # RAG context for the whole operation
        self._original_focus_prefix: Optional[str] = None     # Project root path (from user input or PCM)

        # Plan-related state
        self._full_planner_proto_specification: Optional[str] = None # Stores the Planner's detailed natural language plan
        self._planned_files_list: List[str] = []      # List of relative file paths to modify/create

        # Per-file iteration state
        self._current_file_index: int = -1
        self._current_target_filename: Optional[str] = None # Relative path of the file currently being processed
        self._current_original_file_content_for_ai: Optional[str] = None # Full original content for AI context
        self._current_plan_segment_for_file: Optional[str] = None # Part of proto-spec for current file
        self._last_generated_prompt_for_coder: Optional[str] = None # The prompt that was sent to the Coder AI
        self._user_feedback_for_current_file: Optional[str] = None # To store feedback for refinement

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
        self._original_context_from_rag = None
        self._original_focus_prefix = None
        self._full_planner_proto_specification = None
        self._planned_files_list = []
        self._current_file_index = -1
        self._current_target_filename = None
        self._current_original_file_content_for_ai = None
        self._current_plan_segment_for_file = None
        self._last_generated_prompt_for_coder = None
        self._user_feedback_for_current_file = None
        if self._handler:
            self._handler.cancel_modification() # Ensure handler is also reset

    def is_active(self) -> bool:
        """Returns True if a modification sequence is currently active."""
        return self._is_active

    def is_awaiting_llm_response(self) -> bool:
        """Returns True if the coordinator is active and awaiting an LLM response."""
        return self._is_active and self._is_awaiting_llm

    def start_sequence(self, query: str, context_from_rag: str, focus_prefix: str):
        if self._is_active:
            logger.warning("MC: start_sequence called while active. Resetting previous sequence.")
            self._reset_state()

        logger.info(f"MC: Starting new modification sequence. Query: '{query[:50]}...'")
        self._is_active = True
        self._original_query = query
        self._original_context_from_rag = context_from_rag
        self._original_focus_prefix = focus_prefix # This should be the project root absolute path
        self._is_awaiting_llm = False # Will be set true by request methods

        self._request_initial_plan_from_planner()

    def _request_initial_plan_from_planner(self):
        logger.debug("MC: Requesting initial detailed plan (proto-specification) from Planner AI.")
        prompt_text = (
            f"You are an expert AI system planner. Your task is to create a detailed step-by-step plan "
            f"to implement the following user request:\n"
            f"USER REQUEST: \"{self._original_query}\"\n\n"
            f"ASSOCIATED PROJECT CONTEXT (from RAG):\n{self._original_context_from_rag or 'N/A'}\n\n"
            f"PROJECT ROOT FOCUS (base path for relative file paths): {self._original_focus_prefix or 'N/A'}\n\n"
            f"Your response MUST include two main parts:\n"
            f"1. A machine-parseable line identifying all files to be modified or created. This line MUST start "
            f"   EXACTLY with 'FILES_TO_MODIFY: ' followed by a Python-style list of relative file paths "
            f"   (e.g., FILES_TO_MODIFY: ['src/file1.py', 'app/new_module/file2.py']). "
            f"   If no files need changes, use FILES_TO_MODIFY: []. This line should appear first or very early.\n"
            f"2. After that line, provide a DETAILED natural language proto-specification. For EACH file listed in FILES_TO_MODIFY, "
            f"   you MUST include a dedicated section clearly marked like this:\n"
            f"   --- FILE_PLAN_START: path/to/filename.ext ---\n"
            f"   [Detailed logic, changes, new functions/classes for this specific file]\n"
            f"   --- FILE_PLAN_END: path/to/filename.ext ---\n"
            f"   The proto-specification should also cover:\n"
            f"   - The refined user goal based on the request.\n"
            f"   - Key RAG insights derived from the RAG context that informed your plan.\n"
            f"   - Descriptions of any new data structures or interfaces if applicable.\n"
            f"   - Potential constraints, edge cases, or important considerations for the implementation.\n"
            f"   - Any outstanding questions for the user, if absolutely necessary for planning (prefer to make sound assumptions if possible).\n\n"
            f"Ensure the plan is comprehensive enough for another AI to generate the code for each file based on its segment of the plan and original file content (if modifying).\n"
            f"Generate the plan now."
        )
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[prompt_text])]
        self._is_awaiting_llm = True
        self._current_phase = ModPhase.AWAITING_PLAN
        self.status_update.emit("[System: Asking Planner AI to create a detailed modification plan...]")
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
            self.status_update.emit("[System: Planner AI indicates no files require modification based on the request.]")
            self._handle_sequence_end("completed_no_files", "Planner found no files to modify.")
            return

        logger.info(f"MC: Plan successfully parsed. Files to modify/create: {self._planned_files_list}")
        files_str_display = ", ".join([f"`{f}`" for f in self._planned_files_list])
        self.status_update.emit(f"[System: Planner AI will process: {files_str_display}]")

        self._current_file_index = -1 # Reset for the start of new file processing
        QTimer.singleShot(0, self._proceed_to_next_file_step) # Use QTimer for safety

    def _proceed_to_next_file_step(self):
        self._is_awaiting_llm = False
        self._current_file_index += 1
        self._user_feedback_for_current_file = None

        if not (0 <= self._current_file_index < len(self._planned_files_list)):
            self._complete_sequence_if_done()
            return

        self._current_target_filename = self._planned_files_list[self._current_file_index]
        self.status_update.emit(
            f"[System: Processing file {self._current_file_index + 1}/{len(self._planned_files_list)}: `{self._current_target_filename}`.]"
        )

        self._current_original_file_content_for_ai = self._read_original_file_content(self._current_target_filename)
        self._current_plan_segment_for_file = self._extract_plan_segment_for_file(
            self._current_target_filename,
            self._full_planner_proto_specification or ""
        )
        if not self._current_plan_segment_for_file:
            logger.warning(f"MC: Could not extract plan segment for '{self._current_target_filename}'. Using generic instructions.")
            self._current_plan_segment_for_file = f"Implement changes for '{self._current_target_filename}' based on the overall user request: \"{self._original_query}\"."
            self.status_update.emit(f"[System Warning: No detailed plan segment found for `{self._current_target_filename}`. Coder AI will use general plan.]")

        self._request_generator_prompt_from_planner(
            target_filename_for_generator=self._current_target_filename,
            original_file_content_for_context=self._current_original_file_content_for_ai,
            plan_segment_for_this_file=self._current_plan_segment_for_file,
            user_feedback_for_refinement=None
        )

    def _read_original_file_content(self, relative_filename: str) -> Optional[str]:
        content: Optional[str] = None
        full_path: Optional[str] = None

        if self._original_focus_prefix and os.path.isdir(self._original_focus_prefix):
            norm_relative_path = relative_filename.lstrip('/').lstrip('\\')
            full_path = os.path.normpath(os.path.join(self._original_focus_prefix, norm_relative_path))
        elif os.path.isabs(relative_filename): # If AI gives absolute path (less ideal)
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
                                               plan_segment_for_this_file: str,
                                               user_feedback_for_refinement: Optional[str]):
        logger.debug(f"MC: Asking Planner AI to craft prompt for Generator AI regarding '{target_filename_for_generator}'. Feedback: {bool(user_feedback_for_refinement)}")
        self.status_update.emit(f"[System: Planner AI is formulating instructions for Coder AI regarding `{target_filename_for_generator}`...]")

        file_op_type = "UPDATE the existing file" if original_file_content_for_context else "CREATE a NEW file"
        if user_feedback_for_refinement:
            file_op_type = f"REFINE the previously generated code for {file_op_type}"

        prompt_for_planner = (
            f"You are an expert AI assistant responsible for creating precise instructions for a specialized code generation LLM (e.g., CodeLlama, StarCoder2).\n"
            f"The overall user request is: \"{self._original_query}\"\n"
            f"The current task is to {file_op_type}: `{target_filename_for_generator}`.\n\n"
            f"The high-level plan for this specific file is:\n"
            f"--- PLAN SEGMENT FOR `{target_filename_for_generator}` ---\n"
            f"{plan_segment_for_this_file}\n"
            f"--- END PLAN SEGMENT ---\n\n"
        )

        if user_feedback_for_refinement:
            prompt_for_planner += (
                f"The user has provided the following feedback on the Coder AI's previous attempt for this file:\n"
                f"--- USER FEEDBACK ---\n"
                f"{user_feedback_for_refinement}\n"
                f"--- END USER FEEDBACK ---\n\n"
                f"The Coder AI was last given this prompt (which may have been generated by you):\n"
                f"--- PREVIOUS PROMPT TO CODER AI ---\n"
                f"{self._last_generated_prompt_for_coder or 'No previous prompt recorded.'}\n"
                f"--- END PREVIOUS PROMPT ---\n\n"
            )

        if original_file_content_for_context:
            max_orig_context_for_planner_prompt = 4000 # Or some other reasonable limit
            display_orig_content = original_file_content_for_context
            if len(original_file_content_for_context) > max_orig_context_for_planner_prompt:
                display_orig_content = original_file_content_for_context[:max_orig_context_for_planner_prompt] + \
                                       "\n... [Original Content Truncated for this Planner Instruction] ..."
            prompt_for_planner += (
                f"CONTEXT: The original content of `{target_filename_for_generator}` (if it exists) is:\n"
                # Determine language for syntax highlighting in prompt (basic heuristic)
                # This is for the prompt *to the planner*, not the final coder prompt's markdown label.
                f"```{'python' if target_filename_for_generator.endswith('.py') else ''}\n{display_orig_content}\n```\n\n"
            )
        else:
            prompt_for_planner += f"CONTEXT: The file `{target_filename_for_generator}` is new and should be created from scratch.\n\n"

        # --- BEGIN MODIFIED SECTION ---
        prompt_for_planner += (
            f"YOUR TASK: Generate the complete and precise instruction text that will be sent *directly* to the code generation LLM. "
            f"This instruction MUST guide the Coder AI to produce the FULL and correct code for the *entire* file `{target_filename_for_generator}`.\n"
            f"The instruction you generate should:\n"
            f"1. Clearly state the target filename: `{target_filename_for_generator}`.\n"
            f"2. Incorporate the user's original request context: \"{self._original_query}\" and this file's PLAN SEGMENT.\n"
            f"3. If updating, explicitly include the *complete original content* of `{target_filename_for_generator}` within the instruction for the Coder AI, clearly marked (e.g., inside a specific code block like ```python_original\\n...\\n```). Emphasize preserving unchanged parts perfectly.\n"
            f"4. If creating a new file, state this clearly.\n"
            f"5. If refining based on user feedback, clearly address the feedback and integrate it with previous instructions/context.\n"
            f"6. Remind the Coder AI that its response MUST be ONLY a single standard Markdown fenced code block. This block MUST be labeled with the programming language followed by the exact filename `{target_filename_for_generator}` (e.g., ```python {target_filename_for_generator}\\n...\\n``` or ```javascript {target_filename_for_generator}\\n...\\n```). "
            f"**CRITICAL EMPHASIS: The Coder AI's response must contain NO other text, no explanations, no summaries, no greetings, no apologies, no conversational elements whatsoever, neither before nor after this single required code block.** "
            f"The Coder AI's entire response must be ONLY the code block itself, starting with ```<language> {target_filename_for_generator} and ending with ```.\n\n"
            f"OUTPUT ONLY THE INSTRUCTION TEXT FOR THE CODER AI. Do not add any conversational preamble or explanation of your own."
        )
        # --- END MODIFIED SECTION ---

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
            self._handle_sequence_end("error_empty_generator_prompt", err_msg)
            return

        self._request_code_generation_from_coder(self._last_generated_prompt_for_coder)

    def _request_code_generation_from_coder(self, generator_instruction: str):
        logger.debug(f"MC: Requesting code generation from Coder AI for '{self._current_target_filename}'.")
        self.status_update.emit(f"[System: Sending instructions to Coder AI for `{self._current_target_filename}`...]")

        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[generator_instruction])]
        self._is_awaiting_llm = True
        self._current_phase = ModPhase.AWAITING_CODE_GENERATION
        self.request_llm_call.emit(GENERATOR_BACKEND_ID, history_for_llm)

    def process_llm_response(self, backend_id: str, response_message: ChatMessage): # response_message is ChatMessage
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
                # <<< MODIFICATION START: Pass response_message to _handle_coder_ai_code_response >>>
                self._handle_coder_ai_code_response(response_text, response_message)
                # <<< MODIFICATION END >>>
            else:
                self._is_awaiting_llm = False
                err_msg = f"Unexpected LLM response from '{backend_id}' for phase {self._current_phase}."
                self.modification_error.emit(err_msg)
                self._handle_sequence_end("error_unexpected_response_phase", err_msg)
        except Exception as e:
            self._is_awaiting_llm = False
            logger.exception(f"MC: Error processing LLM response from '{backend_id}': {e}")
            self._handle_sequence_end("error_processing_llm_response", f"Error processing {backend_id} response: {e}")

    # <<< MODIFICATION START: Update signature and add flagging logic >>>
    def _handle_coder_ai_code_response(self, generated_code_text: str, coder_response_message: ChatMessage):
    # <<< MODIFICATION END >>>
        logger.info(f"MC: Handling Coder AI's response for code file '{self._current_target_filename}'. Message ID: {coder_response_message.id}")
        self._is_awaiting_llm = False
        if not self._current_target_filename:
            self._handle_sequence_end("error_internal_state", "MC Error: Missing target filename during code generation response.")
            return

        if self._handler.process_llm_code_generation_response(generated_code_text, self._current_target_filename):
            parsed_filename_content_tuple = self._handler.get_last_emitted_filename_and_content()
            if parsed_filename_content_tuple and parsed_filename_content_tuple[0] == self._current_target_filename:
                actual_filename, actual_content = parsed_filename_content_tuple

                # <<< MODIFICATION START: Flag the ChatMessage object >>>
                if coder_response_message.metadata is None:
                    coder_response_message.metadata = {}
                coder_response_message.metadata["code_block_processed_by_mc"] = True
                coder_response_message.metadata["original_filename_for_viewer"] = actual_filename
                logger.info(f"MC: Flagged ChatMessage ID {coder_response_message.id} as processed by MC for '{actual_filename}'")
                # <<< MODIFICATION END >>>

                self.file_ready_for_display.emit(actual_filename, actual_content)
                self.status_update.emit(f"[System: Code for `{self._current_target_filename}` received. Review in Code Viewer and provide feedback or type 'next'.]")
                self._current_phase = ModPhase.AWAITING_USER_CONFIRMATION
            else:
                mismatch_info = f"Expected '{self._current_target_filename}', MH provided '{parsed_filename_content_tuple[0] if parsed_filename_content_tuple else 'None'}'."
                logger.error(f"MC: Internal Mismatch after MH parsing. {mismatch_info}")
                self.status_update.emit(f"[System Warning: Issue processing Coder AI output for `{self._current_target_filename}`. {mismatch_info} Check logs. You can try 'refine' or 'next'.]")
                self._current_phase = ModPhase.AWAITING_USER_CONFIRMATION
        # If process_llm_code_generation_response returns False, _handle_mh_parsing_error will be triggered by MH signal

    def process_user_input(self, user_command: str):
        if not self._is_active: return
        if self._is_awaiting_llm:
            self.status_update.emit("[System: Please wait for the current AI step to complete.]")
            return

        logger.info(f"MC: User input during active sequence: '{user_command[:30]}...' (Phase: {self._current_phase})")
        command_lower = user_command.lower().strip()

        if command_lower in ["cancel", "stop", "abort"]:
            self._handle_sequence_end("cancelled_by_user", "User cancelled the modification process.")
            return

        if self._current_phase == ModPhase.AWAITING_USER_CONFIRMATION:
            if command_lower in ["next", "ok", "okay", "continue", "yes", "proceed", "go", "next file", "looks good", "accept", "lg"]:
                self.status_update.emit(f"[System: Changes for `{self._current_target_filename}` accepted.]")
                QTimer.singleShot(0, self._proceed_to_next_file_step)
            else:
                if not self._current_target_filename: # Should not happen here
                    self._handle_sequence_end("error_refinement_no_target", "MC Error: Cannot refine, no current target file.")
                    return

                self._user_feedback_for_current_file = user_command
                self.status_update.emit(f"[System: Refining `{self._current_target_filename}` based on feedback: \"{user_command[:50]}...\"]")

                if self._current_plan_segment_for_file is None:
                    logger.error(f"MC Error: Plan segment for '{self._current_target_filename}' is missing. Cannot refine.")
                    self.modification_error.emit(f"Internal error: Missing plan details for refining '{self._current_target_filename}'.")
                    return

                # Assert types for mypy before lambda
                assert self._current_target_filename is not None
                assert self._current_plan_segment_for_file is not None

                QTimer.singleShot(0, lambda: self._request_generator_prompt_from_planner(
                    target_filename_for_generator=self._current_target_filename, # type: ignore
                    original_file_content_for_context=self._current_original_file_content_for_ai,
                    plan_segment_for_this_file=self._current_plan_segment_for_file, # type: ignore
                    user_feedback_for_refinement=self._user_feedback_for_current_file
                ))
        else:
            self.status_update.emit(f"[System: Command '{user_command[:20]}' not applicable in current phase '{self._current_phase}'. Awaiting AI or 'next'/'cancel'.]")

    @pyqtSlot(str)
    def _handle_mh_parsing_error(self, error_message: str):
        if not self._is_active: return
        logger.error(f"MC: Received parsing error from MH for file '{self._current_target_filename}': {error_message}")
        self.status_update.emit(f"[System Error: Coder AI output for `{self._current_target_filename}` was not in the expected format. You can try 'refine' with more specific instructions or 'next' to skip this file.]")
        self._current_phase = ModPhase.AWAITING_USER_CONFIRMATION
        self._is_awaiting_llm = False

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

        original_query_summary = self._original_query[:75] + '...' if self._original_query and len(self._original_query) > 75 else self._original_query or "User's request"
        self.modification_sequence_complete.emit(reason, original_query_summary)
        self._reset_state()

    def _complete_sequence_if_done(self):
        if self._current_file_index >= len(self._planned_files_list) and self._planned_files_list:
            self.status_update.emit("[System: All planned files have been processed.]")
            self._handle_sequence_end("completed", "All planned files processed.")
        elif not self._planned_files_list and self._current_phase != ModPhase.AWAITING_PLAN:
            pass