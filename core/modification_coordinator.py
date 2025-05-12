# core/modification_coordinator.py
# This is the FULL content for this file, with only the initial plan prompt modified.

import logging
import ast
import re
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

try:
    from .modification_handler import ModificationHandler
    from .backend_coordinator import BackendCoordinator
    from .project_context_manager import ProjectContextManager
    from .models import ChatMessage, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
except ImportError as e:
    logging.critical(f"ModificationCoordinator: Failed to import core components: {e}")
    ModificationHandler = type("ModificationHandler", (object,), {}) # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {}) # type: ignore
    ProjectContextManager = type("ProjectContextManager", (object,), {}) # type: ignore
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    USER_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "system", "error" # type: ignore

logger = logging.getLogger(__name__)

# Backend ID Constants (matching ChatManager if defined there, or define here)
# Ensure these are correct for your setup.
DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default" # Not directly used in this class but good for context
PLANNER_BACKEND_ID = "gemini_planner"
GENERATOR_BACKEND_ID = "ollama_generator"

class ModPhase:
    IDLE = "IDLE"
    AWAITING_PLAN = "AWAITING_PLAN"
    AWAITING_GEMINI_REFINEMENT = "AWAITING_GEMINI_REFINEMENT"
    AWAITING_CODE_GENERATION = "AWAITING_CODE_GENERATION"
    AWAITING_USER_CONFIRMATION = "AWAITING_USER_CONFIRMATION"

class ModificationCoordinator(QObject):
    request_llm_call = pyqtSignal(str, list) # backend_id, history_to_send
    file_ready_for_display = pyqtSignal(str, str) # filename, content
    modification_sequence_complete = pyqtSignal(str) # reason
    modification_error = pyqtSignal(str) # error_message
    status_update = pyqtSignal(str) # message_for_user_chat
    codeGeneratedAndSummaryNeeded = pyqtSignal(str, str, str)

    def __init__(self,
                 modification_handler: ModificationHandler,
                 backend_coordinator: BackendCoordinator,
                 project_context_manager: ProjectContextManager,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([modification_handler, backend_coordinator, project_context_manager]):
            logger.critical("ModificationCoordinator requires valid handler, backend_coord, and project_manager.")
            raise ValueError("Missing critical dependencies for ModificationCoordinator.")

        self._handler = modification_handler
        self._backend_coordinator = backend_coordinator
        self._project_context_manager = project_context_manager
        self._is_active: bool = False
        self._is_awaiting_llm: bool = False
        self._current_phase: str = ModPhase.IDLE
        self._original_query: Optional[str] = None
        self._original_context: Optional[str] = None
        self._original_focus_prefix: Optional[str] = None
        self._planned_files_list: List[str] = []
        self._current_file_index: int = -1
        self._current_target_filename: Optional[str] = None
        self._last_standard_instruction_for_refinement: Optional[str] = None
        self._last_refined_instruction_for_generation: Optional[str] = None

        self._connect_handler_signals()
        logger.info("ModificationCoordinator initialized for Gemini-Refined Prompt Pipeline™.")

    def _connect_handler_signals(self):
        if not self._handler: return
        try:
            self._handler.code_file_ready.connect(self._handle_handler_code_ready)
            self._handler.status_message_ready.connect(self._handle_handler_status_message)
            self._handler.modification_parsing_error.connect(self._handle_handler_parsing_error)
            logger.debug("ModificationCoordinator connected to ModificationHandler signals.")
        except Exception as e:
            logger.exception(f"Error connecting ModificationHandler signals in Coordinator: {e}")

    def is_active(self) -> bool:
        return self._is_active

    def is_awaiting_llm_response(self) -> bool:
        return self._is_active and self._is_awaiting_llm

    def start_sequence(self, query: str, context: str, focus_prefix: str):
        if self._is_active:
            logger.warning("MC: start_sequence called while active. Resetting.")
            self._reset_state()
        logger.info("MC: Starting new modification sequence.")
        self._is_active = True
        self._original_query = query
        self._original_context = context
        self._original_focus_prefix = focus_prefix
        self._is_awaiting_llm = False
        self._request_initial_plan()

    def process_llm_response(self, backend_id: str, response_message: ChatMessage):
        if not self._is_active or not self._is_awaiting_llm:
            logger.warning(
                f"MC: process_llm_response from '{backend_id}' called when not active or not awaiting. Phase: {self._current_phase}")
            return

        logger.info(f"MC: Processing LLM response from '{backend_id}' during phase '{self._current_phase}'.")
        self._is_awaiting_llm = False
        response_text = response_message.text.strip()

        try:
            if self._current_phase == ModPhase.AWAITING_PLAN and backend_id == PLANNER_BACKEND_ID:
                self._handle_plan_response(response_text)
            elif self._current_phase == ModPhase.AWAITING_GEMINI_REFINEMENT and backend_id == PLANNER_BACKEND_ID:
                self._handle_gemini_refinement_response(response_text)
                return # <<< Your existing fix for Bug 2
            elif self._current_phase == ModPhase.AWAITING_CODE_GENERATION and backend_id == GENERATOR_BACKEND_ID:
                generated_code_str = response_text
                coder_instructions_str = self._last_refined_instruction_for_generation or "No specific instructions provided."
                target_filename_str = self._current_target_filename or "Unknown file"
                logger.info(f"MC: Code generation response received for '{target_filename_str}'. Emitting summary request.")
                self.codeGeneratedAndSummaryNeeded.emit(generated_code_str, coder_instructions_str, target_filename_str)
                self._handle_code_generation_response(generated_code_str)
            else:
                logger.error(f"MC: Received LLM response from '{backend_id}' in unexpected phase '{self._current_phase}'.")
                self._is_awaiting_llm = False
                self._handle_sequence_end("error_unexpected_response_phase",
                                          f"Unexpected LLM response for phase {self._current_phase} from {backend_id}")
        except Exception as e:
            logger.exception(f"MC: Error processing LLM response from '{backend_id}': {e}")
            self._is_awaiting_llm = False
            self._handle_sequence_end("error_processing_llm_response", f"Error processing {backend_id} response: {e}")

    def process_llm_error(self, backend_id: str, error_message: str):
        if not self._is_active: return
        logger.error(
            f"MC: Received LLM error from '{backend_id}' during phase '{self._current_phase}': {error_message}")
        self._is_awaiting_llm = False
        QTimer.singleShot(10, lambda: self._handle_sequence_end(f"error_backend_{backend_id}", f"LLM Error ({backend_id}): {error_message}"))

    def process_user_input(self, user_command: str):
        if not self._is_active:
            logger.warning(f"MC: process_user_input ('{user_command[:20]}...') called but not active.")
            return
        if self._is_awaiting_llm:
            self.status_update.emit("[System: Please wait for the current AI step to complete.]")
            return

        logger.info(f"MC: Processing user input: '{user_command[:20]}...' in phase '{self._current_phase}'")
        command_lower = user_command.lower()

        try:
            if command_lower in ["cancel", "stop", "abort"]:
                self._handle_sequence_end("cancelled_by_user")
            elif self._current_phase == ModPhase.AWAITING_USER_CONFIRMATION:
                if command_lower in ["next", "ok", "okay", "continue", "yes", "proceed", "go", "next file"]:
                    self._current_file_index += 1
                    if self._current_file_index < len(self._planned_files_list):
                        QTimer.singleShot(10, self._proceed_to_next_file_step)
                    else:
                        self.status_update.emit("[System: All planned files processed.]")
                        self._handle_sequence_end("completed")
                else:
                    if not self._current_target_filename:
                        self._handle_sequence_end("error_refinement_no_target", "Cannot refine: no target file context.")
                        return
                    standard_refinement_instruction = self._handler.prepare_codellama_refinement_instruction(
                        self._current_target_filename, user_command,
                        self._last_refined_instruction_for_generation or self._last_standard_instruction_for_refinement or "No previous instruction."
                    )
                    QTimer.singleShot(10, lambda: self._request_prompt_refinement(standard_refinement_instruction, self._current_target_filename or "unknown_file"))
            else:
                self.status_update.emit(
                    f"[System: Command '{user_command[:20]}' not applicable in current phase '{self._current_phase}'.]")
        except Exception as e:
            logger.exception(f"MC: Error processing user input '{user_command[:20]}...':")
            self._handle_sequence_end("error_user_input", f"Failed to process command: {e}")

    def cancel_sequence(self, reason: str = "cancelled_externally"):
        if not self._is_active: return
        logger.info(f"MC: Sequence cancellation requested (Reason: {reason}).")
        self._handle_sequence_end(reason)

    def _request_initial_plan(self):
        logger.debug("MC: Requesting initial plan from planner.")
        # --- START OF MODIFIED PROMPT ---
        prompt_text = (
            "You are a file planning assistant. Your sole task is to identify files for modification based on the user's request. "
            "Analyze the following request and context:\n"
            f"User's main request: \"{self._original_query or 'Not provided.'}\"\n"
            f"Associated project context: \"{self._original_context or 'N/A'}\"\n"
            f"Specific focus prefix: \"{self._original_focus_prefix or 'N/A'}\"\n\n"
            "CRITICAL INSTRUCTION: Your entire response MUST be a single line starting with the exact text 'FILES_TO_MODIFY: ' "
            "followed by a Python-style list of strings representing the relative file paths to be modified. "
            "Do not include any other text, explanations, apologies, greetings, or summaries whatsoever, neither before nor after this single required line.\n\n"
            "EXAMPLE OF VALID RESPONSE:\n"
            "FILES_TO_MODIFY: ['src/main.py', 'app/utils.py', 'new_module/core.py']\n\n"
            "IF NO FILES NEED MODIFICATION, YOUR RESPONSE MUST BE:\n"
            "FILES_TO_MODIFY: []\n\n"
            "Provide your response now:"
        )
        # --- END OF MODIFIED PROMPT ---
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[prompt_text])]
        self.request_llm_call.emit(PLANNER_BACKEND_ID, history_for_llm)
        self._current_phase = ModPhase.AWAITING_PLAN
        self._is_awaiting_llm = True
        self.status_update.emit("[System: Requesting modification plan from Planner AI...]")

    def _request_prompt_refinement(self, standard_instruction: str, target_filename: str):
        logger.debug(f"MC: Requesting prompt refinement from planner for '{target_filename}'.")
        self._last_standard_instruction_for_refinement = standard_instruction
        self._current_target_filename = target_filename
        refinement_prompt_text = (
            f"Original user request for context: \"{self._original_query}\"\n"
            f"We are currently focused on modifying the file: \"{target_filename}\"\n"
            "The overall plan involves these files: " + (f"{self._planned_files_list}" if self._planned_files_list else "[Not yet determined or N/A]") + "\n\n"
            "Below is a standard instruction intended for a code generation LLM (e.g., CodeLlama) "
            f"to generate or update the content for \"{target_filename}\":\n"
            "--- STANDARD INSTRUCTION START ---\n"
            f"{standard_instruction}\n"
            "--- STANDARD INSTRUCTION END ---\n\n"
            "Your task is to REFINE this standard instruction. Make it as clear, complete, and effective as possible for the code generation LLM. "
            "Ensure the refined instruction explicitly guides the LLM to produce the full, correct code for the specified file, adhering to the user's original request and the overall plan.\n"
            "IMPORTANT: Your response should ONLY be the refined instruction itself. Do NOT include any preamble, conversational fluff, apologies, or self-references like \"Okay, here's the refined instruction:\". Just the instruction text."
        )
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[refinement_prompt_text])]
        self.request_llm_call.emit(PLANNER_BACKEND_ID, history_for_llm)
        self._current_phase = ModPhase.AWAITING_GEMINI_REFINEMENT
        self._is_awaiting_llm = True
        self.status_update.emit(f"[System: Asking Planner AI to refine prompt for '{target_filename}'...]")

    def _request_code_generation_from_generator(self, refined_instruction: str):
        logger.debug(f"MC: Requesting code generation from generator for '{self._current_target_filename}'.")
        self._last_refined_instruction_for_generation = refined_instruction
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[refined_instruction])]
        self.request_llm_call.emit(GENERATOR_BACKEND_ID, history_for_llm)
        self._current_phase = ModPhase.AWAITING_CODE_GENERATION
        self._is_awaiting_llm = True
        self.status_update.emit(f"[System: Requesting code for '{self._current_target_filename}' from Generator AI...]")

    def _handle_plan_response(self, response_text: str):
        logger.debug("MC: Handling planner's response for file list.")
        parsed_list, error = self._parse_files_to_modify_list(response_text)
        if parsed_list is not None:
            self._planned_files_list = parsed_list
            if not self._planned_files_list:
                self.status_update.emit("[System: Planner AI indicates no files require modification.]")
                QTimer.singleShot(10, lambda: self._handle_sequence_end("completed_no_files"))
                return
            logger.info(f"MC: Plan received. Files to modify: {self._planned_files_list}")
            self.status_update.emit(f"[System: Plan received. Files: {', '.join(self._planned_files_list)}]")
            self._current_file_index = 0
            QTimer.singleShot(10, self._proceed_to_next_file_step)
        else:
            err_msg = f"Failed to parse file list from Planner AI: {error}. Response: '{response_text[:200]}...'"
            self.modification_error.emit(err_msg)
            QTimer.singleShot(10, lambda: self._handle_sequence_end("error_plan_parsing", err_msg))

    def _handle_gemini_refinement_response(self, response_text: str):
        logger.debug("MC: Handling planner's response for prompt refinement.")
        refined_instruction = response_text.strip()
        if not refined_instruction:
            err_msg = "Planner AI returned an empty refinement for the prompt."
            self.modification_error.emit(err_msg)
            QTimer.singleShot(10, lambda: self._handle_sequence_end("error_empty_refinement", err_msg))
            return
        logger.info(f"MC: Prompt refinement received for '{self._current_target_filename}'.")
        self.status_update.emit(f"[System: Planner AI refined prompt for '{self._current_target_filename}'.]")
        QTimer.singleShot(10, lambda: self._request_code_generation_from_generator(refined_instruction))

    def _handle_code_generation_response(self, response_text: str): # response_text is already stripped
        logger.debug(f"MC: Handling generator's response for code file '{self._current_target_filename}'.")
        if not self._current_target_filename:
            self._handle_sequence_end("error_internal_state", "Missing target filename during code generation response.")
            return
        # ModificationHandler will parse the response_text and emit code_file_ready or error
        if self._handler.process_llm_code_generation_response(response_text, self._current_target_filename):
            # Success (code_file_ready will be emitted by handler)
            pass
        else:
            # Error (modification_parsing_error will be emitted by handler)
            # Sequence end will be triggered by _handle_handler_parsing_error
            pass

    def _proceed_to_next_file_step(self):
        if not (0 <= self._current_file_index < len(self._planned_files_list)):
            logger.error("MC: _proceed_to_next_file_step called with invalid index or empty plan.")
            self._handle_sequence_end("error_internal_state", "Invalid file index for next step.")
            return
        self._current_target_filename = self._planned_files_list[self._current_file_index]
        self.status_update.emit(
            f"[System: Preparing to process file {self._current_file_index + 1}/{len(self._planned_files_list)}: '{self._current_target_filename}'...]")
        standard_instruction = self._handler.prepare_standard_codellama_instruction(
            self._current_target_filename,
            self._original_query or "User request not available.",
            self._planned_files_list
        )
        self._request_prompt_refinement(standard_instruction, self._current_target_filename)

    def _parse_files_to_modify_list(self, response_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
        marker = "FILES_TO_MODIFY:"
        logger.debug(f"Attempting to parse plan. Raw response: '{response_text[:500]}...'") # From combined.txt
        marker_pos = response_text.find(marker)
        if marker_pos == -1:
            return None, f"Marker '{marker}' not found." # From combined.txt
        list_str_start = marker_pos + len(marker)
        potential_list_str = response_text[list_str_start:].strip()
        match = re.match(r"(\[.*?\])", potential_list_str) # From combined.txt
        if match:
            list_str_for_eval = match.group(1)
        else:
            first_line_after_marker = potential_list_str.split('\n', 1)[0].strip() # From combined.txt
            if not (first_line_after_marker.startswith('[') and first_line_after_marker.endswith(']')):
                return None, "List does not appear to be correctly formatted with brackets on the first line." # From combined.txt
            list_str_for_eval = first_line_after_marker
        try:
            parsed_list = ast.literal_eval(list_str_for_eval) # From combined.txt
            if not isinstance(parsed_list, list):
                return None, "Parsed data is not a list." # From combined.txt
            cleaned_list = [str(f).strip().replace("\\", "/") for f in parsed_list if isinstance(f, (str, int, float))] # From combined.txt
            cleaned_list = [f for f in cleaned_list if f] # From combined.txt
            return cleaned_list, None
        except (ValueError, SyntaxError, TypeError) as e:
            return None, f"Error parsing list string '{list_str_for_eval}': {e}" # From combined.txt

    @pyqtSlot(str, str)
    def _handle_handler_code_ready(self, filename: str, content: str):
        logger.debug(f"MC: Handler emitted code_file_ready for '{filename}'.")
        if self._is_active:
            if filename == self._current_target_filename:
                self.file_ready_for_display.emit(filename, content)
                self._current_phase = ModPhase.AWAITING_USER_CONFIRMATION
            else:
                logger.warning(f"MC: Handler emitted code for '{filename}', but expected '{self._current_target_filename}'. Forwarding anyway.")
                self.file_ready_for_display.emit(filename, content)

    @pyqtSlot(str)
    def _handle_handler_status_message(self, message: str):
        if self._is_active: self.status_update.emit(message)

    @pyqtSlot(str)
    def _handle_handler_parsing_error(self, error_message: str):
        if not self._is_active: return
        logger.error(f"MC: Received parsing error from handler: {error_message}")
        self._handle_sequence_end("error_code_parsing", f"Generator AI Output Error: {error_message}")

    def _handle_sequence_end(self, reason: str, error_details: Optional[str] = None):
        if not self._is_active: return
        logger.info(f"MC: Ending sequence (Reason: {reason}).")
        if error_details:
            logger.error(f"  Error details for sequence end: {error_details}")
            # self.modification_error.emit(error_details) # Already emitted from other places if it's a parsing error
        self.modification_sequence_complete.emit(reason)
        self._reset_state()

    def _reset_state(self):
        logger.debug("MC: Resetting internal state.")
        self._is_active = False
        self._is_awaiting_llm = False
        self._current_phase = ModPhase.IDLE
        self._original_query = None
        self._original_context = None
        self._original_focus_prefix = None
        self._planned_files_list = []
        self._current_file_index = -1
        self._current_target_filename = None
        self._last_standard_instruction_for_refinement = None
        self._last_refined_instruction_for_generation = None
        if self._handler:
            self._handler.cancel_modification()