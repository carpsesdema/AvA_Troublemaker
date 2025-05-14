# core/user_input_handler.py
import logging
import uuid  # Make sure uuid is imported
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from core.models import ChatMessage, USER_ROLE
# --- CORRECTED IMPORT ---
from core.user_input_processor import UserInputProcessor, ProcessResult
# --- END CORRECTED IMPORT ---
from core.project_context_manager import ProjectContextManager
from core.modification_coordinator import ModificationCoordinator

# Conditional import for ProjectSummaryCoordinator
if TYPE_CHECKING:
    from core.project_summary_coordinator import ProjectSummaryCoordinator

logger = logging.getLogger(__name__)


class UserInputHandler(QObject):
    # Existing signals
    normal_chat_request_ready = pyqtSignal(list)  # List[ChatMessage]
    modification_sequence_start_requested = pyqtSignal(str, list, str, str)
    modification_user_input_received = pyqtSignal(str, str)
    processing_error_occurred = pyqtSignal(str)

    # --- NEW SIGNAL for displaying user commands without full LLM processing ---
    user_command_for_display_only = pyqtSignal(ChatMessage)

    # --- END NEW SIGNAL ---

    def __init__(self,
                 user_input_processor: UserInputProcessor,
                 project_context_manager: ProjectContextManager,
                 modification_coordinator: Optional[ModificationCoordinator],
                 project_summary_coordinator: Optional['ProjectSummaryCoordinator'],  # Added
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([user_input_processor, project_context_manager]):
            err_msg = "UserInputHandler requires UserInputProcessor and ProjectContextManager."
            logger.critical(err_msg)
            raise ValueError(err_msg)

        self._uip = user_input_processor
        self._pcm = project_context_manager
        self._mc = modification_coordinator
        self._psc = project_summary_coordinator  # Store it

        if self._psc:
            logger.info("UserInputHandler initialized with ProjectSummaryCoordinator.")
        else:
            logger.warning(
                "UserInputHandler initialized WITHOUT ProjectSummaryCoordinator. Manual summary command disabled.")
        logger.info("UserInputHandler initialized.")

    def handle_user_message(self,
                            text: str,
                            image_data: List[Dict[str, Any]],
                            focus_paths: Optional[List[str]],
                            rag_available: bool,
                            rag_initialized_for_project: bool):
        user_query_text_raw = text.strip()
        is_mod_active = self._mc and self._mc.is_active() if self._mc else False
        current_pid = self._pcm.get_active_project_id() if self._pcm else None
        if not current_pid:
            logger.error("UserInputHandler: Could not determine current_project_id from ProjectContextManager.")
            self.processing_error_occurred.emit("Internal error: Active project context not found.")
            return

        try:
            proc_result = self._uip.process(
                user_query_text=user_query_text_raw,
                image_data=image_data or [],
                is_modification_active=is_mod_active,
                current_project_id=current_pid,
                focus_paths=focus_paths,
                rag_available=rag_available,
                rag_initialized=rag_initialized_for_project
            )
        except Exception as e_uip:
            logger.exception(f"UserInputHandler: UserInputProcessor encountered an error: {e_uip}")
            self.processing_error_occurred.emit(f"Error processing your input: {e_uip}")
            return

        action = proc_result.action_type
        payload = proc_result.prompt_or_history

        logger.info(f"UserInputHandler: UIP returned action '{action}'.")

        if action == "NORMAL_CHAT":
            if isinstance(proc_result.prompt_or_history, list) and proc_result.prompt_or_history and isinstance(
                    proc_result.prompt_or_history[0], ChatMessage):
                logger.debug(
                    f"UserInputHandler: NORMAL_CHAT. Emitting clean UI message from raw text: '{user_query_text_raw[:100]}...'")
                ui_message_parts = [user_query_text_raw] + (image_data or [])
                ui_chat_message_for_display = ChatMessage(role=USER_ROLE, parts=ui_message_parts)
                self.normal_chat_request_ready.emit([ui_chat_message_for_display])
            else:
                logger.error(
                    f"UserInputHandler: NORMAL_CHAT action received invalid payload: {proc_result.prompt_or_history}")
                self.processing_error_occurred.emit("Internal error preparing chat message.")

        elif action == "START_MODIFICATION":
            logger.info("UserInputHandler: Handling START_MODIFICATION.")
            original_query = proc_result.original_query or ""
            context_for_mc = proc_result.original_context or ""
            focus_prefix_for_mc = proc_result.original_focus_prefix or ""
            self.modification_sequence_start_requested.emit(
                original_query, image_data or [], context_for_mc, focus_prefix_for_mc
            )

        elif action in ["NEXT_MODIFICATION", "REFINE_MODIFICATION", "COMPLETE_MODIFICATION"]:
            user_command_for_mc = payload
            if isinstance(user_command_for_mc, str):
                self.modification_user_input_received.emit(user_command_for_mc, action)
            else:
                logger.error(
                    f"UserInputHandler: Modification action '{action}' received non-string payload: {user_command_for_mc}")
                self.processing_error_occurred.emit(f"Internal error processing modification command for '{action}'.")

        elif action == "REQUEST_PROJECT_SUMMARY":
            project_id_for_summary = str(payload) if payload else None
            original_query_for_ui = proc_result.original_query

            if not project_id_for_summary:
                logger.error("UserInputHandler: REQUEST_PROJECT_SUMMARY action received invalid project_id in payload.")
                self.processing_error_occurred.emit("Internal error: Could not determine project for summary.")
                return

            # 1. Emit user's command for UI display ONLY
            if original_query_for_ui:
                ui_message_parts = [original_query_for_ui]
                # Create a new ChatMessage with a unique ID for UI display
                ui_chat_message = ChatMessage(
                    role=USER_ROLE,
                    parts=ui_message_parts,
                    id=str(uuid.uuid4())  # Generate a new unique ID
                )
                self.user_command_for_display_only.emit(ui_chat_message)
                logger.debug(
                    f"Emitted user's summary command (ID: {ui_chat_message.id}, Text: '{original_query_for_ui[:50]}...') for UI display ONLY.")

            # 2. Call ProjectSummaryCoordinator directly
            if self._psc:
                logger.info(
                    f"UserInputHandler: Calling ProjectSummaryCoordinator to generate summary for project '{project_id_for_summary}'.")
                self._psc.generate_project_summary(project_id_for_summary)
            else:
                logger.error("ProjectSummaryCoordinator not available. Cannot process summary request.")
                self.processing_error_occurred.emit("Project summary feature is currently unavailable.")

        elif action == "NO_ACTION":
            logger.info("UserInputHandler: UserInputProcessor determined no action required.")
        else:
            logger.error(f"UserInputHandler: Unknown action type from UserInputProcessor: {action}")
            self.processing_error_occurred.emit(f"Unknown internal action: {action}")