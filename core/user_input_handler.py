# core/user_input_handler.py
import logging
from typing import List, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from .user_input_processor import UserInputProcessor, ProcessResult
from .project_context_manager import ProjectContextManager
from .modification_coordinator import ModificationCoordinator
from utils import constants

logger = logging.getLogger(__name__)


class UserInputHandler(QObject):
    normal_chat_request_ready = pyqtSignal(list)
    modification_sequence_start_requested = pyqtSignal(str, str, str)
    modification_user_input_received = pyqtSignal(str, str)
    processing_error_occurred = pyqtSignal(str)

    def __init__(self,
                 user_input_processor: UserInputProcessor,
                 project_context_manager: ProjectContextManager,
                 modification_coordinator: Optional[ModificationCoordinator],
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([user_input_processor, project_context_manager]):
            err_msg = "UserInputHandler requires UserInputProcessor and ProjectContextManager."
            logger.critical(err_msg)
            raise ValueError(err_msg)

        self._uip = user_input_processor
        self._pcm = project_context_manager
        self._mc = modification_coordinator
        logger.info("UserInputHandler initialized.")

    # --- UPDATED SIGNATURE to accept new arguments ---
    def handle_user_message(self,
                            text: str,
                            image_data: List[Dict[str, Any]],
                            focus_paths: Optional[List[str]],
                            rag_available: bool,
                            rag_initialized_for_project: bool):
        # --- END UPDATED SIGNATURE ---
        """
        Primary entry point for processing a user's message.
        It uses UserInputProcessor and then emits the appropriate signal.
        """
        user_query_text_raw = text.strip()

        is_mod_active = self._mc and self._mc.is_active() if self._mc else False
        current_pid = self._pcm.get_active_project_id()  # PCM ensures this returns a valid string

        try:
            # Now call UserInputProcessor with all the necessary arguments
            proc_result = self._uip.process(
                user_query_text=user_query_text_raw,
                image_data=image_data or [],  # Ensure it's a list
                is_modification_active=is_mod_active,
                current_project_id=current_pid,
                focus_paths=focus_paths,  # Pass through
                rag_available=rag_available,  # Pass through
                rag_initialized=rag_initialized_for_project  # Pass through
            )
        except Exception as e_uip:
            logger.exception(f"UserInputHandler: UserInputProcessor encountered an error: {e_uip}")
            self.processing_error_occurred.emit(f"Error processing your input: {e_uip}")
            return

        action = proc_result.action_type
        payload = proc_result.prompt_or_history

        logger.info(f"UserInputHandler: UIP returned action '{action}'.")

        if action == "NORMAL_CHAT":
            if isinstance(payload, list) and payload and isinstance(payload[0], ChatMessage):
                self.normal_chat_request_ready.emit(payload)
            else:
                logger.error(f"UserInputHandler: NORMAL_CHAT action received invalid payload: {payload}")
                self.processing_error_occurred.emit("Internal error preparing chat message.")

        elif action == "START_MODIFICATION":
            self.modification_sequence_start_requested.emit(
                proc_result.original_query or "",
                proc_result.original_context or "",
                proc_result.original_focus_prefix or ""
            )

        elif action in ["NEXT_MODIFICATION", "REFINE_MODIFICATION", "COMPLETE_MODIFICATION"]:
            user_command_for_mc = user_query_text_raw
            if action == "COMPLETE_MODIFICATION":
                user_command_for_mc = "complete"
            elif action == "NEXT_MODIFICATION" and user_query_text_raw.lower() in self._uip._NEXT_COMMANDS:
                user_command_for_mc = "next"
            self.modification_user_input_received.emit(user_command_for_mc, action)

        elif action == "NO_ACTION":
            logger.info("UserInputHandler: UserInputProcessor determined no action required.")

        else:
            logger.error(f"UserInputHandler: Unknown action type from UserInputProcessor: {action}")
            self.processing_error_occurred.emit(f"Unknown internal action: {action}")