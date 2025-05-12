# core/user_input_handler.py
import logging
from typing import List, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

from core.models import ChatMessage, USER_ROLE
from .user_input_processor import UserInputProcessor, ProcessResult
from .project_context_manager import ProjectContextManager
from .modification_coordinator import ModificationCoordinator

logger = logging.getLogger(__name__)


class UserInputHandler(QObject):
    normal_chat_request_ready = pyqtSignal(list)  # List[ChatMessage]
    modification_sequence_start_requested = pyqtSignal(str, str, str)  # query, context, focus_prefix
    modification_user_input_received = pyqtSignal(str, str)  # user_command, action_type
    processing_error_occurred = pyqtSignal(str)  # error_message

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

    def handle_user_message(self,
                            text: str,
                            image_data: List[Dict[str, Any]],
                            focus_paths: Optional[List[str]],
                            rag_available: bool,
                            rag_initialized_for_project: bool):
        user_query_text_raw = text.strip()
        is_mod_active = self._mc and self._mc.is_active() if self._mc else False
        current_pid = self._pcm.get_active_project_id()

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
        payload = proc_result.prompt_or_history  # For NORMAL_CHAT, this is the augmented message

        logger.info(f"UserInputHandler: UIP returned action '{action}'.")

        if action == "NORMAL_CHAT":
            if isinstance(payload, list) and payload and isinstance(payload[0], ChatMessage):
                # For NORMAL_CHAT, the payload[0] is the message already prepared by UIP,
                # which includes RAG context in its text.
                # To display a clean user message, we create a new one.

                logger.debug(
                    f"UserInputHandler: NORMAL_CHAT. Creating clean UI message from raw text: '{user_query_text_raw[:100]}...'")
                ui_message_parts = [user_query_text_raw] + (image_data or [])
                ui_chat_message = ChatMessage(role=USER_ROLE, parts=ui_message_parts)

                # We need to pass BOTH the clean UI message and the augmented LLM message to ChatManager
                # For now, let's just emit the clean one. ChatManager will use history for LLM.
                # This means the LLM will get the clean version from history if we don't adjust ChatManager.
                # This is a TODO: ChatManager needs to use augmented prompt for LLM call for normal chat.
                self.normal_chat_request_ready.emit([ui_chat_message])

                # For ChatManager to use the augmented prompt, it would need `payload[0]`.
                # One way is to emit it too, or have ChatManager reconstruct.
                # For now, the LLM will receive the clean prompt via history for normal chat.

            else:
                logger.error(f"UserInputHandler: NORMAL_CHAT action received invalid payload: {payload}")
                self.processing_error_occurred.emit("Internal error preparing chat message.")

        elif action == "START_MODIFICATION":
            logger.info("UserInputHandler: Handling START_MODIFICATION. Preparing clean user message for display.")
            original_query = proc_result.original_query or ""
            original_context = proc_result.original_context or ""
            original_focus_prefix = proc_result.original_focus_prefix or ""

            # For the UI, just use the original_query and any image_data.
            user_message_parts_for_ui = [original_query] + (image_data or [])
            user_chat_message_for_ui = ChatMessage(role=USER_ROLE, parts=user_message_parts_for_ui)

            logger.debug(
                f"Emitting normal_chat_request_ready for user's clean modification prompt: '{user_chat_message_for_ui.text[:100]}...'")
            self.normal_chat_request_ready.emit([user_chat_message_for_ui])

            logger.debug("Emitting modification_sequence_start_requested with full context for MC.")
            self.modification_sequence_start_requested.emit(
                original_query,
                original_context,
                original_focus_prefix
            )

        elif action in ["NEXT_MODIFICATION", "REFINE_MODIFICATION", "COMPLETE_MODIFICATION"]:
            user_command_for_mc = user_query_text_raw
            self.modification_user_input_received.emit(user_command_for_mc, action)

        elif action == "NO_ACTION":
            logger.info("UserInputHandler: UserInputProcessor determined no action required.")

        else:
            logger.error(f"UserInputHandler: Unknown action type from UserInputProcessor: {action}")
            self.processing_error_occurred.emit(f"Unknown internal action: {action}")