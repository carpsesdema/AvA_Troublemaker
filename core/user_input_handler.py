# core/user_input_handler.py
# UPDATED: Modified modification_sequence_start_requested signal and its emission.

import logging
from typing import List, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

from core.models import ChatMessage, USER_ROLE # USER_ROLE is used for creating the UI message
from .user_input_processor import UserInputProcessor, ProcessResult
from .project_context_manager import ProjectContextManager
from .modification_coordinator import ModificationCoordinator

logger = logging.getLogger(__name__)


class UserInputHandler(QObject):
    normal_chat_request_ready = pyqtSignal(list)  # List[ChatMessage] - For normal chat

    # --- MODIFICATION START: Signal signature updated ---
    modification_sequence_start_requested = pyqtSignal(str, list, str, str)
    # Emits: original_query_text, image_data_list, context_for_mc, focus_prefix_for_mc
    # --- MODIFICATION END ---

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
                            image_data: List[Dict[str, Any]], # This is the raw image_data from input bar
                            focus_paths: Optional[List[str]],
                            rag_available: bool,
                            rag_initialized_for_project: bool):
        user_query_text_raw = text.strip()
        is_mod_active = self._mc and self._mc.is_active() if self._mc else False
        current_pid = self._pcm.get_active_project_id()

        try:
            proc_result = self._uip.process(
                user_query_text=user_query_text_raw,
                image_data=image_data or [], # Pass the raw image_data to UIP
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
        # For NORMAL_CHAT, payload is the augmented message List[ChatMessage]
        # For START_MODIFICATION, these are now specific fields in ProcessResult
        # For REFINE/NEXT_MODIFICATION, payload is the user_command string

        logger.info(f"UserInputHandler: UIP returned action '{action}'.")

        if action == "NORMAL_CHAT":
            payload = proc_result.prompt_or_history # This is List[ChatMessage] from UIP
            if isinstance(payload, list) and payload and isinstance(payload[0], ChatMessage):
                # For NORMAL_CHAT, the UIP already prepared the message for the backend (with RAG).
                # For the UI, we want to show the user's *original* input.
                ui_message_parts = [user_query_text_raw] + (image_data or [])
                ui_chat_message = ChatMessage(role=USER_ROLE, parts=ui_message_parts)

                # ChatManager will use the augmented message from payload[0] for the LLM call,
                # but it needs the clean ui_chat_message to add to history for display.
                # We can emit both, or ChatManager can reconstruct.
                # For now, let's emit the clean UI message. ChatManager will then take this,
                # and also needs to be aware of the augmented prompt from UIP.
                # This interaction needs refinement in ChatManager.

                # ---> Current behavior based on existing CM:
                # CM adds the emitted message to history.
                # Then CM prepares history for backend, including the just-added message.
                # So, the augmented message from UIP needs to be the one added to history *by CM*.
                # The `normal_chat_request_ready` signal should probably carry the *original* user input
                # and ChatManager should then get the augmented version from UIP or reconstruct.

                # Let's adjust: `normal_chat_request_ready` will emit the *original* user message elements.
                # ChatManager's slot `_handle_uih_normal_chat_request` will construct the ChatMessage
                # and then use the *augmented* version for the LLM call.
                # UIP's `prompt_or_history` for NORMAL_CHAT IS the augmented message.

                # For now, to keep it simple, we emit the augmented message from UIP.
                # ChatManager will add this to history. The UI will render its text part.
                # This means the user sees the RAG context in their own bubble if RAG was added.
                # This is not ideal for UI, but simpler for now.
                # A better approach: UIP returns original text AND augmented text.
                # UIH emits original text. CM adds original to history for UI, uses augmented for LLM.
                # For this iteration, let's assume `payload[0]` is what CM should use.

                # --- Reverting to simpler logic for this step: Emit the clean message for UI.
                # --- ChatManager will handle the augmented prompt separately if needed.
                logger.debug(
                    f"UserInputHandler: NORMAL_CHAT. Emitting clean UI message from raw text: '{user_query_text_raw[:100]}...'")
                ui_message_parts = [user_query_text_raw] + (image_data or [])
                ui_chat_message = ChatMessage(role=USER_ROLE, parts=ui_message_parts)
                self.normal_chat_request_ready.emit([ui_chat_message]) # List containing one message
            else:
                logger.error(f"UserInputHandler: NORMAL_CHAT action received invalid payload: {proc_result.prompt_or_history}")
                self.processing_error_occurred.emit("Internal error preparing chat message.")

        # --- MODIFICATION START: Handle START_MODIFICATION differently ---
        elif action == "START_MODIFICATION":
            logger.info("UserInputHandler: Handling START_MODIFICATION.")
            # Extract details from ProcessResult
            original_query = proc_result.original_query or ""
            context_for_mc = proc_result.original_context or ""
            focus_prefix_for_mc = proc_result.original_focus_prefix or ""
            # image_data is already available from the handle_user_message parameters

            logger.debug("Emitting modification_sequence_start_requested with all necessary data.")
            self.modification_sequence_start_requested.emit(
                original_query,       # User's raw query text
                image_data or [],     # List of image data dicts
                context_for_mc,       # RAG/context string for MC
                focus_prefix_for_mc   # Focus prefix string for MC
            )
            # DO NOT emit normal_chat_request_ready here.
        # --- MODIFICATION END ---

        elif action in ["NEXT_MODIFICATION", "REFINE_MODIFICATION", "COMPLETE_MODIFICATION"]:
            # For these, `prompt_or_history` from UIP is the user_command string
            user_command_for_mc = proc_result.prompt_or_history
            if isinstance(user_command_for_mc, str):
                 self.modification_user_input_received.emit(user_command_for_mc, action)
            else:
                logger.error(f"UserInputHandler: Modification action '{action}' received non-string payload: {user_command_for_mc}")
                self.processing_error_occurred.emit(f"Internal error processing modification command for '{action}'.")


        elif action == "NO_ACTION":
            logger.info("UserInputHandler: UserInputProcessor determined no action required.")

        else:
            logger.error(f"UserInputHandler: Unknown action type from UserInputProcessor: {action}")
            self.processing_error_occurred.emit(f"Unknown internal action: {action}")