# core/chat_message_state_handler.py

import logging
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSlot

# --- Import necessary components and enums ---
try:
    from .backend_coordinator import BackendCoordinator
    from ui.chat_list_model import ChatListModel # Assuming ChatListModel is in ui module
    from .message_enums import MessageLoadingState
    from .models import ChatMessage # For type hinting if needed
except ImportError:
    # Fallbacks for development or if structure is different
    BackendCoordinator = type("BackendCoordinator", (QObject,), { # type: ignore
        "stream_started": type("pyqtSignal", (object,), {"connect": lambda x: None}),
        "response_completed": type("pyqtSignal", (object,), {"connect": lambda x: None}),
        "response_error": type("pyqtSignal", (object,), {"connect": lambda x: None}),
    })
    ChatListModel = type("ChatListModel", (QObject,), { # type: ignore
        "update_message_loading_state_by_id": lambda x, y: False
    })
    from enum import Enum, auto
    class MessageLoadingState(Enum): IDLE=auto(); LOADING=auto(); COMPLETED=auto(); ERROR=auto() # type: ignore
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    logging.error("ChatMessageStateHandler: Failed to import dependencies correctly.")


logger = logging.getLogger(__name__)

class ChatMessageStateHandler(QObject):
    """
    Handles transitions of AI message loading states in the ChatListModel
    based on signals from the BackendCoordinator.
    """

    def __init__(self,
                 model: ChatListModel,
                 backend_coordinator: BackendCoordinator,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(model, ChatListModel):
            raise TypeError("ChatMessageStateHandler requires a valid ChatListModel instance.")
        if not isinstance(backend_coordinator, BackendCoordinator):
            raise TypeError("ChatMessageStateHandler requires a valid BackendCoordinator instance.")

        self._model = model
        self._backend_coordinator = backend_coordinator

        self._connect_signals()
        logger.info("ChatMessageStateHandler initialized and connected to BackendCoordinator signals.")

    def _connect_signals(self):
        """Connect to signals from BackendCoordinator."""
        # These signals now include request_id as the first parameter
        self._backend_coordinator.stream_started.connect(self._handle_stream_started)
        self._backend_coordinator.response_completed.connect(self._handle_response_completed)
        self._backend_coordinator.response_error.connect(self._handle_response_error)

    @pyqtSlot(str) # request_id
    def _handle_stream_started(self, request_id: str):
        """
        When BackendCoordinator signals a stream has started for a request_id.
        Ensures the corresponding AI message in the model is marked as LOADING.
        ChatManager should have already added the placeholder message with this ID
        and possibly set it to LOADING. This handler makes sure it is.
        """
        logger.info(f"StateHandler: Stream started for request_id '{request_id}'. Setting state to LOADING.")
        # It's possible ChatManager already set it to LOADING. This is a confirmation/backup.
        # If the message doesn't exist (which would be an error in ChatManager's logic),
        # update_message_loading_state_by_id will log a warning.
        self._model.update_message_loading_state_by_id(request_id, MessageLoadingState.LOADING)

    @pyqtSlot(str, ChatMessage, dict) # request_id, completed_message, usage_stats
    def _handle_response_completed(self, request_id: str, completed_message: ChatMessage, usage_stats: dict):
        """
        When BackendCoordinator signals a response is complete for a request_id.
        Updates the corresponding AI message in the model to COMPLETED.
        """
        logger.info(f"StateHandler: Response completed for request_id '{request_id}'. Setting state to COMPLETED.")
        # The ChatListModel's update_message_loading_state_by_id will find the
        # message by its 'id' (which matches request_id) and update its 'loading_state'.
        success = self._model.update_message_loading_state_by_id(request_id, MessageLoadingState.COMPLETED)
        if not success:
            logger.warning(f"StateHandler: Failed to find message with ID '{request_id}' to mark as COMPLETED.")
            # At this point, ChatManager has already handled the content of the message.
            # This handler is purely for the loading state.

    @pyqtSlot(str, str) # request_id, error_message
    def _handle_response_error(self, request_id: str, error_message: str):
        """
        When BackendCoordinator signals an error for a request_id.
        Updates the corresponding AI message in the model to COMPLETED (to stop animation)
        or potentially ERROR if a specific error icon is desired.
        """
        logger.info(f"StateHandler: Response error for request_id '{request_id}'. Error: '{error_message}'. Setting state to COMPLETED (or ERROR).")
        # For now, let's set it to COMPLETED to ensure the loading animation stops.
        # If we want a distinct error icon, we could change this to MessageLoadingState.ERROR
        # and the delegate would need to handle drawing that.
        success = self._model.update_message_loading_state_by_id(request_id, MessageLoadingState.COMPLETED) # Or MessageLoadingState.ERROR
        if not success:
            logger.warning(f"StateHandler: Failed to find message with ID '{request_id}' to mark as ERRORED/COMPLETED.")
        # ChatManager has already handled displaying the error content.