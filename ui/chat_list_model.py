# ui/chat_list_model.py
# UPDATED - Only emit dataChanged on finalization, not per chunk. Added updateMessage method.
# UPDATED - Added LoadingStatusRole and method to update loading state by message ID.
# NEW - Added find_message_row_by_id method.

import logging
from typing import List, Optional, Any, Union

from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt, QObject

# --- Core Model Imports ---
try:
    from core.models import ChatMessage
    from core.message_enums import MessageLoadingState
except ImportError: # Fallback for different environment setups during development
    try:
        from ..core.models import ChatMessage # type: ignore
        from ..core.message_enums import MessageLoadingState # type: ignore
    except ImportError as e:
        logging.critical(f"ChatListModel: Failed to import core.models or core.message_enums: {e}")
        # Define dummy classes if import fails, so the rest of the app can try to load
        class ChatMessage: pass # type: ignore
        from enum import Enum, auto
        class MessageLoadingState(Enum): IDLE = auto(); LOADING = auto(); COMPLETED = auto(); ERROR = auto() # type: ignore

logger = logging.getLogger(__name__)

# Define custom roles
ChatMessageRole = Qt.ItemDataRole.UserRole + 1
LoadingStatusRole = Qt.ItemDataRole.UserRole + 2 # New role for loading indicator state

class ChatListModel(QAbstractListModel):
    """
    A QAbstractListModel to manage the list of ChatMessage objects
    for display in a QListView. Includes support for message loading states.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._messages: List[ChatMessage] = []
        logger.info("ChatListModel initialized.")

    # --- QAbstractListModel Interface ---

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Returns the number of messages in the model."""
        return 0 if parent.isValid() else len(self._messages)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Returns the data for a given index and role."""
        if not index.isValid() or not (0 <= index.row() < len(self._messages)):
            return None

        message = self._messages[index.row()]

        if role == ChatMessageRole:
            return message # Return the full ChatMessage object
        elif role == Qt.ItemDataRole.DisplayRole:
            # Basic display text, delegate handles rich rendering
            return f"[{message.role}] {message.text[:50]}..."
        elif role == LoadingStatusRole: # Handle the new role
            return message.loading_state

        return None

    # --- Custom Methods for Data Manipulation ---

    def addMessage(self, message: ChatMessage):
        """Adds a single message to the end of the model."""
        if not isinstance(message, ChatMessage):
            logger.error(f"Attempted to add invalid type to ChatListModel: {type(message)}")
            return

        logger.debug(f"Model: Adding message (Role: {message.role}, ID: {message.id}, LoadingState: {message.loading_state.name if hasattr(message, 'loading_state') else 'N/A'})")
        row_to_insert = len(self._messages)
        self.beginInsertRows(QModelIndex(), row_to_insert, row_to_insert)
        self._messages.append(message)
        self.endInsertRows()
        logger.debug(f"Model: Message added. New count: {len(self._messages)}")

    def appendChunkToLastMessage(self, chunk: str):
        """Appends a text chunk to the *internal data* of the last message."""
        if not self._messages:
            logger.warning("Model: Cannot append chunk, message list is empty.")
            return
        if not isinstance(chunk, str):
            logger.warning(f"Model: Invalid chunk type: {type(chunk)}")
            return

        last_message = self._messages[-1]

        current_text = ""
        text_part_index = -1
        for i, part in enumerate(last_message.parts):
            if isinstance(part, str):
                current_text = part
                text_part_index = i
                break
        updated_text = current_text + chunk

        if text_part_index != -1:
            last_message.parts[text_part_index] = updated_text
        else:
            last_message.parts.insert(0, updated_text)

        if last_message.metadata is None: last_message.metadata = {}
        last_message.metadata["is_streaming"] = True

        last_model_idx = self.index(len(self._messages) - 1, 0)
        self.dataChanged.emit(last_model_idx, last_model_idx, [ChatMessageRole])


    def finalizeLastMessage(self):
        """
        Marks the last message as no longer streaming internally.
        """
        if not self._messages:
            logger.warning("Model: Cannot finalize, message list is empty.")
            return

        last_message = self._messages[-1]
        logger.debug(f"Model: Finalizing streaming for last message (ID: {last_message.id}, Role: {last_message.role}).")

        if last_message.metadata and last_message.metadata.get("is_streaming"):
            last_message.metadata["is_streaming"] = False
            model_index = self.index(len(self._messages) -1, 0)
            self.dataChanged.emit(model_index, model_index, [ChatMessageRole])
        else:
            logger.debug("Model: Finalize called but last message wasn't marked as streaming.")

    def updateMessage(self, index: int, message: ChatMessage):
        """Replaces the message at the given index."""
        if not (0 <= index < len(self._messages)):
            logger.error(f"Model: Invalid index {index} provided for updateMessage.")
            return
        if not isinstance(message, ChatMessage):
            logger.error(f"Model: Invalid message type {type(message)} provided for updateMessage.")
            return

        logger.debug(f"Model: Updating message at index {index} (Role: {message.role}, ID: {message.id})")
        # Ensure the loading state of the existing message isn't inadvertently overwritten
        # if the incoming message object doesn't have it set correctly (e.g. if it's a raw new obj)
        # However, ChatManager sends the *updated placeholder*, so its loading_state should be the one from history.
        # If `message` is the updated placeholder, its loading_state will be what was in the model.
        # If `ChatMessageStateHandler` changes it, that will be a separate dataChanged.
        existing_loading_state = self._messages[index].loading_state
        self._messages[index] = message
        if self._messages[index].loading_state == MessageLoadingState.IDLE and \
           existing_loading_state != MessageLoadingState.IDLE and \
           message.role == "model": # If AI model and new state is IDLE, preserve existing
            self._messages[index].loading_state = existing_loading_state

        model_idx = self.index(index, 0)
        # Emit for all roles that might change content or visual state.
        # LoadingStatusRole is important if the message object itself might change it,
        # though ChatMessageStateHandler is the primary driver for loading_state changes.
        self.dataChanged.emit(model_idx, model_idx, [ChatMessageRole, LoadingStatusRole, Qt.ItemDataRole.DisplayRole])

    def update_message_loading_state_by_id(self, message_id: str, new_state: MessageLoadingState) -> bool:
        """Finds a message by its ID and updates its loading_state."""
        if not isinstance(message_id, str) or not message_id:
            logger.warning("Model: update_message_loading_state_by_id called with invalid message_id.")
            return False

        for row in range(len(self._messages)):
            message = self._messages[row]
            if hasattr(message, 'id') and message.id == message_id:
                if hasattr(message, 'loading_state') and message.loading_state != new_state:
                    logger.info(f"Model: Updating loading state for message ID '{message_id}' from {message.loading_state.name} to {new_state.name}.")
                    message.loading_state = new_state
                    model_index = self.index(row, 0)
                    self.dataChanged.emit(model_index, model_index, [LoadingStatusRole])
                    return True
                elif not hasattr(message, 'loading_state'):
                    logger.warning(f"Message ID '{message_id}' found, but it lacks 'loading_state' attribute.")
                    # Initialize it if missing, especially for AI messages
                    if message.role == "model": # MODEL_ROLE constant might not be imported here
                        logger.info(f"Initializing loading_state for message ID '{message_id}' to {new_state.name}.")
                        message.loading_state = new_state
                        model_index = self.index(row, 0)
                        self.dataChanged.emit(model_index, model_index, [LoadingStatusRole])
                        return True
                    return False
                return True # State is already the same, no change needed
        logger.warning(f"Model: Could not find message with ID '{message_id}' to update loading state.")
        return False

    # --- NEW METHOD ---
    def find_message_row_by_id(self, message_id: str) -> Optional[int]:
        """Finds the row index of a message by its ID."""
        if not isinstance(message_id, str) or not message_id:
            logger.warning("Model: find_message_row_by_id called with invalid message_id.")
            return None
        for row, msg in enumerate(self._messages):
            if hasattr(msg, 'id') and msg.id == message_id:
                return row
        return None
    # --- END NEW METHOD ---

    def loadHistory(self, history: List[ChatMessage]):
        """Replaces the entire model content with a new history list."""
        logger.info(f"Model: loadHistory called (Incoming Count: {len(history)})")
        self.beginResetModel()
        self._messages = list(history) # Replace internal list
        self.endResetModel()
        logger.info(f"Model: Model reset complete (Internal Count: {len(self._messages)})")

    def clearMessages(self):
        """Removes all messages from the model."""
        logger.info("Model: Clearing all messages.")
        self.beginResetModel()
        self._messages = []
        self.endResetModel()
        logger.info("Model: Messages cleared.")

    def getMessage(self, row: int) -> Optional[ChatMessage]:
        """Safely retrieves a message by row index."""
        if 0 <= row < len(self._messages):
            return self._messages[row]
        return None

    def getAllMessages(self) -> List[ChatMessage]:
        """Returns a copy of the internal message list."""
        return list(self._messages)