# /// ui/chat_display_area.py
# SynaChat/ui/chat_display_area.py
# UPDATED - Added update_message_in_model slot and reverted ResizeMode

import logging
from typing import List, Dict, Any, Optional

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListView, QAbstractItemView, QSizePolicy,
    QMenu, QApplication # Added QMenu, QApplication for context menu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QModelIndex, QPoint, pyqtSignal # Added QPoint, pyqtSignal
from PyQt6.QtGui import QAction # Added QAction for context menu

# --- Local Imports ---
from core.models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE # Imported SYSTEM_ROLE, ERROR_ROLE
# Import the new Model and Delegate
from .chat_list_model import ChatListModel, ChatMessageRole
from .chat_item_delegate import ChatItemDelegate

logger = logging.getLogger(__name__)

class ChatDisplayArea(QWidget):
    """
    Manages the chat message display area using QListView with a custom model and delegate.
    Receives signals to update the ChatListModel.
    """
    textCopied = pyqtSignal(str, str) # Signal to notify that text has been copied (message, color)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatDisplayAreaWidget")

        # --- Core Components ---
        self.chat_list_view: Optional[QListView] = None
        self.chat_list_model: Optional[ChatListModel] = None
        self.chat_item_delegate: Optional[ChatItemDelegate] = None

        self._init_ui()
        self._connect_model_signals() # Connect signals from the model

    def _init_ui(self):
        """Initialize the UI elements."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.chat_list_view = QListView(self)
        self.chat_list_view.setObjectName("ChatListView") # For QSS styling
        self.chat_list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chat_list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection) # Usually no selection needed
        # --- REVERTED ResizeMode ---
        self.chat_list_view.setResizeMode(QListView.ResizeMode.Adjust) # Default is Adjust
        logger.info(">>> DISPLAY_AREA: Set ResizeMode back to Adjust.")
        # ---------------------------
        self.chat_list_view.setUniformItemSizes(False) # Items have different heights
        self.chat_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel) # Smoother scrolling
        self.chat_list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # --- Model Setup ---
        self.chat_list_model = ChatListModel(self)
        self.chat_list_view.setModel(self.chat_list_model)

        # --- Delegate Setup ---
        self.chat_item_delegate = ChatItemDelegate(self)
        self.chat_list_view.setItemDelegate(self.chat_item_delegate)

        # --- Context Menu Setup ---
        self.chat_list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list_view.customContextMenuRequested.connect(self._show_chat_bubble_context_menu)
        logger.info("ChatDisplayArea context menu policy set.")
        # --- End Context Menu Setup ---


        outer_layout.addWidget(self.chat_list_view)
        self.setLayout(outer_layout)
        logger.info("ChatDisplayArea UI initialized with QListView, Model, and Delegate.")

    # --- Connect internal model signals ---
    def _connect_model_signals(self):
        if self.chat_list_model:
            # Connect modelReset signal to clear the delegate's cache
            self.chat_list_model.modelReset.connect(self._handle_model_reset)
            # Could connect rowsInserted if specific actions needed after add
            # self.chat_list_model.rowsInserted.connect(self._handle_rows_inserted)

    @pyqtSlot()
    def _handle_model_reset(self):
        """Called when the model emits modelReset."""
        logger.info(">>> DISPLAY_AREA: Handling modelReset signal.")
        if self.chat_item_delegate:
            logger.info(">>> DISPLAY_AREA: Clearing delegate cache.")
            self.chat_item_delegate.clearCache()
        self._scroll_to_bottom() # Scroll after reset (usually means clear or load history)

    # --- Slots to Interact with the Model (Called by MainWindow) ---

    @pyqtSlot(ChatMessage)
    def add_message_to_model(self, message: ChatMessage):
        """Adds a new, complete message to the model."""
        logger.debug(f"DisplayArea: Adding message to model (Role: {message.role})")
        if self.chat_list_model:
            self.chat_list_model.addMessage(message)
            self._scroll_to_bottom() # Scroll after adding
        else:
            logger.error("Cannot add message: chat_list_model is None.")

    # --- ADDED: Method to update an existing message ---
    @pyqtSlot(int, ChatMessage)
    def update_message_in_model(self, index: int, message: ChatMessage):
        """Updates the message at a specific index in the model."""
        logger.debug(f"DisplayArea: Updating message in model at index {index} (Role: {message.role})")
        if self.chat_list_model:
            self.chat_list_model.updateMessage(index, message)
            # Optional: Scroll if the updated item is the last one? Maybe not needed.
            # if index == self.chat_list_model.rowCount() - 1:
            #     self._scroll_to_bottom()
        else:
             logger.error("Cannot update message: chat_list_model is None.")
    # --- END ADDED ---

    @pyqtSlot(ChatMessage)
    def start_streaming_in_model(self, initial_message: ChatMessage):
        """Adds the initial placeholder message for a stream to the model."""
        logger.debug(f"DisplayArea: Starting stream in model (Role: {initial_message.role})")
        if self.chat_list_model:
            if initial_message.metadata is None: initial_message.metadata = {}
            initial_message.metadata["is_streaming"] = True
            self.chat_list_model.addMessage(initial_message) # Add the placeholder
            self._scroll_to_bottom()
        else:
            logger.error("Cannot start stream: chat_list_model is None.")

    @pyqtSlot(str)
    def append_stream_chunk_to_model(self, chunk: str):
        """Appends a text chunk to the last message in the model."""
        if self.chat_list_model:
            # Use the model's method which now only updates internally
            self.chat_list_model.appendChunkToLastMessage(chunk)
            v_scrollbar = self.chat_list_view.verticalScrollBar()
            # --- Optimized scrolling: only if needed ---
            if v_scrollbar and v_scrollbar.value() >= v_scrollbar.maximum() - v_scrollbar.pageStep() // 2:
                self._scroll_to_bottom()
        else:
            logger.error("Cannot append chunk: chat_list_model is None.")

    @pyqtSlot()
    def finalize_stream_in_model(self):
        """Marks the last message in the model as finalized and triggers UI update."""
        logger.debug("DisplayArea: Finalizing stream in model.")
        if self.chat_list_model:
            # Finalize in model (this will now emit dataChanged)
            self.chat_list_model.finalizeLastMessage()
            # --- RESTORED Scroll Timer ---
            QTimer.singleShot(100, self._scroll_to_bottom)
            logger.info(">>> DISPLAY_AREA: Re-enabled auto-scroll on stream finalization.")
            # --- END ---
        else:
            logger.error("Cannot finalize stream: chat_list_model is None.")

    @pyqtSlot(list)
    def load_history_into_model(self, history: List[ChatMessage]):
        """Loads a complete history list into the model."""
        logger.info(f"DisplayArea: Received request to load history (count: {len(history)}) into model.")
        if self.chat_list_model:
            self.chat_list_model.loadHistory(history) # This will trigger modelReset
        else:
            logger.error("Cannot load history: chat_list_model is None.")

    @pyqtSlot()
    def clear_model_display(self):
        """Clears all messages from the model."""
        logger.info("DisplayArea: Received request to clear model display.")
        if self.chat_list_model:
            self.chat_list_model.clearMessages() # This triggers modelReset
        else:
            logger.error("Cannot clear display: chat_list_model is None.")

    # --- Context Menu Methods ---
    @pyqtSlot(QPoint)
    def _show_chat_bubble_context_menu(self, pos: QPoint):
        """Handles the request to show a context menu for a chat bubble."""
        index = self.chat_list_view.indexAt(pos)
        if not index.isValid():
            logger.debug("Context menu requested outside of any item.")
            return

        message = self.chat_list_model.data(index, ChatMessageRole)

        if message and message.role not in [SYSTEM_ROLE, ERROR_ROLE] and message.text.strip():
            context_menu = QMenu(self)
            copy_action = context_menu.addAction("Copy Message Text")
            copy_action.triggered.connect(lambda checked=False, msg_text=message.text: self._copy_message_text(msg_text))

            context_menu.exec(self.chat_list_view.mapToGlobal(pos))
            logger.debug(f"Context menu shown for message role: {message.role}")
        elif message:
             logger.debug(f"Context menu requested for a message type ({message.role}) that cannot be copied or has no text.")

    def _copy_message_text(self, text: str):
        """Copies the given text to the system clipboard."""
        try:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)
                logger.info("Message text copied to clipboard via context menu.")
                self.textCopied.emit("Message text copied to clipboard.", "#98c379")
            else:
                logger.error("Could not access system clipboard.")
                self.textCopied.emit("Error: Could not access clipboard.", "#e06c75")
        except Exception as e:
            logger.exception(f"Error copying text to clipboard: {e}")
            self.textCopied.emit(f"Error copying text: {e}", "#e06c75")
    # --- End Context Menu Methods ---


    # --- Scrolling ---
    def _scroll_to_bottom(self):
        """Scrolls the list view to the bottom."""
        if self.chat_list_view and self.chat_list_model and self.chat_list_model.rowCount() > 0:
            QTimer.singleShot(0, lambda: self.chat_list_view.scrollToBottom())


    # --- Public Accessor for the Model (if needed by MainWindow/others) ---
    def get_model(self) -> Optional[ChatListModel]:
        return self.chat_list_model