# /// ui/chat_display_area.py
# SynaChat/ui/chat_display_area.py
# UPDATED - Added update_message_in_model slot and reverted ResizeMode

import logging
from typing import List, Dict, Any, Optional

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListView, QAbstractItemView, QSizePolicy,
    QMenu, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QModelIndex, QPoint, pyqtSignal
from PyQt6.QtGui import QAction

# --- Local Imports ---
from core.models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE
from .chat_list_model import ChatListModel, ChatMessageRole
from .chat_item_delegate import ChatItemDelegate

logger = logging.getLogger(__name__)

class ChatDisplayArea(QWidget):
    textCopied = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatDisplayAreaWidget")
        self.chat_list_view: Optional[QListView] = None
        self.chat_list_model: Optional[ChatListModel] = None
        self.chat_item_delegate: Optional[ChatItemDelegate] = None
        self._init_ui()
        self._connect_model_signals()

    def _init_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.chat_list_view = QListView(self)
        self.chat_list_view.setObjectName("ChatListView")
        self.chat_list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chat_list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.chat_list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.chat_list_view.setUniformItemSizes(False)
        self.chat_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.chat_list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_list_model = ChatListModel(self)
        self.chat_list_view.setModel(self.chat_list_model)
        self.chat_item_delegate = ChatItemDelegate(self)
        self.chat_list_view.setItemDelegate(self.chat_item_delegate)
        self.chat_list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list_view.customContextMenuRequested.connect(self._show_chat_bubble_context_menu)
        outer_layout.addWidget(self.chat_list_view)
        self.setLayout(outer_layout)
        logger.info("ChatDisplayArea UI initialized.")

    def _connect_model_signals(self):
        if self.chat_list_model:
            self.chat_list_model.modelReset.connect(self._handle_model_reset)

    @pyqtSlot()
    def _handle_model_reset(self):
        logger.info(">>> DISPLAY_AREA: Handling modelReset signal.")
        if self.chat_item_delegate:
            logger.info(">>> DISPLAY_AREA: Clearing delegate cache.")
            self.chat_item_delegate.clearCache()
        self._scroll_to_bottom()

    @pyqtSlot(ChatMessage)
    def add_message_to_model(self, message: ChatMessage):
        # ---- NEW DISTINCT LOG ----
        logger.error(f"@@@@@@@@ CDA add_message_to_model ENTRY: Role={message.role}, ID={message.id}, Text='{message.text[:100]}...' @@@@@@@@")
        # ---- END NEW DISTINCT LOG ----
        logger.debug(f"DisplayArea: Adding message to model (Role: {message.role})")
        if self.chat_list_model:
            self.chat_list_model.addMessage(message)
            self._scroll_to_bottom()
        else:
            logger.error("Cannot add message: chat_list_model is None.")

    @pyqtSlot(int, ChatMessage)
    def update_message_in_model(self, index: int, message: ChatMessage):
        logger.debug(f"DisplayArea: Updating message in model at index {index} (Role: {message.role})")
        if self.chat_list_model:
            self.chat_list_model.updateMessage(index, message)
        else:
             logger.error("Cannot update message: chat_list_model is None.")

    @pyqtSlot(ChatMessage)
    def start_streaming_in_model(self, initial_message: ChatMessage):
        logger.debug(f"DisplayArea: Starting stream in model (Role: {initial_message.role})")
        if self.chat_list_model:
            if initial_message.metadata is None: initial_message.metadata = {}
            initial_message.metadata["is_streaming"] = True
            self.chat_list_model.addMessage(initial_message)
            self._scroll_to_bottom()
        else:
            logger.error("Cannot start stream: chat_list_model is None.")

    @pyqtSlot(str)
    def append_stream_chunk_to_model(self, chunk: str):
        if self.chat_list_model:
            self.chat_list_model.appendChunkToLastMessage(chunk)
            v_scrollbar = self.chat_list_view.verticalScrollBar()
            if v_scrollbar and v_scrollbar.value() >= v_scrollbar.maximum() - v_scrollbar.pageStep() // 2:
                self._scroll_to_bottom()
        else:
            logger.error("Cannot append chunk: chat_list_model is None.")

    @pyqtSlot()
    def finalize_stream_in_model(self):
        logger.debug("DisplayArea: Finalizing stream in model.")
        if self.chat_list_model:
            self.chat_list_model.finalizeLastMessage()
            QTimer.singleShot(100, self._scroll_to_bottom)
        else:
            logger.error("Cannot finalize stream: chat_list_model is None.")

    @pyqtSlot(list)
    def load_history_into_model(self, history: List[ChatMessage]):
        logger.info(f"DisplayArea: Received request to load history (count: {len(history)}) into model.")
        if self.chat_list_model:
            self.chat_list_model.loadHistory(history)
        else:
            logger.error("Cannot load history: chat_list_model is None.")

    @pyqtSlot()
    def clear_model_display(self):
        logger.info("DisplayArea: Received request to clear model display.")
        if self.chat_list_model:
            self.chat_list_model.clearMessages()
        else:
            logger.error("Cannot clear display: chat_list_model is None.")

    @pyqtSlot(QPoint)
    def _show_chat_bubble_context_menu(self, pos: QPoint):
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

    def _scroll_to_bottom(self):
        if self.chat_list_view and self.chat_list_model and self.chat_list_model.rowCount() > 0:
            QTimer.singleShot(0, lambda: self.chat_list_view.scrollToBottom())

    def get_model(self) -> Optional[ChatListModel]:
        return self.chat_list_model