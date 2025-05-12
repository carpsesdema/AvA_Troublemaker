# ui/chat_tab_widget.py
import logging
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSlot

# Assuming ChatDisplayArea and ChatInputBar are in the same 'ui' directory
try:
    from .chat_display_area import ChatDisplayArea
    from .chat_input_bar import ChatInputBar
except ImportError as e:
    logging.critical(f"ChatTabWidget: Failed to import ChatDisplayArea or ChatInputBar: {e}")
    # Define dummy classes if import fails, so the rest of the app can try to load
    ChatDisplayArea = type("ChatDisplayArea", (QWidget,), {})
    ChatInputBar = type("ChatInputBar", (QWidget,), {})

logger = logging.getLogger(__name__)

class ChatTabWidget(QWidget):
    """
    A QWidget that serves as the content for each tab in the main chat interface.
    It contains a ChatDisplayArea for showing messages and a ChatInputBar for user input.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatTabContentWidget") # For styling if needed

        self.chat_display_area: ChatDisplayArea = ChatDisplayArea(self)
        self.chat_input_bar: ChatInputBar = ChatInputBar(self)

        self._init_layout()
        logger.debug(f"ChatTabWidget '{self.objectName()}' initialized.")

    def _init_layout(self):
        """Sets up the layout for the chat display and input bar."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # No margins within the tab content
        layout.setSpacing(0) # No spacing between display and input bar

        layout.addWidget(self.chat_display_area, 1) # Display area takes available stretch
        layout.addWidget(self.chat_input_bar)

        self.setLayout(layout)

    # --- Public methods to access internal widgets if needed by MainWindow ---
    def get_chat_display_area(self) -> ChatDisplayArea:
        return self.chat_display_area

    def get_chat_input_bar(self) -> ChatInputBar:
        return self.chat_input_bar

    # --- Slots (potentially for MainWindow to interact with, or for internal use) ---
    @pyqtSlot()
    def clear_input(self):
        """Clears the text in this tab's input bar."""
        self.chat_input_bar.clear_text()

    @pyqtSlot()
    def set_focus_to_input(self):
        """Sets focus to this tab's input bar."""
        self.chat_input_bar.set_focus()