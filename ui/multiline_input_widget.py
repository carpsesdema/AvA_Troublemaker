# Llama_Syn/ui/multiline_input_widget.py
# UPDATED FILE - Implemented dynamic height resizing

import logging
from typing import Optional

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QFont, QFontMetrics, QTextOption, QKeyEvent
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer

# --- Local Imports ---
from utils import constants

logger = logging.getLogger(__name__)

class MultilineInputWidget(QWidget):
    """
    A custom QWidget containing a QTextEdit configured for multi-line chat input.
    Handles Enter/Shift+Enter key presses and emits appropriate signals.
    Dynamically resizes vertically based on content within min/max limits.
    """
    sendMessageRequested = pyqtSignal() # Signal emitted when Enter is pressed
    textChanged = pyqtSignal()        # Re-emitted signal from internal QTextEdit

    # Configuration for resizing
    MIN_LINES = 1
    MAX_LINES = 8 # Adjust maximum lines as desired
    LINE_PADDING = 8 # Extra vertical padding calculation

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("MultilineInputWidget")

        self._text_edit: Optional[QTextEdit] = None
        self._min_height: int = 30 # Fallback min height
        self._max_height: int = 200 # Fallback max height

        self._init_ui()
        self._calculate_height_limits() # Calculate min/max based on font
        self._connect_signals()
        self._update_height() # Set initial height

    def _init_ui(self):
        """Set up the internal layout and QTextEdit widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._text_edit = QTextEdit(self)
        self._text_edit.setObjectName("UserInputTextEdit") # For QSS styling
        self._text_edit.setAcceptRichText(False)
        self._text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        # Placeholder text is not natively supported well with dynamic height, skipping.

        # --- Font Setup (remains same) ---
        try:
            font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
            self._text_edit.setFont(font)
        except Exception as e:
            logger.error(f"Error setting font for MultilineInputWidget: {e}")

        layout.addWidget(self._text_edit)
        self.setLayout(layout)

        # Remove fixed height setting from here
        # self._text_edit.setFixedHeight(target_height)

    def _calculate_height_limits(self):
        """Calculates min/max height based on font metrics and line counts."""
        if not self._text_edit:
            return
        try:
            fm = QFontMetrics(self._text_edit.font())
            line_height = fm.height()
            # Calculate base height for min/max lines
            min_base = line_height * self.MIN_LINES
            max_base = line_height * self.MAX_LINES

            # Add padding/margins (consider document margins and extra padding)
            doc_margin = int(self._text_edit.document().documentMargin()) # Usually small
            vertical_padding = self.LINE_PADDING + (doc_margin * 2)

            self._min_height = min_base + vertical_padding
            self._max_height = max_base + vertical_padding
            logger.debug(f"Calculated height limits: Min={self._min_height}, Max={self._max_height} (lineH={line_height})")

        except Exception as e:
            logger.error(f"Error calculating height limits: {e}. Using defaults.")
            # Keep fallback values assigned in __init__

    def _connect_signals(self):
        """Connect internal QTextEdit signals."""
        if self._text_edit:
            # Connect internal textChanged to the re-emitting signal AND height update
            self._text_edit.textChanged.connect(self.textChanged.emit)
            self._text_edit.textChanged.connect(self._update_height) # Connect to our slot

    @pyqtSlot()
    def _update_height(self):
        """Calculates and sets the widget's height based on content."""
        if not self._text_edit:
            return

        # Ensure document width is set correctly for accurate height calculation
        # Use viewport width if available, otherwise widget width
        vp_width = self._text_edit.viewport().width()
        effective_width = vp_width if vp_width > 0 else self.width()
        if effective_width > 0:
            self._text_edit.document().setTextWidth(effective_width)

        doc_height = self._text_edit.document().size().height()

        # Add estimated margins/padding similar to calculation
        doc_margin = int(self._text_edit.document().documentMargin())
        vertical_padding = self.LINE_PADDING + (doc_margin * 2)
        target_height = int(doc_height + vertical_padding)

        # Clamp height between min and max
        clamped_height = max(self._min_height, min(target_height, self._max_height))

        # Only resize if the clamped height differs from current fixed height
        if self.height() != clamped_height:
            logger.debug(f"Resizing MultilineInput: docH={doc_height:.1f} -> targetH={target_height} -> clampedH={clamped_height}")
            self.setFixedHeight(clamped_height)
            # Required to inform the layout system about the size change
            self.updateGeometry()


    def keyPressEvent(self, event: QKeyEvent):
        """Handle Enter and Shift+Enter key presses."""
        key = event.key()
        modifiers = event.modifiers()

        is_enter = key in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        is_shift_pressed = modifiers & Qt.KeyboardModifier.ShiftModifier

        if is_enter and not is_shift_pressed:
            logger.debug("Enter pressed in MultilineInputWidget, emitting sendMessageRequested.")
            self.sendMessageRequested.emit()
            event.accept() # Prevent newline insertion
        elif is_enter and is_shift_pressed:
            # Allow default behavior (insert newline) - QTextEdit handles this
             super().keyPressEvent(event)
             # Manually trigger height update after newline insertion if needed,
             # though textChanged should handle it.
             # QTimer.singleShot(0, self._update_height)
        else:
            # Other key pressed: Allow default behavior
            super().keyPressEvent(event)

    # --- Public Methods (remain the same) ---
    def get_text(self) -> str:
        """Returns the plain text content, stripped of leading/trailing whitespace."""
        return self._text_edit.toPlainText().strip() if self._text_edit else ""

    def clear_text(self):
        """Clears the text editor."""
        if self._text_edit:
            self._text_edit.clear()
            # After clearing, reset to minimum height immediately
            self.setFixedHeight(self._min_height)
            self.updateGeometry()


    def set_focus(self):
        """Sets keyboard focus to the text editor."""
        if self._text_edit:
            self._text_edit.setFocus()

    def set_enabled(self, enabled: bool):
        """Enables or disables the text editor."""
        if self._text_edit:
            self._text_edit.setEnabled(enabled)

    def setPlainText(self, text: str):
        """Sets the plain text content."""
        if self._text_edit:
            self._text_edit.setPlainText(text)
            # Update height after programmatically setting text
            QTimer.singleShot(0, self._update_height)