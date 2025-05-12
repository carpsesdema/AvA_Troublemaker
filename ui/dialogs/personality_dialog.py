# SynAva_1.0/ui/dialogs/personality_dialog.py
# Contains the EditPersonalityDialog class, extracted from the original dialogs.py

import logging
from typing import Optional

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QDialogButtonBox, QLabel, QWidget
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# --- Local Imports ---
# Adjust paths based on your final structure
try:
    # --- MODIFIED IMPORT ---
    from utils import constants # Was from ...utils
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE # Was from ...utils.constants
    # --- END MODIFIED IMPORT ---
except ImportError as e:
    # Fallback for potential structure issues during refactoring
    logging.error(f"Error importing dependencies in personality_dialog.py: {e}. Check relative paths.")
    # Define dummy values if needed for the script to be syntactically valid
    class constants: CHAT_FONT_FAMILY="Arial"; CHAT_FONT_SIZE=10 # type: ignore

logger = logging.getLogger(__name__)

# --- Personality Editor Dialog ---
class EditPersonalityDialog(QDialog):
    """
    A modal dialog for editing the AI's system prompt or personality.
    """
    def __init__(self, current_prompt: Optional[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Personality / System Prompt")
        self.setObjectName("PersonalityDialog")
        self.setMinimumSize(550, 400)
        self.setModal(True) # Modal dialog

        # --- UI Setup ---
        self._init_widgets(current_prompt)
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self, current_prompt: Optional[str]):
        """Initialize the widgets for the dialog."""
        dialog_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
        label_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)

        # Informational label
        self.info_label = QLabel(
            "Enter the system prompt or personality instructions for the AI below.\n"
            "Leave empty to use the default behavior."
        )
        self.info_label.setFont(label_font)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #9aabbf; margin-bottom: 5px;") # Example style

        # Text editor for the prompt
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setObjectName("PersonalityPromptEdit")
        self.prompt_edit.setFont(dialog_font)
        self.prompt_edit.setPlaceholderText("e.g., You are a helpful assistant specializing in Python...")
        self.prompt_edit.setPlainText(current_prompt or "") # Populate with current prompt
        self.prompt_edit.setAcceptRichText(False) # Plain text only
        self.prompt_edit.setMinimumHeight(150) # Ensure decent editing space

        # Standard OK/Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

    def _init_layout(self):
        """Set up the layout for the dialog."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        layout.addWidget(self.info_label)
        layout.addWidget(self.prompt_edit, 1) # Allow text edit to stretch
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.button_box.accepted.connect(self.accept) # Connect OK to accept
        self.button_box.rejected.connect(self.reject) # Connect Cancel to reject

    def get_prompt_text(self) -> str:
        """Returns the entered prompt text, stripped of whitespace."""
        return self.prompt_edit.toPlainText().strip()

    # --- Dialog Lifecycle Overrides ---
    def showEvent(self, event):
        """Set focus to the text editor when the dialog is shown."""
        super().showEvent(event)
        self.prompt_edit.setFocus() # Set initial focus