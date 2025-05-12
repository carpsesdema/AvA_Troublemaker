# SynAva_1.0/ui/dialogs/code_viewer_dialog.py
# Contains the CodeViewerWindow class, extracted from the original dialogs.py

import logging
import os
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QSplitter,
    QApplication, QMessageBox, QTreeWidget, QTreeWidgetItem, QWidget
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSlot, QPoint
from PyQt6.QtGui import QFont, QClipboard, QIcon, QFontDatabase, QTextOption

# --- Local Imports ---
# Adjust paths based on your final structure
try:
    # --- MODIFIED IMPORTS ---
    from utils import constants # Was from ...utils
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE # Was from ...utils.constants
    from utils.syntax_highlighter import PythonSyntaxHighlighter # Was from ...utils.syntax_highlighter
    from ui.widgets import COPY_ICON, CHECK_ICON # Was from ..widgets
    # --- END MODIFIED IMPORTS ---
except ImportError as e:
    # Fallback for potential structure issues during refactoring
    logging.error(f"Error importing dependencies in code_viewer_dialog.py: {e}. Check relative paths.")
    # Define dummy values if needed for the script to be syntactically valid
    class constants: CHAT_FONT_FAMILY="Arial"; CHAT_FONT_SIZE=10
    class PythonSyntaxHighlighter: pass # type: ignore
    COPY_ICON, CHECK_ICON = QIcon(), QIcon()

logger = logging.getLogger(__name__)

# --- Code Viewer Window ---
class CodeViewerWindow(QDialog):
    """
    A non-modal dialog to display code blocks or full file contents,
    organized by filename using a QTreeWidget.
    """
    # Constant for data role (not strictly needed now as content is in dict)
    CODE_CONTENT_ROLE = Qt.ItemDataRole.UserRole + 10

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Code Viewer")
        self.setObjectName("CodeViewerWindow")
        self.setMinimumSize(700, 500)
        self.setModal(False) # Non-modal window

        # Store file content associated with tree items {filename: content}
        self._file_contents: Dict[str, str] = {}
        self._current_filename: Optional[str] = None # Track currently displayed file

        # --- UI Setup ---
        self._init_widgets()
        self._init_layout()
        self._connect_signals()

        # Attach syntax highlighter
        self._attach_highlighter()

    def _init_widgets(self):
        """Initialize the widgets for the dialog."""
        # Font setup
        code_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        code_font.setPointSize(constants.CHAT_FONT_SIZE)
        logger.info(f"CodeViewerWindow using Monospace Font: {code_font.family()} {code_font.pointSize()}pt")
        self.code_font = code_font # Store for later use if needed

        # Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # File Tree (using QTreeWidget)
        self.file_tree = QTreeWidget()
        self.file_tree.setObjectName("CodeFileTree")
        self.file_tree.setHeaderLabels(["File / Block"]) # Single column header
        self.file_tree.setMinimumWidth(200)
        self.file_tree.header().setStretchLastSection(True) # Header stretches

        # Code Editor
        self.code_edit = QTextEdit()
        self.code_edit.setObjectName("CodeViewerEdit")
        self.code_edit.setReadOnly(True)
        self.code_edit.setFont(self.code_font)
        self.code_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Disable line wrapping
        self.code_edit.setWordWrapMode(QTextOption.WrapMode.NoWrap) # Disable word wrapping

        # Buttons
        self.copy_button = QPushButton(" Copy Code")
        self.copy_button.setToolTip("Copy the code currently shown")
        if not COPY_ICON.isNull(): self.copy_button.setIcon(COPY_ICON)
        self.copy_button.setEnabled(False) # Disabled initially

        self.clear_button = QPushButton("Clear All")
        self.clear_button.setToolTip("Remove all listed files/blocks")
        self.clear_button.setEnabled(False) # Disabled initially

        self.close_button = QPushButton("Close")
        self.close_button.setToolTip("Hide this window")

    def _init_layout(self):
        """Set up the layout for the dialog."""
        layout = QVBoxLayout(self)

        # Add widgets to splitter
        self.splitter.addWidget(self.file_tree)
        self.splitter.addWidget(self.code_edit)
        self.splitter.setSizes([220, 480]) # Initial splitter sizes
        layout.addWidget(self.splitter, 1) # Splitter takes most space

        # Button Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.copy_button)
        button_layout.addStretch()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.file_tree.currentItemChanged.connect(self._display_selected_file_content)
        self.copy_button.clicked.connect(self._copy_selected_code_with_feedback)
        self.clear_button.clicked.connect(self.clear_viewer)
        self.close_button.clicked.connect(self.hide) # Hide instead of accept/reject

    def _attach_highlighter(self):
        """Attaches the syntax highlighter to the code editor."""
        self.highlighter = None
        try:
            # Pass the QTextDocument of the code_edit widget
            self.highlighter = PythonSyntaxHighlighter(self.code_edit.document())
            logger.info("CodeViewerWindow: PythonSyntaxHighlighter attached.")
        except Exception as e_hl:
             logger.error(f"Error attaching PythonSyntaxHighlighter: {e_hl}")

    # --- Public Methods ---
    def update_or_add_file(self, filename: str, content: str):
        """
        Adds a new file entry or updates the content of an existing one.
        Selects the added/updated item.
        """
        if not filename:
            logger.warning("Attempted to add/update file with empty filename.")
            return

        self._file_contents[filename] = content # Store/update content in the dictionary
        self.clear_button.setEnabled(True) # Enable clear button once there's content

        # Find existing item in the tree
        found_item = None
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            if item and item.text(0) == filename:
                found_item = item
                break

        if found_item:
            # Item exists, just select it (content is already updated in dict)
            logger.debug(f"Updating existing file in Code Viewer: {filename}")
            self.file_tree.setCurrentItem(found_item)
            # Trigger display update manually if selection didn't change programmatically
            # This ensures content refreshes if the same item is "updated" again
            if self._current_filename == filename:
                 self._display_selected_file_content(found_item, None)
        else:
            # Item doesn't exist, create a new top-level item
            logger.debug(f"Adding new file to Code Viewer: {filename}")
            new_item = QTreeWidgetItem(self.file_tree)
            new_item.setText(0, filename)
            new_item.setToolTip(0, filename) # Tooltip is the filename
            # No need to store content in item data, we use the dictionary
            self.file_tree.setCurrentItem(new_item) # Select the new item, triggers display

        # Ensure viewer is visible when updated
        if not self.isVisible():
            self.show()
        self.activateWindow()
        self.raise_()

    def add_code_block(self, language: str, code_content: str):
        """Adds a sequentially named code block (for compatibility)."""
        # Generates a unique name based on timestamp
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        lang_display = language.strip().capitalize() if language.strip() else "Code"
        block_name = f"{lang_display} Block ({timestamp})"
        self.update_or_add_file(block_name, code_content)

    def clear_viewer(self):
        """Clears all files and code from the viewer."""
        if not self._file_contents: return # Nothing to clear
        response = QMessageBox.question(self, "Confirm Clear",
                                        "Remove all items from the Code Viewer?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
        if response == QMessageBox.StandardButton.Yes:
            self._file_contents.clear()
            self.file_tree.clear()
            self.code_edit.clear()
            self.copy_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self._current_filename = None
            logger.info("Cleared all files/blocks from CodeViewerWindow.")

    # --- Private Slots and Helpers ---
    @pyqtSlot(QTreeWidgetItem, QTreeWidgetItem)
    def _display_selected_file_content(self, current_item: Optional[QTreeWidgetItem], previous_item: Optional[QTreeWidgetItem]):
        """Displays the code content associated with the selected tree item."""
        self._reset_copy_button_icon() # Reset icon from previous copy if any
        if current_item is None:
            # No item selected, clear the editor
            self.code_edit.clear()
            self.copy_button.setEnabled(False)
            self._current_filename = None
            return

        filename = current_item.text(0) # Filename is the text of the item
        self._current_filename = filename # Track current file

        # Retrieve content from the dictionary
        code_content = self._file_contents.get(filename)

        if code_content is not None:
            self.code_edit.setPlainText(code_content)
            # Re-apply highlighting if highlighter exists
            if self.highlighter:
                try:
                    # Rehighlighting might be needed if content changes or for initial display
                    self.highlighter.rehighlight()
                except Exception as e_rh:
                    logger.error(f"Error rehighlighting code for {filename}: {e_rh}")
            self.copy_button.setEnabled(True) # Enable copy button
        else:
            # Should not happen if item exists, but handle defensively
            logger.warning(f"Content not found in dictionary for selected file: {filename}")
            self.code_edit.setPlainText(f"[Error: Content not found for {filename}]")
            self.copy_button.setEnabled(False)

    def _copy_selected_code_with_feedback(self):
        """Copies the currently displayed code content to the clipboard."""
        code_to_copy = self.code_edit.toPlainText()
        if not code_to_copy or self._current_filename is None:
            logger.warning("Attempted to copy empty or unselected code from CodeViewerWindow.")
            return
        # Avoid copying error messages displayed in the editor
        if code_to_copy.startswith("[Error:"):
            return

        try:
            clipboard = QApplication.clipboard()
            if not clipboard:
                raise RuntimeError("Clipboard not accessible.")
            clipboard.setText(code_to_copy)
            logger.info(f"Copied code for '{self._current_filename}' from viewer.")
            # Provide visual feedback on the button
            if not CHECK_ICON.isNull():
                self.copy_button.setIcon(CHECK_ICON)
            self.copy_button.setEnabled(False) # Temporarily disable after copy
            # Reset icon and state after a delay
            QTimer.singleShot(1500, self._reset_copy_button_icon)
        except Exception as e:
            logger.exception(f"Error copying code from viewer: {e}")
            QMessageBox.warning(self, "Copy Error", f"Could not copy code:\n{e}")

    def _reset_copy_button_icon(self):
        """Resets the copy button icon and enabled state."""
        if not COPY_ICON.isNull():
            self.copy_button.setIcon(COPY_ICON)
        # Re-enable only if valid content is currently displayed
        self.copy_button.setEnabled(bool(self._current_filename) and self._current_filename in self._file_contents)

    # --- Dialog Lifecycle Overrides ---
    def showEvent(self, event):
        """Ensure item selection is handled correctly when the dialog is shown."""
        super().showEvent(event)
        # If there are items but none selected, select the first one
        if self.file_tree.topLevelItemCount() > 0 and self.file_tree.currentItem() is None:
            self.file_tree.setCurrentItem(self.file_tree.topLevelItem(0))
        # If an item is already selected, ensure its content is displayed
        elif self.file_tree.currentItem():
             self._display_selected_file_content(self.file_tree.currentItem(), None)
        # Ensure the window is raised and activated
        self.activateWindow()
        self.raise_()

    def closeEvent(self, event):
        """Override close event to hide the dialog instead of destroying it."""
        self.hide()
        event.ignore() # Prevent the dialog from being destroyed