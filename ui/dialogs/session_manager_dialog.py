# SynAva_1.0/ui/dialogs/session_manager_dialog.py
# Contains the SessionManagerDialog class, extracted from the original dialogs.py

import logging
import os
from datetime import datetime
from typing import Optional

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox, QAbstractItemView, QFileDialog, QLabel, QWidget
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont

# --- Local Imports ---
# Adjust paths based on your final structure
try:
    # --- MODIFIED IMPORTS ---
    from core.chat_manager import ChatManager # Was from ...core.chat_manager
    from utils import constants # Was from ...utils
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE, CONVERSATIONS_DIR # Was from ...utils.constants
    # --- END MODIFIED IMPORTS ---
except ImportError as e:
    # Fallback for potential structure issues during refactoring
    logging.error(f"Error importing dependencies in session_manager_dialog.py: {e}. Check relative paths.")
    # Define dummy values if needed for the script to be syntactically valid
    class ChatManager: pass # type: ignore
    class constants: CHAT_FONT_FAMILY="Arial"; CHAT_FONT_SIZE=10; CONVERSATIONS_DIR="." # type: ignore

logger = logging.getLogger(__name__)

# --- Session Manager Dialog ---
class SessionManagerDialog(QDialog):
    """
    A modal dialog for managing saved chat sessions (loading, saving as, deleting).
    Requires a ChatManager instance to interact with session data.
    """
    def __init__(self, chat_manager: ChatManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if not chat_manager:
            # Log and raise error if ChatManager is not provided
            logger.critical("SessionManagerDialog requires a valid ChatManager instance.")
            raise ValueError("SessionManagerDialog requires ChatManager.")
        self.chat_manager = chat_manager

        self.setWindowTitle("Manage Sessions")
        self.setObjectName("SessionManagerDialog")
        self.setMinimumSize(500, 400)
        self.setModal(True) # Modal dialog

        # --- UI Setup ---
        self._init_widgets()
        self._init_layout()
        self._connect_signals()

        # Initial population and state update
        self.refresh_list()
        self._update_button_states()

    def _init_widgets(self):
        """Initialize the widgets for the dialog."""
        dialog_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
        label_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)

        # Label for the list
        self.list_label = QLabel("Saved Sessions:")
        self.list_label.setFont(label_font)

        # List widget to display sessions
        self.session_list_widget = QListWidget()
        self.session_list_widget.setFont(dialog_font)
        self.session_list_widget.setObjectName("SessionList")
        self.session_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Buttons
        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Load selected session (Double-click)")
        self.save_as_button = QPushButton("Save Current As...")
        self.save_as_button.setToolTip("Save the current chat session with a new name")
        self.save_as_button.setEnabled(True) # Always enabled
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete selected session")
        self.close_button = QPushButton("Close")

    def _init_layout(self):
        """Set up the layout for the dialog."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        layout.addWidget(self.list_label)
        layout.addWidget(self.session_list_widget, 1) # List widget stretches

        # Button Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_as_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.session_list_widget.itemSelectionChanged.connect(self._update_button_states)
        self.session_list_widget.itemDoubleClicked.connect(self._handle_load) # Load on double-click
        self.load_button.clicked.connect(self._handle_load)
        self.save_as_button.clicked.connect(self._handle_save_as)
        self.delete_button.clicked.connect(self._handle_delete)
        self.close_button.clicked.connect(self.reject) # Close rejects the dialog

    # --- Helper Methods ---
    def _get_selected_filepath(self) -> Optional[str]:
        """Gets the full filepath stored in the selected list item's data."""
        selected_items = self.session_list_widget.selectedItems()
        if selected_items:
            # Retrieve the path stored in the UserRole data
            filepath = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if isinstance(filepath, str):
                return filepath
        return None

    def refresh_list(self):
        """Refreshes the list of saved sessions from the ChatManager."""
        current_selection_path = self._get_selected_filepath() # Remember selection
        self.session_list_widget.clear()
        logger.info("Refreshing session list in SessionManagerDialog...")
        try:
            # Get the list of full file paths from ChatManager
            session_filepaths = self.chat_manager.list_saved_sessions()

            if not session_filepaths:
                # Display message if no sessions found
                item = QListWidgetItem("No saved sessions found.")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable) # Make it unselectable
                self.session_list_widget.addItem(item)
                self.session_list_widget.setEnabled(False)
            else:
                self.session_list_widget.setEnabled(True)
                new_selection_index = -1
                # Populate the list widget
                for i, fp in enumerate(session_filepaths):
                    filename = os.path.basename(fp)
                    item = QListWidgetItem(filename)
                    item.setData(Qt.ItemDataRole.UserRole, fp) # Store full path in data role
                    item.setToolTip(fp) # Show full path in tooltip
                    self.session_list_widget.addItem(item)
                    # Restore selection if possible
                    if fp == current_selection_path:
                        new_selection_index = i

                # Set the selection back if it was found
                if new_selection_index != -1:
                    self.session_list_widget.setCurrentRow(new_selection_index)

        except Exception as e:
            logger.exception("Error refreshing session list:")
            self.session_list_widget.clear()
            item = QListWidgetItem("Error loading sessions.")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.session_list_widget.addItem(item)
            self.session_list_widget.setEnabled(False)
        finally:
            self._update_button_states() # Update button states after refresh

    def _update_button_states(self):
        """Enables/disables buttons based on list selection."""
        is_item_selected = self._get_selected_filepath() is not None
        self.load_button.setEnabled(is_item_selected)
        self.delete_button.setEnabled(is_item_selected)
        self.save_as_button.setEnabled(True) # Save As is always enabled

    # --- Action Handlers (Slots) ---
    @pyqtSlot()
    def _handle_load(self):
        """Handles the load button click or double-click."""
        filepath = self._get_selected_filepath()
        if filepath:
            logger.info(f"SessionManagerDialog requesting load of session: '{filepath}'")
            self.chat_manager.load_chat_session(filepath) # Delegate to ChatManager
            self.accept() # Close dialog after successful load action
        else:
            logger.warning("Load action triggered with no session selected.")

    @pyqtSlot()
    def _handle_save_as(self):
        """Handles the Save As button click."""
        logger.info("SessionManagerDialog requesting Save As...")
        # Suggest a filename based on current time and project ID
        project_id = self.chat_manager.get_current_project_id()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suggested_base = f"session_{timestamp}{f'_{project_id[:8]}' if project_id else ''}.json"

        try:
            # Use SessionService (via ChatManager if needed, or directly if passed) for sanitization
            # Assuming ChatManager has access to SessionService or its method
            sanitized_filename = self.chat_manager._session_service.sanitize_filename(suggested_base)
            suggested_path = os.path.join(constants.CONVERSATIONS_DIR, sanitized_filename)
            # Ensure the directory exists
            os.makedirs(constants.CONVERSATIONS_DIR, exist_ok=True)

            # Open file dialog
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Current Session As",
                suggested_path,
                "JSON Session Files (*.json);;All Files (*)"
            )

            if filepath:
                # Ensure .json extension
                if not filepath.lower().endswith(".json"):
                    filepath += ".json"
                # Delegate saving to ChatManager
                if self.chat_manager.save_current_chat_session(filepath):
                    self.refresh_list() # Refresh list to show the newly saved session
                    QMessageBox.information(self, "Save Successful", f"Session saved as:\n{os.path.basename(filepath)}")
                # else: Error should be handled by ChatManager signaling MainWindow
            else:
                logger.info("Save As action cancelled by user.")
        except Exception as e:
            logger.exception("Error during Save As operation:")
            QMessageBox.critical(self, "Save Error", f"An error occurred while trying to save:\n{e}")

    @pyqtSlot()
    def _handle_delete(self):
        """Handles the delete button click."""
        filepath = self._get_selected_filepath()
        if filepath:
            filename = os.path.basename(filepath)
            # Confirm deletion with the user
            confirm = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete the session '{filename}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No # Default to No
            )

            if confirm == QMessageBox.StandardButton.Yes:
                logger.info(f"SessionManagerDialog requesting delete of session: '{filepath}'")
                # Delegate deletion to ChatManager
                if self.chat_manager.delete_chat_session(filepath):
                    self.refresh_list() # Refresh list after successful deletion
                # else: Error should be handled by ChatManager signaling MainWindow
        else:
            logger.warning("Delete action triggered with no session selected.")

    # --- Dialog Lifecycle Overrides ---
    def showEvent(self, event):
        """Refresh the list when the dialog is shown."""
        super().showEvent(event)
        self.refresh_list() # Ensure list is up-to-date when shown