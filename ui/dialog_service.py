# ui/dialog_service.py
import logging
import os
import datetime  # Ensure datetime is imported
from typing import Optional, List, Dict, Any, Tuple

from PyQt6.QtWidgets import QWidget, QFileDialog, QInputDialog, QMenu, QMessageBox, QDialog, QLineEdit
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QCursor

# Import your custom dialogs
try:
    from .dialogs.code_viewer_dialog import CodeViewerWindow
    from .dialogs.personality_dialog import EditPersonalityDialog
    from .dialogs.session_manager_dialog import SessionManagerDialog
    from .dialogs.rag_viewer_dialog import RAGViewerDialog
except ImportError as e:
    logging.critical(f"DialogService: Failed to import custom dialogs: {e}")
    CodeViewerWindow = QDialog
    EditPersonalityDialog = QDialog
    SessionManagerDialog = QDialog
    RAGViewerDialog = QDialog

# Import ChatManager and other necessary types
try:
    from core.chat_manager import ChatManager
    # from core.models import ChatMessage # Not directly used here, but dialogs might
    from services.vector_db_service import VectorDBService  # For RAGViewerDialog
except ImportError as e:
    logging.critical(f"DialogService: Failed to import core/services: {e}")
    ChatManager = type("ChatManager", (object,), {})
    VectorDBService = type("VectorDBService", (object,), {})

from utils import constants

logger = logging.getLogger(__name__)


class DialogService(QObject):
    """
    Manages the creation, display, and basic interaction logic for various
    dialogs used in the application.
    """

    # If DialogService needs to emit signals (e.g., after a dialog result that MainWindow processes)
    # Example: focus_paths_selected_from_rag = pyqtSignal(list)

    def __init__(self, parent_window: QWidget, chat_manager: ChatManager):
        super().__init__(parent_window)
        self.parent_window = parent_window
        if not isinstance(chat_manager, ChatManager):  # Ensure chat_manager is valid
            logger.critical("DialogService initialized with invalid ChatManager instance.")
            raise ValueError("DialogService requires a valid ChatManager instance.")
        self.chat_manager = chat_manager

        # Cache dialog instances where appropriate
        self._code_viewer_window: Optional[CodeViewerWindow] = None
        self._session_manager_dialog: Optional[SessionManagerDialog] = None
        self._rag_viewer_dialog: Optional[RAGViewerDialog] = None  # RAG viewer can be modal, but caching might be ok
        logger.info("DialogService initialized.")

    def _get_upload_filter(self) -> str:
        """Generates the file filter string for upload dialogs."""
        if hasattr(self.parent_window, '_get_upload_filter') and callable(self.parent_window._get_upload_filter):
            return self.parent_window._get_upload_filter()
        logger.warning("DialogService: parent_window missing _get_upload_filter. Using basic filter.")
        return "All Files (*)"

    # --- Session Management Dialogs ---
    def show_session_manager(self) -> None:
        logger.debug("DialogService: Showing session manager.")
        if self.chat_manager.is_overall_busy():  # Use public method
            if hasattr(self.parent_window, 'update_status'):
                self.parent_window.update_status("Cannot manage sessions now, application is busy.", "#e5c07b", True,
                                                 3000)
            return
        try:
            if self._session_manager_dialog is None:
                self._session_manager_dialog = SessionManagerDialog(self.chat_manager, self.parent_window)
            self._session_manager_dialog.refresh_list()
            self._session_manager_dialog.exec()  # Modal execution
        except Exception as e:
            logger.exception("Error showing SessionManagerDialog:")
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Session Manager:\n{e}")

    def get_save_as_path(self) -> Optional[str]:
        logger.debug("DialogService: Getting 'Save As' path.")
        try:
            project_id = self.chat_manager.get_current_project_id()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            sugg_base = f"session_{timestamp}{f'_{project_id[:8]}' if project_id and project_id != constants.GLOBAL_COLLECTION_ID else ''}.json"

            session_service = getattr(self.chat_manager, '_session_service', None)  # Try to get from ChatManager
            if session_service and hasattr(session_service, 'sanitize_filename'):
                sanitized_filename = session_service.sanitize_filename(sugg_base)
            else:
                logger.warning(
                    "DialogService: Could not access SessionService or sanitize_filename. Using raw suggested base.")
                sanitized_filename = sugg_base

            suggested_path = os.path.join(constants.CONVERSATIONS_DIR, sanitized_filename)
            os.makedirs(constants.CONVERSATIONS_DIR, exist_ok=True)

            filepath, _ = QFileDialog.getSaveFileName(
                self.parent_window, "Save Current Session As", suggested_path,
                "JSON Session Files (*.json);;All Files (*)"
            )
            if filepath:
                if not filepath.lower().endswith(".json"):
                    filepath += ".json"
                return filepath
            return None
        except Exception as e:
            logger.exception("Error in get_save_as_path:")
            QMessageBox.critical(self.parent_window, "File Dialog Error", f"Could not get save path:\n{e}")
            return None

    # --- Upload Dialogs ---
    def get_upload_files_paths(self, title: str) -> List[str]:
        logger.debug(f"DialogService: Getting upload file paths for '{title}'.")
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.parent_window, title,
                constants.USER_DATA_DIR, self._get_upload_filter()
            )
            return file_paths if file_paths else []
        except Exception as e:
            logger.exception(f"Error in get_upload_files_paths for title '{title}':")
            QMessageBox.critical(self.parent_window, "File Dialog Error", f"Could not open file dialog:\n{e}")
            return []

    def get_upload_directory_path(self, title: str) -> Optional[str]:
        logger.debug(f"DialogService: Getting upload directory path for '{title}'.")
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self.parent_window, title,
                constants.USER_DATA_DIR,
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
            )
            return dir_path if dir_path else None
        except Exception as e:
            logger.exception(f"Error in get_upload_directory_path for title '{title}':")
            QMessageBox.critical(self.parent_window, "Directory Dialog Error", f"Could not open directory dialog:\n{e}")
            return None

    def show_global_upload_menu(self) -> Optional[str]:  # Returns "file" or "directory" or None
        logger.debug("DialogService: Showing global upload menu.")
        try:
            menu = QMenu(self.parent_window)
            upload_file_action = menu.addAction("Upload File(s) to Global")
            upload_dir_action = menu.addAction("Upload Directory to Global")

            target_pos = QCursor.pos()
            if hasattr(self.parent_window, 'left_panel') and self.parent_window.left_panel and \
                    hasattr(self.parent_window.left_panel, 'upload_global_button'):
                global_upload_button = self.parent_window.left_panel.upload_global_button
                if global_upload_button:
                    target_pos = global_upload_button.mapToGlobal(global_upload_button.rect().bottomLeft())

            chosen_action = menu.exec(target_pos)
            if chosen_action == upload_file_action: return "file"
            if chosen_action == upload_dir_action: return "directory"
            return None
        except Exception as e:
            logger.exception("Error in show_global_upload_menu:")
            return None

    # --- Other Dialogs ---
    def show_edit_personality(self) -> Optional[str]:
        logger.debug("DialogService: Showing edit personality dialog.")
        try:
            current_prompt = self.chat_manager.get_current_personality()
            dialog = EditPersonalityDialog(current_prompt, self.parent_window)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                return dialog.get_prompt_text()
            # If cancelled, MainWindow will update status if needed
            return None
        except Exception as e:
            logger.exception("Error showing EditPersonalityDialog:")
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Personality Editor:\n{e}")
            return None

    def show_code_viewer(self, ensure_creation: bool = True) -> Optional[CodeViewerWindow]:
        logger.debug(f"DialogService: Showing code viewer (ensure_creation={ensure_creation}).")
        try:
            if self._code_viewer_window is None and ensure_creation:
                self._code_viewer_window = CodeViewerWindow(self.parent_window)
            if self._code_viewer_window:
                self._code_viewer_window.show()  # show() is non-blocking
                self._code_viewer_window.activateWindow()
                self._code_viewer_window.raise_()
            return self._code_viewer_window
        except Exception as e:
            logger.exception("Error showing CodeViewerWindow:")
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Code Viewer:\n{e}")
            return None

    def show_rag_viewer(self) -> Optional[RAGViewerDialog]:
        logger.debug("DialogService: Showing RAG viewer.")
        try:
            vector_db_service = getattr(self.chat_manager, '_vector_db_service', None)
            if not isinstance(vector_db_service, VectorDBService):
                logger.error("DialogService: Valid VectorDBService instance not found in ChatManager.")
                QMessageBox.critical(self.parent_window, "Internal Error",
                                     "RAG Viewer cannot be opened:\nVector Database Service not available.")
                return None

            if self._rag_viewer_dialog is None:  # Create if not cached
                self._rag_viewer_dialog = RAGViewerDialog(vector_db_service, self.parent_window)
                # MainWindow will connect the focusRequested signal after getting the instance

            self._rag_viewer_dialog.exec()  # Modal execution
            return self._rag_viewer_dialog  # Return instance if MainWindow needs to reconnect signals
        except Exception as e:
            logger.exception("Error creating or showing RAGViewerDialog:")
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open RAG Viewer:\n{e}")
            return None

    def get_new_project_name(self) -> Optional[str]:
        logger.debug("DialogService: Getting new project name.")
        try:
            project_name_input, ok = QInputDialog.getText(
                self.parent_window, "Create New Project Context",
                "Enter unique project name/ID (e.g., my_cool_project):",
                QLineEdit.EchoMode.Normal, ""
            )
            if ok and project_name_input and project_name_input.strip():
                clean_project_name = project_name_input.strip()
                if clean_project_name.lower() in [constants.GLOBAL_COLLECTION_ID.lower(), "global context", "global"]:
                    QMessageBox.warning(self.parent_window, "Invalid Name",
                                        f"'{clean_project_name}' is a reserved name. Please choose another.")
                    return None
                return clean_project_name
            elif ok:  # User pressed OK but entered empty name
                QMessageBox.warning(self.parent_window, "Invalid Name", "Project name cannot be empty.")
            return None
        except Exception as e:
            logger.exception("Error in get_new_project_name input dialog:")
            QMessageBox.critical(self.parent_window, "Input Dialog Error", f"Could not get project name:\n{e}")
            return None

    def close_non_modal_dialogs(self):
        logger.info("DialogService attempting to close non-modal dialogs.")
        if self._code_viewer_window:
            try:
                self._code_viewer_window.close()  # CodeViewerWindow handles its own hide/ignore logic
            except Exception as e:
                logger.error(f"Error closing CodeViewerWindow: {e}")
        # Other dialogs are modal and will be closed by user interaction or their exec() call returning.