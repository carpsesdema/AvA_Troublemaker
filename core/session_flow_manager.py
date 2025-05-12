# core/session_flow_manager.py
import logging
import os
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

# Assuming services and core components are accessible
from services.session_service import SessionService
from core.project_context_manager import ProjectContextManager
from core.backend_coordinator import BackendCoordinator  # To reconfigure backends on load
from core.models import ChatMessage  # For type hinting if needed, though less direct use here
from utils import constants  # For default model names, GLOBAL_COLLECTION_ID

logger = logging.getLogger(__name__)


class SessionFlowManager(QObject):
    """
    Manages the lifecycle of chat sessions, including loading, saving,
    starting new sessions, and interacting with the SessionService.
    """

    # --- Signals to ChatManager ---
    # Emitted when a session is successfully loaded and ChatManager needs to update its state
    # (new_model_name, new_personality, project_context_data, session_filepath)
    session_loaded = pyqtSignal(str, str, dict, str)

    # Emitted when history for the active project should be cleared (e.g., new session)
    active_history_cleared = pyqtSignal()

    # Emitted to request a status update message in the UI
    status_update_requested = pyqtSignal(str, str, bool, int)  # msg, color, temporary, duration

    # Emitted when an error occurs that should be shown to the user
    error_occurred = pyqtSignal(str, bool)  # msg, is_critical

    # Signals that the overall state (model, personality, all project contexts) needs to be saved by ChatManager
    # (current_chat_model, current_chat_personality, all_project_data_from_pcm)
    request_state_save = pyqtSignal(str, str, dict)

    def __init__(self,
                 session_service: SessionService,
                 project_context_manager: ProjectContextManager,
                 backend_coordinator: BackendCoordinator,  # Needed to reconfigure on load
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([session_service, project_context_manager, backend_coordinator]):
            err_msg = "SessionFlowManager requires SessionService, ProjectContextManager, and BackendCoordinator."
            logger.critical(err_msg)
            raise ValueError(err_msg)

        self._session_service = session_service
        self._project_context_manager = project_context_manager
        self._backend_coordinator = backend_coordinator

        self._current_session_filepath: Optional[str] = None  # Path to the currently loaded *named* session

        logger.info("SessionFlowManager initialized.")

    def get_current_session_filepath(self) -> Optional[str]:
        return self._current_session_filepath

    def set_current_session_filepath(self, filepath: Optional[str]):
        self._current_session_filepath = filepath

    def load_last_session_state_on_startup(self) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], str]:
        """
        Loads state from the last session file specifically for application startup.
        This is called by ChatManager during its initialization.

        Returns:
            Tuple: (model_name, personality_prompt, project_context_data_dict, active_project_id_from_session)
                   project_context_data_dict is None if loading fails.
                   active_project_id_from_session defaults to GLOBAL_COLLECTION_ID.
        """
        logger.info("SFM: Loading last session state for startup...")
        active_project_id_from_session = constants.GLOBAL_COLLECTION_ID
        try:
            model, personality, project_data = self._session_service.get_last_session()
            if project_data:
                # ProjectContextManager will handle the actual loading of its state.
                # We just need to extract the active_project_id if present.
                active_project_id_from_session = project_data.get("current_project_id", constants.GLOBAL_COLLECTION_ID)
                logger.info(
                    f"SFM: Last session loaded. Model: {model}, Pers: {'Set' if personality else 'None'}, ActivePID: {active_project_id_from_session}")
            else:
                logger.info("SFM: No last session data found or error during load.")
            return model, personality, project_data, active_project_id_from_session
        except Exception as e:
            logger.exception("SFM: Error loading last session state:")
            return None, None, None, constants.GLOBAL_COLLECTION_ID

    def start_new_chat_session(self, current_chat_model: str, current_chat_personality: Optional[str]):
        """
        Handles the logic for starting a new chat session.
        Clears the active project's history and resets current named session path.
        """
        logger.info("SFM: Starting new chat session flow...")
        # 1. Clear history in ProjectContextManager for the currently active project
        # This will be done by ChatManager listening to active_history_cleared
        self.active_history_cleared.emit()

        # 2. Reset the current named session filepath
        self._current_session_filepath = None

        # 3. Request ChatManager to save this new (cleared) state
        # It's important to save the state after clearing so the ".last_session_state.json" reflects this.
        # ChatManager will call ProjectContextManager.save_state() to get the latest project data.
        if self._project_context_manager:
            all_project_data = self._project_context_manager.save_state()
            self.request_state_save.emit(current_chat_model, current_chat_personality, all_project_data)

        self.status_update_requested.emit("New session started.", "#98c379", True, 2000)

    def load_named_session(self, filepath: str, default_chat_backend_id: str):
        """
        Loads a specific named session from a file.
        Emits session_loaded signal on success.
        """
        logger.info(f"SFM: Attempting to load named session from: {filepath}")
        loaded_model, loaded_pers, project_context_data = self._session_service.load_session(filepath)

        if project_context_data is None:  # Indicates failure to load
            err_msg = f"Failed to load session from: {os.path.basename(filepath)}"
            self.error_occurred.emit(err_msg, False)
            return

        self._current_session_filepath = filepath  # Store path of successfully loaded named session

        # Determine the active project ID from the loaded data
        active_project_id_from_load = project_context_data.get("current_project_id", constants.GLOBAL_COLLECTION_ID)

        # Emit signal for ChatManager to handle the rest (updating PCM, configuring backends)
        # Provide the default chat model or the one from session.
        chat_model_to_set = loaded_model or constants.DEFAULT_GEMINI_CHAT_MODEL  # Fallback

        self.session_loaded.emit(
            chat_model_to_set,
            loaded_pers,
            project_context_data,  # The full dict for PCM
            active_project_id_from_load  # The active project ID from this session
        )
        self.status_update_requested.emit(f"Session '{os.path.basename(filepath)}' loaded.", "#98c379", True, 3000)

    def save_session_as(self, filepath: str, current_chat_model: str, current_chat_personality: Optional[str]) -> bool:
        """
        Saves the current state (all project contexts) to a new named session file.
        """
        logger.info(f"SFM: Saving session as: {filepath}")
        if not self._project_context_manager:
            self.error_occurred.emit("Cannot save session: ProjectContextManager not available.", True)
            return False

        project_data_to_save = self._project_context_manager.save_state()

        success, actual_fp = self._session_service.save_session(
            filepath,
            current_chat_model,
            current_chat_personality,
            project_data_to_save
        )
        if success and actual_fp:
            self._current_session_filepath = actual_fp  # Update current path to the new save location
            self.status_update_requested.emit(f"Session saved to '{os.path.basename(actual_fp)}'.", "#98c379", True,
                                              3000)
            # Also request a save to the ".last_session_state.json" to reflect this new state as "last"
            self.request_state_save.emit(current_chat_model, current_chat_personality, project_data_to_save)
            return True
        else:
            self.error_occurred.emit(f"Failed to save session to {os.path.basename(filepath)}.", False)
            return False

    def save_current_session_to_last_state(self, current_chat_model: str, current_chat_personality: Optional[str]):
        """
        Saves the current application state (active model, personality, all project data)
        to the ".last_session_state.json" file.
        This is typically called by ChatManager when its internal state changes.
        """
        if not self._project_context_manager:
            logger.error("SFM: Cannot save last state, ProjectContextManager is missing.")
            return

        logger.debug("SFM: Saving current state to .last_session_state.json")
        project_context_data_to_save = self._project_context_manager.save_state()
        self._session_service.save_last_session(
            model_name=current_chat_model,
            personality=current_chat_personality,
            project_context_data=project_context_data_to_save
        )

    def delete_named_session(self, filepath: str) -> bool:
        """Deletes a specific named session file."""
        logger.info(f"SFM: Deleting named session: {filepath}")
        success = self._session_service.delete_session(filepath)
        if success:
            self.status_update_requested.emit(f"Session '{os.path.basename(filepath)}' deleted.", "#98c379", True, 3000)
            if filepath == self._current_session_filepath:
                self._current_session_filepath = None  # Clear if the deleted session was the active named one
                # ChatManager should then save state to update .last_session_state.json if needed
        else:
            self.error_occurred.emit(f"Failed to delete session '{os.path.basename(filepath)}'.", False)
        return success

    def list_saved_sessions(self) -> List[str]:
        """Returns a list of filepaths for all saved named sessions."""
        return self._session_service.list_sessions()