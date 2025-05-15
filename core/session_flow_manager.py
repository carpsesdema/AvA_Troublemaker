# core/session_flow_manager.py
import logging
import os
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from services.session_service import SessionService
from core.project_context_manager import ProjectContextManager
from core.backend_coordinator import BackendCoordinator # Not directly used here, but good for context
from core.models import ChatMessage # For type hinting if any methods were to handle ChatMessage lists directly
from utils import constants

logger = logging.getLogger(__name__)


class SessionFlowManager(QObject):
    # session_loaded: ChatManager's _handle_sfm_session_loaded unpacks session_extra_data from proj_ctx_data
    session_loaded = pyqtSignal(str, str, dict, str)  # model_name, personality, proj_ctx_data, active_pid
    active_history_cleared = pyqtSignal()
    status_update_requested = pyqtSignal(str, str, bool, int)
    error_occurred = pyqtSignal(str, bool)
    # request_state_save: ChatManager provides session_extra_data (which includes generator_model_name)
    request_state_save = pyqtSignal(str, str, dict, dict)  # model, pers, pcm_data, session_extra_data

    def __init__(self,
                 session_service: SessionService,
                 project_context_manager: ProjectContextManager,
                 backend_coordinator: BackendCoordinator, # Added for completeness, though not directly used
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        if not all([session_service, project_context_manager, backend_coordinator]): # Added backend_coordinator to check
            err_msg = "SessionFlowManager requires SessionService, ProjectContextManager, and BackendCoordinator."
            logger.critical(err_msg)
            raise ValueError(err_msg)
        self._session_service = session_service
        self._project_context_manager = project_context_manager
        self._backend_coordinator = backend_coordinator # Store it
        self._current_session_filepath: Optional[str] = None
        logger.info("SessionFlowManager initialized.")

    def get_current_session_filepath(self) -> Optional[str]:
        return self._current_session_filepath

    def set_current_session_filepath(self, filepath: Optional[str]):
        self._current_session_filepath = filepath

    def load_last_session_state_on_startup(self) -> Tuple[
        Optional[str], Optional[str], Optional[Dict[str, Any]], str, Optional[str], Optional[float], Optional[str] # Added Generator Model
    ]:
        """
        Loads state from the last session file specifically for application startup.
        Returns:
            Tuple: (model_name, personality_prompt, project_context_data_dict,
                    active_project_id_from_session, active_chat_backend_id, chat_temperature, generator_model_name)
        """
        logger.info("SFM: Loading last session state for startup...")
        active_project_id_from_session = constants.GLOBAL_COLLECTION_ID
        active_chat_backend_id = None
        chat_temperature = None
        generator_model_name = None # <-- NEW

        try:
            # SessionService.get_last_session() now returns (model, pers, project_data, session_extra_data)
            model, personality, project_data, session_extra_data = self._session_service.get_last_session()
            if project_data:
                active_project_id_from_session = project_data.get("current_project_id", constants.GLOBAL_COLLECTION_ID)

            if session_extra_data:
                active_chat_backend_id = session_extra_data.get("active_chat_backend_id")
                chat_temperature = session_extra_data.get("chat_temperature")
                if chat_temperature is not None:
                    try:
                        chat_temperature = float(chat_temperature)
                    except (ValueError, TypeError):
                        chat_temperature = None
                generator_model_name = session_extra_data.get("generator_model_name") # <-- GET NEW

            logger.info(
                f"SFM: Last session loaded. Model: {model}, Pers: {'Set' if personality else 'None'}, "
                f"ActivePID: {active_project_id_from_session}, ActiveChatBackend: {active_chat_backend_id}, "
                f"Temp: {chat_temperature}, GeneratorModel: {generator_model_name}"
            )
            return model, personality, project_data, active_project_id_from_session, active_chat_backend_id, chat_temperature, generator_model_name # <-- RETURN NEW
        except Exception as e:
            logger.exception("SFM: Error loading last session state:")
            return None, None, None, constants.GLOBAL_COLLECTION_ID, None, None, None # <-- RETURN NEW (None)

    def start_new_chat_session(self, current_chat_model: str, current_chat_personality: Optional[str],
                               current_session_extra_data: Dict[str, Any]): # current_session_extra_data will include generator_model
        logger.info("SFM: Starting new chat session flow...")
        self.active_history_cleared.emit()
        self._current_session_filepath = None
        if self._project_context_manager:
            all_project_data = self._project_context_manager.save_state()
            # Pass current_session_extra_data (which includes generator_model_name from CM)
            self.request_state_save.emit(current_chat_model, current_chat_personality, all_project_data,
                                         current_session_extra_data)
        self.status_update_requested.emit("New session started.", "#98c379", True, 2000)

    def load_named_session(self, filepath: str,
                           default_chat_backend_id: str):
        logger.info(f"SFM: Attempting to load named session from: {filepath}")
        # SessionService.load_session now returns (model, pers, project_data, session_extra_data)
        # session_extra_data will include generator_model_name if present in the file
        loaded_model, loaded_pers, project_context_data, session_extra_data = self._session_service.load_session(
            filepath)

        if project_context_data is None:
            err_msg = f"Failed to load session from: {os.path.basename(filepath)}"
            self.error_occurred.emit(err_msg, False)
            return

        self._current_session_filepath = filepath
        active_project_id_from_load = project_context_data.get("current_project_id", constants.GLOBAL_COLLECTION_ID)

        # Pass session_extra_data (which includes generator_model_name) to ChatManager
        # via the proj_ctx_data dictionary. ChatManager will extract it.
        if session_extra_data:
            project_context_data["session_extra_data_on_load"] = session_extra_data

        chat_model_to_set = loaded_model or constants.DEFAULT_GEMINI_CHAT_MODEL # Default for chat model
        self.session_loaded.emit(chat_model_to_set, loaded_pers, project_context_data, active_project_id_from_load)
        self.status_update_requested.emit(f"Session '{os.path.basename(filepath)}' loaded.", "#98c379", True, 3000)

    def save_session_as(self, filepath: str, current_chat_model: str, current_chat_personality: Optional[str],
                        session_extra_data: Dict[str, Any]) -> bool: # session_extra_data comes from CM with gen model
        logger.info(f"SFM: Saving session as: {filepath}")
        if not self._project_context_manager:
            self.error_occurred.emit("Cannot save session: ProjectContextManager not available.", True)
            return False
        project_data_to_save = self._project_context_manager.save_state()
        success, actual_fp = self._session_service.save_session(
            filepath, current_chat_model, current_chat_personality, project_data_to_save, session_extra_data
        )
        if success and actual_fp:
            self._current_session_filepath = actual_fp
            self.status_update_requested.emit(f"Session saved to '{os.path.basename(actual_fp)}'.", "#98c379", True,
                                              3000)
            # Request ChatManager to update its internal "last state" to this newly saved state
            self.request_state_save.emit(current_chat_model, current_chat_personality, project_data_to_save,
                                         session_extra_data)
            return True
        else:
            self.error_occurred.emit(f"Failed to save session to {os.path.basename(filepath)}.", False)
            return False

    def save_current_session_to_last_state(self, current_chat_model: str,
                                           current_chat_personality: Optional[str],
                                           session_extra_data: Optional[Dict[str, Any]] = None):
        if not self._project_context_manager:
            logger.error("SFM: Cannot save last state, ProjectContextManager is missing.")
            return
        logger.debug("SFM: Saving current state to .last_session_state.json")
        project_context_data_to_save = self._project_context_manager.save_state()
        # session_extra_data (provided by ChatManager) now includes generator_model_name
        self._session_service.save_last_session(
            model_name=current_chat_model,
            personality=current_chat_personality,
            project_context_data=project_context_data_to_save,
            session_extra_data=session_extra_data
        )

    def delete_named_session(self, filepath: str) -> bool:
        logger.info(f"SFM: Deleting named session: {filepath}")
        success = self._session_service.delete_session(filepath)
        if success:
            self.status_update_requested.emit(f"Session '{os.path.basename(filepath)}' deleted.", "#98c379", True, 3000)
            if filepath == self._current_session_filepath:
                self._current_session_filepath = None
        else:
            self.error_occurred.emit(f"Failed to delete session '{os.path.basename(filepath)}'.", False)
        return success

    def list_saved_sessions(self) -> List[str]:
        return self._session_service.list_sessions()