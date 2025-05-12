# core/project_context_manager.py
import logging
import uuid
from typing import List, Dict, Optional, Any

from PyQt6.QtCore import QObject, pyqtSignal

from .models import ChatMessage
from utils import constants  # <-- IMPORT ADDED

logger = logging.getLogger(__name__)


# GLOBAL_CONTEXT_DISPLAY_NAME is now imported from utils.constants
# No local definition needed here.

class ProjectContextManager(QObject):
    active_project_changed = pyqtSignal(str)
    project_list_updated = pyqtSignal(dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._project_histories: Dict[str, List[ChatMessage]] = {}
        self._project_names: Dict[str, str] = {}
        self._current_project_id: Optional[str] = None
        self._active_conversation_history: List[ChatMessage] = []
        logger.info("ProjectContextManager initialized.")
        self._ensure_global_context_initialized()
        if self._current_project_id is None:  # Ensure an active project is set on init
            self.set_active_project(constants.GLOBAL_COLLECTION_ID)

    def _ensure_global_context_initialized(self):
        global_id = constants.GLOBAL_COLLECTION_ID
        if global_id not in self._project_histories:
            logger.debug(f"PCM Initializing internal Global Context (ID: {global_id}).")
            self._project_histories[global_id] = []
            self._project_names[global_id] = constants.GLOBAL_CONTEXT_DISPLAY_NAME  # Use imported

    def create_project(self, project_name_or_id: str) -> Optional[str]:
        if not project_name_or_id or not isinstance(project_name_or_id, str):
            logger.error(f"Attempted to create project with invalid name/id: '{project_name_or_id}'.")
            return None

        if project_name_or_id == constants.GLOBAL_COLLECTION_ID:
            global_id = constants.GLOBAL_COLLECTION_ID
            if global_id not in self._project_histories:
                logger.info(f"Global Context (ID: {global_id}) not found, creating...")
                self._project_histories[global_id] = []
                self._project_names[global_id] = constants.GLOBAL_CONTEXT_DISPLAY_NAME  # Use imported
                self.project_list_updated.emit(dict(self._project_names))
            return global_id

        project_name = project_name_or_id.strip()
        if not project_name:
            logger.error("Attempted to create project with empty name.")
            return None

        for pid_existing, name_existing in self._project_names.items():
            if name_existing.lower() == project_name.lower():
                logger.warning(
                    f"Project with name '{project_name}' already exists (ID: {pid_existing}). Cannot create duplicate name.")
                return None

        project_id = uuid.uuid4().hex
        logger.info(f"Creating new project: Name='{project_name}', ID='{project_id}'")
        self._project_histories[project_id] = []
        self._project_names[project_id] = project_name
        self.project_list_updated.emit(dict(self._project_names))
        self.set_active_project(project_id)
        return project_id

    def set_active_project(self, project_id: str) -> bool:
        self._ensure_global_context_initialized()
        effective_project_id = project_id if project_id and isinstance(project_id,
                                                                       str) and project_id.strip() else constants.GLOBAL_COLLECTION_ID

        if effective_project_id not in self._project_histories:
            logger.warning(f"Project ID '{effective_project_id}' not found. Defaulting to Global.")
            effective_project_id = constants.GLOBAL_COLLECTION_ID
            if effective_project_id not in self._project_histories:
                logger.critical(f"CRITICAL: Global context '{effective_project_id}' missing after attempt to default.")
                return False  # Major issue if global cannot be accessed

        if self._current_project_id == effective_project_id and self._active_conversation_history is \
                self._project_histories[effective_project_id]:
            logger.debug(f"Project '{effective_project_id}' is already active with correct history reference.")
            # Optionally re-emit if UI needs to refresh even if ID is same (e.g. initial load)
            # self.active_project_changed.emit(self._current_project_id)
            return True

        logger.info(
            f"Setting active project to: ID='{effective_project_id}', Name='{self._project_names.get(effective_project_id)}'")
        old_project_id = self._current_project_id
        self._current_project_id = effective_project_id
        self._active_conversation_history = self._project_histories[effective_project_id]

        # Only emit if truly changed to prevent potential loops if called redundantly
        if old_project_id != self._current_project_id:
            self.active_project_changed.emit(self._current_project_id)
        elif old_project_id is None and self._current_project_id is not None:  # Case for initial setting
            self.active_project_changed.emit(self._current_project_id)

        return True

    def get_active_project_id(self) -> str:  # Return type changed to str, as it defaults to global
        if self._current_project_id is None:
            logger.warning("get_active_project_id: _current_project_id is None. Setting to Global.")
            self.set_active_project(constants.GLOBAL_COLLECTION_ID)  # This sets and returns True/False
        return self._current_project_id if self._current_project_id else constants.GLOBAL_COLLECTION_ID

    def get_active_project_name(self) -> Optional[str]:
        active_id = self.get_active_project_id()
        return self._project_names.get(active_id)

    def get_active_conversation_history(self) -> List[ChatMessage]:
        # Ensures _active_conversation_history is correctly referenced
        active_id = self.get_active_project_id()  # This call ensures _current_project_id is set
        # if self._active_conversation_history is not self._project_histories.get(active_id):
        #     logger.warning(f"Active history ref mismatch for '{active_id}'. Correcting.")
        #     self._active_conversation_history = self._project_histories.get(active_id, [])
        return self._active_conversation_history

    def add_message_to_active_project(self, message: ChatMessage):
        # get_active_conversation_history now returns the direct list reference
        active_history_list = self.get_active_conversation_history()
        if not isinstance(message, ChatMessage):
            logger.error(f"Attempted to add invalid message type: {type(message)}")
            return
        active_history_list.append(message)
        # logger.debug(f"Message (Role: {message.role}) added to project '{self.get_active_project_id()}'. History length: {len(active_history_list)}")

    def get_project_history(self, project_id: str) -> Optional[List[ChatMessage]]:
        if project_id == constants.GLOBAL_COLLECTION_ID and project_id not in self._project_histories:
            self._ensure_global_context_initialized()
        return self._project_histories.get(project_id)

    def get_project_name(self, project_id: str) -> Optional[str]:
        if project_id == constants.GLOBAL_COLLECTION_ID and project_id not in self._project_names:
            self._ensure_global_context_initialized()
        return self._project_names.get(project_id)

    def get_all_projects_info(self) -> Dict[str, str]:
        self._ensure_global_context_initialized()
        return dict(self._project_names)

    def delete_project(self, project_id: str) -> bool:
        if project_id == constants.GLOBAL_COLLECTION_ID:
            logger.error("Cannot delete the Global Context project.")
            return False

        if project_id not in self._project_histories:
            logger.warning(f"Cannot delete project '{project_id}': Not found.")
            return False

        logger.info(f"Deleting project: ID='{project_id}', Name='{self._project_names.get(project_id)}'")
        was_active = (self._current_project_id == project_id)

        del self._project_histories[project_id]
        if project_id in self._project_names:
            del self._project_names[project_id]

        self.project_list_updated.emit(dict(self._project_names))

        if was_active:
            logger.info(f"Deleted project '{project_id}' was active. Setting active project to Global.")
            self.set_active_project(constants.GLOBAL_COLLECTION_ID)
        return True

    def load_state(self, project_context_data: Dict[str, Any]):
        logger.info("Loading state into ProjectContextManager...")
        self._project_histories = {}
        self._project_names = {}
        self._current_project_id = None
        self._active_conversation_history = []

        if not isinstance(project_context_data, dict):
            logger.error(
                f"Invalid project_context_data format: Expected dict, got {type(project_context_data)}. State not loaded.")
        else:
            loaded_histories = project_context_data.get("project_histories", {})
            if isinstance(loaded_histories, dict):
                self._project_histories = loaded_histories
            else:
                logger.warning("Project histories missing or invalid in loaded data.")

            loaded_names = project_context_data.get("project_names", {})
            if isinstance(loaded_names, dict):
                self._project_names = loaded_names
            else:
                logger.warning("Project names missing or invalid in loaded data.")

        self._ensure_global_context_initialized()
        self.project_list_updated.emit(dict(self._project_names))

        loaded_current_project_id = project_context_data.get("current_project_id") if isinstance(project_context_data,
                                                                                                 dict) else None
        active_to_set = loaded_current_project_id if loaded_current_project_id and isinstance(loaded_current_project_id,
                                                                                              str) and loaded_current_project_id in self._project_histories else constants.GLOBAL_COLLECTION_ID

        self.set_active_project(
            active_to_set)  # This sets self._current_project_id and self._active_conversation_history
        logger.info(f"ProjectContextManager state loaded. Active project set to: {self._current_project_id}")

    def save_state(self) -> Dict[str, Any]:
        self._ensure_global_context_initialized()
        active_id_to_save = self.get_active_project_id()  # Ensure it's valid

        logger.info(
            f"Saving ProjectContextManager state. Active Project: {active_id_to_save}, Histories: {len(self._project_histories)}")
        return {
            "project_histories": self._project_histories,
            "project_names": self._project_names,
            "current_project_id": active_id_to_save
        }