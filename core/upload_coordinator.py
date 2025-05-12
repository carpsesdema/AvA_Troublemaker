# core/upload_coordinator.py
import logging
import asyncio
import os # For os.path.basename
from typing import List, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal

# Assuming ChatMessage, UploadService, ProjectContextManager are accessible
try:
    from .models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE
    from services.upload_service import UploadService # Adjust path as needed
    from .project_context_manager import ProjectContextManager # Adjust path as needed
    from utils import constants # For GLOBAL_COLLECTION_ID
except ImportError as e:
    logging.critical(f"UploadCoordinator: Failed to import critical dependencies: {e}")
    # Define fallbacks for type hinting or basic functionality
    ChatMessage = type("ChatMessage", (object,), {})
    UploadService = type("UploadService", (object,), {})
    ProjectContextManager = type("ProjectContextManager", (object,), {})
    constants = type("constants", (object,), {"GLOBAL_COLLECTION_ID": "global_collection"})
    SYSTEM_ROLE, ERROR_ROLE = "system", "error"

logger = logging.getLogger(__name__)

class UploadCoordinator(QObject):
    """
    Manages RAG document upload processes, coordinating with UploadService
    and ProjectContextManager.
    """

    upload_started = pyqtSignal(bool, str) # (is_global_upload, item_description)
    upload_summary_received = pyqtSignal(ChatMessage)
    upload_error = pyqtSignal(str)
    busy_state_changed = pyqtSignal(bool)

    def __init__(self,
                 upload_service: UploadService,
                 project_context_manager: ProjectContextManager, # Added type hint
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        if not upload_service:
            raise ValueError("UploadCoordinator requires a valid UploadService instance.")
        if not project_context_manager:
            raise ValueError("UploadCoordinator requires a valid ProjectContextManager instance.")

        self._upload_service = upload_service
        self._project_context_manager = project_context_manager
        self._current_upload_task: Optional[asyncio.Task] = None
        self._is_busy: bool = False
        logger.info("UploadCoordinator initialized.")

    def _set_busy(self, busy: bool):
        if self._is_busy != busy:
            self._is_busy = busy
            self.busy_state_changed.emit(self._is_busy)
        logger.debug(f"UploadCoordinator busy state set to: {self._is_busy}")

    async def _internal_process_upload(self, upload_func: Callable[[], Optional[ChatMessage]], operation_description: str):
        logger.info(f"UploadCoordinator: Starting async task for: {operation_description}")
        # Note: _set_busy(True) is called by the public methods before creating the task
        summary_message = None
        try:
            summary_message = await asyncio.to_thread(upload_func)
        except asyncio.CancelledError:
            logger.info(f"Upload task '{operation_description}' cancelled by request.")
            # Create a system message for cancellation. ChatManager will add this.
            summary_message = ChatMessage(role=SYSTEM_ROLE, parts=["[Upload cancelled by user.]"],
                                          metadata={"is_cancellation_summary": True})
        except Exception as e:
            logger.exception(f"Error during upload task '{operation_description}': {e}")
            self.upload_error.emit(f"Failed during {operation_description}: {e}")
            # Create an error message. ChatManager will add this.
            summary_message = ChatMessage(role=ERROR_ROLE, parts=[f"Upload Error for {operation_description}: {e}"],
                                          metadata={"is_error_summary": True})
        finally:
            if self._current_upload_task is asyncio.current_task(): # Check if this is the task being cleared
                self._current_upload_task = None
            self._set_busy(False) # Always set busy to false when task finishes
            if summary_message:
                self.upload_summary_received.emit(summary_message)
            logger.info(f"UploadCoordinator: Async task for '{operation_description}' finished.")

    def _initiate_upload(self, upload_callable: Callable[[], Optional[ChatMessage]], description: str, is_global: bool, item_info: str):
        if self._is_busy:
            logger.warning("UploadCoordinator is already busy. Ignoring new upload request.")
            self.upload_error.emit("Upload processor busy. Please wait.")
            return

        self._set_busy(True) # Set busy before creating task
        self.upload_started.emit(is_global, item_info) # Emit signal after setting busy

        self._current_upload_task = asyncio.create_task(
            self._internal_process_upload(upload_callable, description)
        )

    def upload_files_to_current_project(self, file_paths: List[str]):
        if not file_paths: return
        active_project_id = self._project_context_manager.get_active_project_id() or constants.GLOBAL_COLLECTION_ID
        logger.info(f"UploadCoordinator: Request to upload {len(file_paths)} files to project '{active_project_id}'.")
        upload_callable = lambda: self._upload_service.process_files_for_context(file_paths, collection_id=active_project_id)
        description = f"uploading {len(file_paths)} files to '{active_project_id}'"
        self._initiate_upload(upload_callable, description, is_global=(active_project_id == constants.GLOBAL_COLLECTION_ID), item_info=f"{len(file_paths)} file(s)")

    def upload_directory_to_current_project(self, dir_path: str):
        if not dir_path: return
        active_project_id = self._project_context_manager.get_active_project_id() or constants.GLOBAL_COLLECTION_ID
        dir_name = os.path.basename(dir_path)
        logger.info(f"UploadCoordinator: Request to upload directory '{dir_name}' to project '{active_project_id}'.")
        upload_callable = lambda: self._upload_service.process_directory_for_context(dir_path, collection_id=active_project_id)
        description = f"uploading directory '{dir_name}' to '{active_project_id}'"
        self._initiate_upload(upload_callable, description, is_global=(active_project_id == constants.GLOBAL_COLLECTION_ID), item_info=f"directory '{dir_name}'")

    def upload_files_to_global(self, file_paths: List[str]):
        if not file_paths: return
        logger.info(f"UploadCoordinator: Request to upload {len(file_paths)} files to GLOBAL context.")
        upload_callable = lambda: self._upload_service.process_files_for_context(file_paths, collection_id=constants.GLOBAL_COLLECTION_ID)
        description = f"uploading {len(file_paths)} files to GLOBAL"
        self._initiate_upload(upload_callable, description, is_global=True, item_info=f"{len(file_paths)} file(s)")

    def upload_directory_to_global(self, dir_path: str):
        if not dir_path: return
        dir_name = os.path.basename(dir_path)
        logger.info(f"UploadCoordinator: Request to upload directory '{dir_name}' to GLOBAL context.")
        upload_callable = lambda: self._upload_service.process_directory_for_context(dir_path, collection_id=constants.GLOBAL_COLLECTION_ID)
        description = f"uploading directory '{dir_name}' to GLOBAL"
        self._initiate_upload(upload_callable, description, is_global=True, item_info=f"directory '{dir_name}'")

    def cancel_current_upload(self):
        if self._current_upload_task and not self._current_upload_task.done():
            logger.info("UploadCoordinator: Cancelling ongoing upload task...")
            self._current_upload_task.cancel()
            logger.debug("Cancellation requested for upload task.")
        else:
            logger.debug("UploadCoordinator: No active upload task to cancel.")
            if self._is_busy: self._set_busy(False)

    def is_busy(self) -> bool:
        return self._is_busy