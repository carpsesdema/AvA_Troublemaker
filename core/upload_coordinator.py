# core/upload_coordinator.py
import logging
import asyncio
import os
import re
from typing import List, Optional, Callable, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

try:
    from .models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE
    from services.upload_service import UploadService
    from .project_context_manager import ProjectContextManager
    from utils import constants

    # Conditional import for ProjectSummaryCoordinator for type hinting
    if TYPE_CHECKING:
        from .project_summary_coordinator import ProjectSummaryCoordinator
except ImportError as e:
    logging.critical(f"UploadCoordinator: Failed to import critical dependencies: {e}")
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    UploadService = type("UploadService", (object,), {})  # type: ignore
    ProjectContextManager = type("ProjectContextManager", (object,), {})  # type: ignore
    constants = type("constants", (object,), {"GLOBAL_COLLECTION_ID": "global_collection"})  # type: ignore
    SYSTEM_ROLE, ERROR_ROLE = "system", "error"  # type: ignore
    if TYPE_CHECKING:
        ProjectSummaryCoordinator = type("ProjectSummaryCoordinator", (object,), {})  # type: ignore

logger = logging.getLogger(__name__)

# The AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES constant logic has been removed.
# If this was its only use, the constant can be removed from utils/constants.py.

class UploadCoordinator(QObject):
    upload_started = pyqtSignal(bool, str)
    upload_summary_received = pyqtSignal(ChatMessage)  # This is the RAG processing summary
    upload_error = pyqtSignal(str)
    busy_state_changed = pyqtSignal(bool)

    def __init__(self,
                 upload_service: UploadService,
                 project_context_manager: ProjectContextManager,
                 project_summary_coordinator: Optional['ProjectSummaryCoordinator'], # Remains for manual summary
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        if not upload_service:
            raise ValueError("UploadCoordinator requires a valid UploadService instance.")
        if not project_context_manager:
            raise ValueError("UploadCoordinator requires a valid ProjectContextManager instance.")

        self._upload_service = upload_service
        self._project_context_manager = project_context_manager
        self._project_summary_coordinator = project_summary_coordinator # Keep for manual summary
        self._current_upload_task: Optional[asyncio.Task] = None
        self._is_busy: bool = False

        if self._project_summary_coordinator:
            logger.info("UploadCoordinator initialized with ProjectSummaryCoordinator (for manual summaries).")
        else:
            logger.warning("UploadCoordinator initialized WITHOUT ProjectSummaryCoordinator. Manual summary command will not function.")
        logger.info("UploadCoordinator initialized.")

    def _set_busy(self, busy: bool):
        if self._is_busy != busy:
            self._is_busy = busy
            self.busy_state_changed.emit(self._is_busy)
        logger.debug(f"UploadCoordinator busy state set to: {self._is_busy}")

    async def _internal_process_upload(
            self,
            upload_func: Callable[[], Optional[ChatMessage]],
            operation_description: str,
            target_collection_id: Optional[str] = None, # Still useful for logging/context
            num_items_for_upload: int = 0 # Still useful for logging/context
    ):
        logger.info(f"UploadCoordinator: Starting async task for: {operation_description}")
        summary_message: Optional[ChatMessage] = None
        # The logic for rag_processing_successful_for_some_items and files_added_count
        # specifically for triggering automatic summaries has been removed.

        try:
            # This summary_message is the RAG processing summary from UploadService
            summary_message = await asyncio.to_thread(upload_func)

            # <<< THE ENTIRE AUTOMATIC PROJECT SUMMARY TRIGGER BLOCK HAS BEEN DELETED >>>
            # No conditions checked here to call self._project_summary_coordinator.generate_project_summary

        except asyncio.CancelledError:
            logger.info(f"Upload task '{operation_description}' cancelled by request.")
            summary_message = ChatMessage(role=SYSTEM_ROLE, parts=["[Upload cancelled by user.]"],
                                          metadata={"is_cancellation_summary": True})
        except Exception as e:
            logger.exception(f"Error during upload task '{operation_description}': {e}")
            self.upload_error.emit(f"Failed during {operation_description}: {e}")
            summary_message = ChatMessage(role=ERROR_ROLE, parts=[f"Upload Error for {operation_description}: {e}"],
                                          metadata={"is_error_summary": True})
        finally:
            if self._current_upload_task is asyncio.current_task():
                self._current_upload_task = None
            self._set_busy(False)
            if summary_message:  # This is the RAG processing summary from UploadService
                self.upload_summary_received.emit(summary_message)
            logger.info(f"UploadCoordinator: Async task for '{operation_description}' finished.")

    def _initiate_upload(self,
                         upload_callable: Callable[[], Optional[ChatMessage]],
                         description: str,
                         is_global: bool,
                         item_info: str,
                         target_collection_id_for_summary: Optional[str] = None, # Keep for context
                         num_items_for_upload: int = 0 # Keep for context
                         ):
        if self._is_busy:
            logger.warning("UploadCoordinator is already busy. Ignoring new upload request.")
            self.upload_error.emit("Upload processor busy. Please wait.")
            return

        self._set_busy(True)
        self.upload_started.emit(is_global, item_info)

        self._current_upload_task = asyncio.create_task(
            self._internal_process_upload(
                upload_callable,
                description,
                target_collection_id=target_collection_id_for_summary,
                num_items_for_upload=num_items_for_upload
            )
        )

    def upload_files_to_current_project(self, file_paths: List[str]):
        if not file_paths: return
        active_project_id = self._project_context_manager.get_active_project_id() or constants.GLOBAL_COLLECTION_ID
        logger.info(f"UploadCoordinator: Request to upload {len(file_paths)} files to project '{active_project_id}'.")
        upload_callable = lambda: self._upload_service.process_files_for_context(file_paths,
                                                                                 collection_id=active_project_id)
        description = f"uploading {len(file_paths)} files to '{active_project_id}'"
        self._initiate_upload(
            upload_callable,
            description,
            is_global=(active_project_id == constants.GLOBAL_COLLECTION_ID),
            item_info=f"{len(file_paths)} file(s)",
            target_collection_id_for_summary=active_project_id,
            num_items_for_upload=len(file_paths)
        )

    def upload_directory_to_current_project(self, dir_path: str):
        if not dir_path: return
        active_project_id = self._project_context_manager.get_active_project_id() or constants.GLOBAL_COLLECTION_ID
        dir_name = os.path.basename(dir_path)
        logger.info(f"UploadCoordinator: Request to upload directory '{dir_name}' to project '{active_project_id}'.")
        upload_callable = lambda: self._upload_service.process_directory_for_context(dir_path,
                                                                                     collection_id=active_project_id)
        description = f"uploading directory '{dir_name}' to '{active_project_id}'"
        self._initiate_upload(
            upload_callable,
            description,
            is_global=(active_project_id == constants.GLOBAL_COLLECTION_ID),
            item_info=f"directory '{dir_name}'",
            target_collection_id_for_summary=active_project_id,
            num_items_for_upload=1
        )

    def upload_files_to_global(self, file_paths: List[str]):
        if not file_paths: return
        logger.info(f"UploadCoordinator: Request to upload {len(file_paths)} files to GLOBAL context.")
        upload_callable = lambda: self._upload_service.process_files_for_context(file_paths,
                                                                                 collection_id=constants.GLOBAL_COLLECTION_ID)
        description = f"uploading {len(file_paths)} files to GLOBAL"
        self._initiate_upload(
            upload_callable,
            description,
            is_global=True,
            item_info=f"{len(file_paths)} file(s)",
            target_collection_id_for_summary=constants.GLOBAL_COLLECTION_ID,
            num_items_for_upload=len(file_paths)
        )

    def upload_directory_to_global(self, dir_path: str):
        if not dir_path: return
        dir_name = os.path.basename(dir_path)
        logger.info(f"UploadCoordinator: Request to upload directory '{dir_name}' to GLOBAL context.")
        upload_callable = lambda: self._upload_service.process_directory_for_context(dir_path,
                                                                                     collection_id=constants.GLOBAL_COLLECTION_ID)
        description = f"uploading directory '{dir_name}' to GLOBAL"
        self._initiate_upload(
            upload_callable,
            description,
            is_global=True,
            item_info=f"directory '{dir_name}'",
            target_collection_id_for_summary=constants.GLOBAL_COLLECTION_ID,
            num_items_for_upload=1
        )

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