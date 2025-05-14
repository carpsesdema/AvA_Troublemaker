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

# Threshold for automatically triggering a project summary after upload
# For example, if more than 5 new files are successfully processed for a project's RAG.
AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES = 5


class UploadCoordinator(QObject):
    upload_started = pyqtSignal(bool, str)
    upload_summary_received = pyqtSignal(ChatMessage)  # This is the RAG processing summary
    upload_error = pyqtSignal(str)
    busy_state_changed = pyqtSignal(bool)

    # No new signal needed here to request summary, it will call PSC directly

    def __init__(self,
                 upload_service: UploadService,
                 project_context_manager: ProjectContextManager,
                 project_summary_coordinator: Optional['ProjectSummaryCoordinator'],  # Added
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        if not upload_service:
            raise ValueError("UploadCoordinator requires a valid UploadService instance.")
        if not project_context_manager:
            raise ValueError("UploadCoordinator requires a valid ProjectContextManager instance.")

        self._upload_service = upload_service
        self._project_context_manager = project_context_manager
        self._project_summary_coordinator = project_summary_coordinator  # Store it
        self._current_upload_task: Optional[asyncio.Task] = None
        self._is_busy: bool = False

        if self._project_summary_coordinator:
            logger.info("UploadCoordinator initialized with ProjectSummaryCoordinator.")
        else:
            logger.warning("UploadCoordinator initialized WITHOUT ProjectSummaryCoordinator. Auto-summary disabled.")
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
            target_collection_id: Optional[str] = None,  # To know which project was affected
            num_items_for_upload: int = 0  # Number of files/items in this specific upload operation
    ):
        logger.info(f"UploadCoordinator: Starting async task for: {operation_description}")
        summary_message: Optional[ChatMessage] = None
        rag_processing_successful_for_some_items = False

        try:
            # This summary_message is the RAG processing summary from UploadService
            summary_message = await asyncio.to_thread(upload_func)
            if summary_message and summary_message.role != ERROR_ROLE:
                # Check metadata for how many files were actually added successfully
                # (UploadService's summary_message should ideally contain this info)
                files_added_count = 0
                if summary_message.metadata and "upload_summary" in summary_message.metadata:
                    # Example: "upload_summary": "5/7 processed to DB..."
                    try:
                        match = re.search(r"(\d+)/\d+ processed to DB", str(summary_message.metadata["upload_summary"]))
                        if match:
                            files_added_count = int(match.group(1))
                    except Exception:
                        pass  # Ignore parsing errors for this

                if files_added_count > 0:
                    rag_processing_successful_for_some_items = True

                # --- Trigger Project Summary if conditions met ---
                if target_collection_id and target_collection_id != constants.GLOBAL_COLLECTION_ID and \
                        rag_processing_successful_for_some_items and self._project_summary_coordinator:

                    # Heuristic: Trigger if a significant number of files were added in this batch
                    if files_added_count >= AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES:
                        logger.info(
                            f"Upload to project '{target_collection_id}' added {files_added_count} files (>= threshold {AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES}). "
                            f"Requesting project summary via ProjectSummaryCoordinator.")
                        try:
                            # Use QTimer to ensure this call happens on the main thread if PSC has Qt affinity
                            # or if PSC might interact with UI-related components indirectly.
                            # For now, direct call as PSC is a QObject but its work is mostly backend.
                            QTimer.singleShot(0, lambda: self._project_summary_coordinator.generate_project_summary(
                                target_collection_id))
                        except Exception as e_psc_call:
                            logger.error(
                                f"Error trying to trigger project summary for '{target_collection_id}': {e_psc_call}")
                    else:
                        logger.info(
                            f"Upload to project '{target_collection_id}' added {files_added_count} files (< threshold {AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES}). "
                            f"Automatic project summary not triggered.")
                # --- End Trigger Project Summary ---

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
            if summary_message:  # This is the RAG processing summary
                self.upload_summary_received.emit(summary_message)
            logger.info(f"UploadCoordinator: Async task for '{operation_description}' finished.")

    def _initiate_upload(self,
                         upload_callable: Callable[[], Optional[ChatMessage]],
                         description: str,
                         is_global: bool,
                         item_info: str,
                         target_collection_id_for_summary: Optional[str] = None,  # Pass the project ID
                         num_items_for_upload: int = 0
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

        # Note: To get num_items_for_upload for a directory, we'd ideally scan it first.
        # For simplicity, we'll pass 0 and let UploadService determine actual count for summary trigger.
        # Or, UploadService can provide the count in its summary_message.
        # Let's assume UploadService's summary message will contain enough info to gauge significance.
        upload_callable = lambda: self._upload_service.process_directory_for_context(dir_path,
                                                                                     collection_id=active_project_id)
        description = f"uploading directory '{dir_name}' to '{active_project_id}'"
        self._initiate_upload(
            upload_callable,
            description,
            is_global=(active_project_id == constants.GLOBAL_COLLECTION_ID),
            item_info=f"directory '{dir_name}'",
            target_collection_id_for_summary=active_project_id,
            num_items_for_upload=1  # Treat directory as one "item" for this high-level count
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
            target_collection_id_for_summary=constants.GLOBAL_COLLECTION_ID,  # Global summaries might not be desired
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