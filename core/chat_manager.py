# core/chat_manager.py
import logging
import asyncio
import os
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot

from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from backend.interface import BackendInterface

# --- Service Imports ---
from services.code_summary_service import CodeSummaryService
from services.model_info_service import ModelInfoService
# Types for services obtained from orchestrator (if needed for hints)
from services.session_service import SessionService
from services.upload_service import UploadService, VECTOR_DB_SERVICE_AVAILABLE
from services.vector_db_service import VectorDBService

# --- Core Component Imports (for type hints and direct use) ---
from .project_context_manager import ProjectContextManager
from .backend_coordinator import BackendCoordinator
from .session_flow_manager import SessionFlowManager
from .upload_coordinator import UploadCoordinator
from .rag_handler import RagHandler
from .user_input_handler import UserInputHandler
from .user_input_processor import UserInputProcessor  # May not be directly used by CM if UIH handles it
from .application_orchestrator import ApplicationOrchestrator

# Conditional imports for modification features
try:
    from .modification_handler import ModificationHandler

    MOD_HANDLER_AVAILABLE = True
except ImportError:
    ModificationHandler = None  # type: ignore
    MOD_HANDLER_AVAILABLE = False
try:
    from .modification_coordinator import ModificationCoordinator, ModPhase

    MOD_COORDINATOR_AVAILABLE = True
except ImportError:
    ModificationCoordinator = None  # type: ignore
    ModPhase = None  # type: ignore
    MOD_COORDINATOR_AVAILABLE = False

from utils import constants

try:
    from config import get_api_key
except ImportError:
    def get_api_key():  # type: ignore
        return None


    logging.info("config.py not found or get_api_key not defined, using dummy for API key.")

logger = logging.getLogger(__name__)

# Backend ID constants (still used by ChatManager for logic)
DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
OLLAMA_CHAT_BACKEND_ID = "ollama_chat"
PLANNER_BACKEND_ID = "gemini_planner"
GENERATOR_BACKEND_ID = "ollama_generator"


# --- CONSTANTS REMOVED ---
# PLANNER_PROMPT_TEMPLATE_FOR_SUMMARY is now in CodeSummaryService

class ChatManager(QObject):
    # --- Signals (remain unchanged) ---
    history_changed = pyqtSignal(list)
    new_message_added = pyqtSignal(object)
    status_update = pyqtSignal(str, str, bool, int)
    error_occurred = pyqtSignal(str, bool)
    busy_state_changed = pyqtSignal(bool)
    config_state_changed = pyqtSignal(str, bool)
    stream_chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()
    stream_started = pyqtSignal(str)
    code_file_updated = pyqtSignal(str, str)
    current_project_changed = pyqtSignal(str)
    project_inventory_updated = pyqtSignal(dict)
    available_models_changed = pyqtSignal(list)
    token_usage_updated = pyqtSignal(str, int, int, int)

    def __init__(self, orchestrator: ApplicationOrchestrator, parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ChatManager initializing with ApplicationOrchestrator...")

        if not isinstance(orchestrator, ApplicationOrchestrator):
            err_msg = "ChatManager requires a valid ApplicationOrchestrator instance."
            logger.critical(err_msg)
            raise TypeError(err_msg)

        self._orchestrator = orchestrator

        # Get components from orchestrator
        self._backend_adapters_dict = self._orchestrator.get_all_backend_adapters_dict()
        self._project_context_manager = self._orchestrator.get_project_context_manager()
        if self._project_context_manager: self._project_context_manager.setParent(self)
        self._backend_coordinator = self._orchestrator.get_backend_coordinator()
        if self._backend_coordinator: self._backend_coordinator.setParent(self)
        self._session_flow_manager = self._orchestrator.get_session_flow_manager()
        if self._session_flow_manager: self._session_flow_manager.setParent(self)
        self._upload_coordinator = self._orchestrator.get_upload_coordinator()
        if self._upload_coordinator: self._upload_coordinator.setParent(self)
        self._user_input_handler = self._orchestrator.get_user_input_handler()
        if self._user_input_handler: self._user_input_handler.setParent(self)
        self._modification_coordinator = self._orchestrator.get_modification_coordinator()
        if self._modification_coordinator: self._modification_coordinator.setParent(self)

        self._rag_handler = self._orchestrator.get_rag_handler()  # Keep reference if needed
        self._modification_handler_instance = self._orchestrator.get_modification_handler_instance()
        if self._modification_handler_instance and isinstance(self._modification_handler_instance, QObject):
            if self._modification_handler_instance.parent() is None:
                self._modification_handler_instance.setParent(self)

        # Get service instances (needed for some direct checks or operations)
        self._session_service: Optional[SessionService] = getattr(orchestrator, '_session_service', None)
        self._vector_db_service: Optional[VectorDBService] = getattr(orchestrator, '_vector_db_service', None)

        # Instantiate new services that ChatManager uses directly
        self._code_summary_service = CodeSummaryService()
        self._model_info_service = ModelInfoService()

        self._connect_component_signals()
        self._initialize_state_variables()
        logger.info("ChatManager core initialization using orchestrator complete.")

    # --- INITIALIZATION METHODS REMOVED ---
    # _initialize_adapters() - Moved to ApplicationOrchestrator
    # _initialize_coordinators_and_handlers() - Moved to ApplicationOrchestrator

    def _connect_component_signals(self):
        logger.debug("ChatManager connecting component signals...")
        if self._project_context_manager:
            try:
                self._project_context_manager.project_list_updated.connect(self._handle_pcm_project_list_updated)
                self._project_context_manager.active_project_changed.connect(self._handle_pcm_active_project_changed)
            except Exception as e:
                logger.error(f"Error connecting PCM signals: {e}", exc_info=True)
        else:
            logger.warning("PCM not available for signal connection.")

        if self._backend_coordinator:
            try:
                self._backend_coordinator.stream_chunk_received.connect(self._handle_backend_chunk_received)
                self._backend_coordinator.response_completed.connect(self._handle_backend_response_completed)
                self._backend_coordinator.response_error.connect(self._handle_backend_response_error)
                self._backend_coordinator.busy_state_changed.connect(self._handle_backend_busy_changed)
                self._backend_coordinator.configuration_changed.connect(self._handle_backend_configuration_changed)
            except Exception as e:
                logger.error(f"Error connecting BC signals: {e}", exc_info=True)
        else:
            logger.warning("BC not available for signal connection.")

        if self._upload_coordinator:
            try:
                self._upload_coordinator.upload_started.connect(self._handle_upload_started)
                self._upload_coordinator.upload_summary_received.connect(self._handle_upload_summary)
                self._upload_coordinator.upload_error.connect(self._handle_upload_error)
                self._upload_coordinator.busy_state_changed.connect(self._handle_upload_busy_changed)
            except Exception as e:
                logger.error(f"Error connecting UC signals: {e}", exc_info=True)
        else:
            logger.warning("UC not available for signal connection.")

        if self._session_flow_manager:
            try:
                self._session_flow_manager.session_loaded.connect(self._handle_sfm_session_loaded)
                self._session_flow_manager.active_history_cleared.connect(self._handle_sfm_active_history_cleared)
                self._session_flow_manager.status_update_requested.connect(self.status_update)
                self._session_flow_manager.error_occurred.connect(self.error_occurred)
                self._session_flow_manager.request_state_save.connect(self._handle_sfm_request_state_save)
            except Exception as e:
                logger.error(f"Error connecting SFM signals: {e}", exc_info=True)
        else:
            logger.warning("SFM not available for signal connection.")

        if self._user_input_handler:
            try:
                self._user_input_handler.normal_chat_request_ready.connect(self._handle_uih_normal_chat_request)
                self._user_input_handler.modification_sequence_start_requested.connect(
                    self._handle_uih_mod_start_request)
                self._user_input_handler.modification_user_input_received.connect(self._handle_uih_mod_user_input)
                self._user_input_handler.processing_error_occurred.connect(self._handle_uih_processing_error)
            except Exception as e:
                logger.error(f"Error connecting UIH signals: {e}", exc_info=True)
        else:
            logger.warning("UIH not available for signal connection.")

        if self._modification_coordinator:
            logger.debug("Connecting ModificationCoordinator signals...")
            try:
                self._modification_coordinator.request_llm_call.connect(self._handle_mc_request_llm_call)
                self._modification_coordinator.file_ready_for_display.connect(self._handle_mc_file_ready)
                self._modification_coordinator.modification_sequence_complete.connect(self._handle_mc_sequence_complete)
                self._modification_coordinator.modification_error.connect(self._handle_mc_error)
                self._modification_coordinator.status_update.connect(self._handle_mc_status_update)
                self._modification_coordinator.codeGeneratedAndSummaryNeeded.connect(
                    self._handle_code_generated_and_summary_needed)
                logger.info("ChatManager connected to ModificationCoordinator signals.")
            except Exception as e_connect_mc:
                logger.error(f"Error connecting ModificationCoordinator signals: {e_connect_mc}", exc_info=True)
        elif MOD_COORDINATOR_AVAILABLE:
            logger.warning("ModificationCoordinator was expected but instance not found for signal connection.")
        else:
            logger.debug("ModificationCoordinator not available, skipping its signal connections.")
        logger.debug("ChatManager component signal connection process finished.")

    def _initialize_state_variables(self):
        self._overall_busy: bool = False
        self._current_chat_model_name: str = constants.DEFAULT_GEMINI_CHAT_MODEL
        self._current_chat_personality_prompt: Optional[str] = None
        self._current_chat_temperature: float = 0.7
        self._chat_backend_configured_successfully: bool = False
        self._available_chat_models: List[str] = []
        self._current_chat_focus_paths: Optional[List[str]] = None
        self._rag_available: bool = (self._vector_db_service is not None and self._vector_db_service.is_ready())
        self._rag_initialized: bool = self._rag_available
        logger.debug("ChatManager state variables initialized.")

    def initialize(self):
        logger.info("ChatManager late initialization process starting...")
        if not self._session_flow_manager or not self._project_context_manager or not self._user_input_handler or not self._backend_coordinator:
            critical_missing = [
                name for comp, name in [
                    (self._session_flow_manager, "SFM"),
                    (self._project_context_manager, "PCM"),
                    (self._user_input_handler, "UIH"),
                    (self._backend_coordinator, "BC")
                ] if not comp
            ]
            logger.critical(f"Cannot initialize ChatManager: Critical components missing: {critical_missing}")
            self.error_occurred.emit(
                f"Critical error during ChatManager init ({', '.join(critical_missing)} missing). App may not function.",
                True)
            return

        loaded_model, loaded_personality, project_data_from_sfm, active_pid_from_sfm = \
            self._session_flow_manager.load_last_session_state_on_startup()

        if project_data_from_sfm:
            self._project_context_manager.load_state(project_data_from_sfm)
        else:
            self._project_context_manager.set_active_project(constants.GLOBAL_COLLECTION_ID)

        self._perform_orphan_cleanup(self._project_context_manager.save_state())
        self._set_initial_active_project(active_pid_from_sfm, None)
        self._configure_initial_backends(loaded_model, loaded_personality)
        self.update_status_based_on_state()
        current_active_id = self._project_context_manager.get_active_project_id()
        self._update_rag_initialized_state(emit_status=False, project_id=current_active_id)
        logger.info(
            f"ChatManager late initialization complete. Active project: {current_active_id}, Default Chat Model: {self._current_chat_model_name}")

    def _perform_orphan_cleanup(self, project_context_data_from_pcm: Optional[Dict[str, Any]]):
        if not (self._project_context_manager and self._vector_db_service):
            logger.warning("Skipping orphan cleanup: PCM or VDB service not available.")
            return
        logger.info("Checking for orphaned global_collection entries...")
        orphaned_ids_to_delete = []
        all_projects = self._project_context_manager.get_all_projects_info()
        for pid, name in all_projects.items():
            if name == constants.GLOBAL_CONTEXT_DISPLAY_NAME and pid != constants.GLOBAL_COLLECTION_ID:
                orphaned_ids_to_delete.append(pid)
        if orphaned_ids_to_delete:
            logger.info(f"Attempting to delete {len(orphaned_ids_to_delete)} orphaned entries from PCM and VDB...")
            deleted_count = 0;
            failed_pcm_delete = [];
            failed_vdb_delete = []
            for orphan_id in orphaned_ids_to_delete:
                pcm_deleted, vdb_deleted = False, False
                try:
                    if self._project_context_manager.delete_project(orphan_id): pcm_deleted = True
                except Exception as e_pcm_del:
                    failed_pcm_delete.append(orphan_id); logger.exception(f"PCM orphan delete error: {e_pcm_del}")
                try:
                    if self._vector_db_service.delete_collection(orphan_id): vdb_deleted = True  # type: ignore
                except Exception as e_vdb_del:
                    failed_vdb_delete.append(orphan_id); logger.exception(f"VDB orphan delete error: {e_vdb_del}")
                if pcm_deleted: deleted_count += 1
            if deleted_count > 0: logger.info(f"Successfully processed {deleted_count} orphaned entries for deletion.")
            if failed_pcm_delete or failed_vdb_delete:
                logger.error(f"Cleanup issues: Failed PCM: {failed_pcm_delete}, Failed VDB: {failed_vdb_delete}")
                self.error_occurred.emit("Errors during data cleanup (check logs).", False)
        else:
            logger.info("No orphaned global_collection entries found.")

    def _set_initial_active_project(self, target_active_project_id: str, _):  # Filepath arg unused
        if not self._project_context_manager:
            self.error_occurred.emit("Internal Error: Project Manager missing for init.", True);
            return
        if not self._project_context_manager.get_project_history(target_active_project_id):
            logger.warning(
                f"Initial active project ID '{target_active_project_id}' not found in PCM. Defaulting to Global.")
            target_active_project_id = constants.GLOBAL_COLLECTION_ID
        self._project_context_manager.set_active_project(target_active_project_id)

    def _configure_initial_backends(self, loaded_chat_model: Optional[str], loaded_chat_personality: Optional[str]):
        if not self._backend_coordinator:
            logger.error("BackendCoordinator not available for initial config.")
            self._chat_backend_configured_successfully = False
            self.error_occurred.emit("Internal Error: Backend Coordinator missing.", True)
            return

        self._current_chat_model_name = loaded_chat_model or constants.DEFAULT_GEMINI_CHAT_MODEL
        self._current_chat_personality_prompt = loaded_chat_personality
        gemini_api_key = get_api_key()

        if gemini_api_key:
            logger.info(f"Configuring {DEFAULT_CHAT_BACKEND_ID} with model {self._current_chat_model_name}")
            self._backend_coordinator.configure_backend(
                DEFAULT_CHAT_BACKEND_ID, gemini_api_key, self._current_chat_model_name,
                self._current_chat_personality_prompt
            )
            logger.info(f"Configuring {PLANNER_BACKEND_ID} with model {constants.DEFAULT_GEMINI_PLANNER_MODEL}")
            self._backend_coordinator.configure_backend(
                PLANNER_BACKEND_ID, gemini_api_key, constants.DEFAULT_GEMINI_PLANNER_MODEL,
                "You are an expert software development planner and a helpful summarizer."
            )
        else:
            logger.warning(
                f"Gemini API Key not found. Default chat ('{DEFAULT_CHAT_BACKEND_ID}') and Planner ('{PLANNER_BACKEND_ID}') backends will not be configured.")
            if self._backend_coordinator:
                self._backend_coordinator.configuration_changed.emit(DEFAULT_CHAT_BACKEND_ID,
                                                                     self._current_chat_model_name, False, [])
                self._backend_coordinator.configuration_changed.emit(PLANNER_BACKEND_ID,
                                                                     constants.DEFAULT_GEMINI_PLANNER_MODEL, False, [])

        if self._backend_coordinator:
            logger.info(f"Configuring {GENERATOR_BACKEND_ID} with model {constants.DEFAULT_OLLAMA_MODEL}")
            self._backend_coordinator.configure_backend(
                GENERATOR_BACKEND_ID, None, constants.DEFAULT_OLLAMA_MODEL,  # type: ignore
                "You are a helpful code generation assistant."
            )
            logger.info(f"Configuring {OLLAMA_CHAT_BACKEND_ID} with model {constants.DEFAULT_OLLAMA_MODEL}")
            self._backend_coordinator.configure_backend(
                OLLAMA_CHAT_BACKEND_ID, None, constants.DEFAULT_OLLAMA_MODEL, None  # type: ignore
            )

    @pyqtSlot(dict)
    def _handle_pcm_project_list_updated(self, projects_dict: Dict[str, str]):
        self.project_inventory_updated.emit(projects_dict)
        if self._project_context_manager:
            current_active_id_in_pcm = self._project_context_manager.get_active_project_id()
            if current_active_id_in_pcm not in projects_dict and current_active_id_in_pcm != constants.GLOBAL_COLLECTION_ID:
                logger.info(
                    f"Active project '{current_active_id_in_pcm}' no longer in inventory. Setting active to Global.")
                self.set_current_project(constants.GLOBAL_COLLECTION_ID)
            elif not projects_dict and current_active_id_in_pcm != constants.GLOBAL_COLLECTION_ID:
                logger.info("Project inventory is empty. Setting active to Global.")
                self.set_current_project(constants.GLOBAL_COLLECTION_ID)

    @pyqtSlot(str)
    def _handle_pcm_active_project_changed(self, new_active_project_id: str):
        logger.info(f"CM: PCM active project changed to: {new_active_project_id}")
        active_history = self.get_project_history(new_active_project_id)
        self.history_changed.emit(active_history[:])  # type: ignore
        self.current_project_changed.emit(new_active_project_id)
        self._update_rag_initialized_state(emit_status=True, project_id=new_active_project_id)
        self._trigger_save_last_session_state()

    @pyqtSlot(str, str, dict, str)
    def _handle_sfm_session_loaded(self, model_name: str, personality: Optional[str],
                                   project_context_data: Dict[str, Any], active_project_id_from_session: str):
        logger.info(
            f"CM: Handling session_loaded from SFM. Model: {model_name}, ActivePID: {active_project_id_from_session}")
        if not (self._project_context_manager and self._backend_coordinator):
            logger.error("CM: Cannot handle session load, PCM or BC missing.")
            self.error_occurred.emit("Critical error loading session state.", True)
            return
        self._project_context_manager.load_state(project_context_data)
        self._current_chat_model_name = model_name or constants.DEFAULT_GEMINI_CHAT_MODEL
        self._current_chat_personality_prompt = personality
        self._configure_initial_backends(self._current_chat_model_name, self._current_chat_personality_prompt)
        self.current_project_changed.emit(active_project_id_from_session)
        self._update_rag_initialized_state(emit_status=True, project_id=active_project_id_from_session)
        self.update_status_based_on_state()

    @pyqtSlot()
    def _handle_sfm_active_history_cleared(self):
        logger.info("CM: Handling active_history_cleared from SFM.")
        if self._project_context_manager:
            active_project_id = self._project_context_manager.get_active_project_id()
            if active_project_id:
                history_for_active_project = self._project_context_manager.get_project_history(active_project_id)
                if history_for_active_project is not None:
                    history_for_active_project.clear()
                self.history_changed.emit([])  # type: ignore
            else:
                logger.warning("CM: SFM cleared history but no active project ID in PCM.")

    @pyqtSlot(str, str, dict)
    def _handle_sfm_request_state_save(self, model_name: str, personality: Optional[str],
                                       all_project_data: Dict[str, Any]):
        logger.info("CM: Handling request_state_save from SFM.")
        if self._session_flow_manager:
            self._session_flow_manager.save_current_session_to_last_state(
                current_chat_model=model_name, current_chat_personality=personality
            )

    @pyqtSlot(str, str)
    def _handle_backend_chunk_received(self, backend_id: str, chunk: str):
        is_mod_coord_active_and_waiting = self._modification_coordinator and self._modification_coordinator.is_awaiting_llm_response()
        if backend_id == DEFAULT_CHAT_BACKEND_ID and not is_mod_coord_active_and_waiting:
            self.stream_chunk_received.emit(chunk)
        elif is_mod_coord_active_and_waiting:
            logger.debug(f"CM: Suppressing stream chunk from '{backend_id}' during modification response.")
        else:
            logger.debug(
                f"CM: Chunk from backend '{backend_id}', not emitting to main chat UI as it's not default chat or MC is not waiting.")

    # --- _handle_backend_response_completed - MODIFIED ---
    @pyqtSlot(str, ChatMessage, dict)  # backend_id, completed_message, usage_stats_with_metadata
    def _handle_backend_response_completed(self, backend_id: str, completed_message: ChatMessage,
                                           usage_stats_with_metadata: dict):
        logger.info(
            f"CM: Received completed response from BC for backend '{backend_id}'. Stats/Metadata: {usage_stats_with_metadata}")
        purpose = usage_stats_with_metadata.get("purpose")
        original_target_filename = usage_stats_with_metadata.get("original_target_filename")

        if purpose == "code_summary" and original_target_filename:
            logger.info(f"CM: Handling completed response as a CODE SUMMARY for '{original_target_filename}'.")
            self.status_update.emit(f"AvA's summary for '{original_target_filename}' is ready!", "#98c379", True, 3000)
            summary_msg_text = f"✨ **AvA's Summary for {original_target_filename}:** ✨\n\n{completed_message.text}"
            summary_chat_message = ChatMessage(
                role=SYSTEM_ROLE,
                parts=[summary_msg_text],
                metadata={"is_ava_summary": True, "target_file": original_target_filename, "is_internal": False}
            )
            if self._project_context_manager:
                self._project_context_manager.add_message_to_active_project(summary_chat_message)
                self.new_message_added.emit(summary_chat_message)
                self._trigger_save_last_session_state()
            return

        if self._modification_coordinator and self._modification_coordinator.is_awaiting_llm_response():
            if backend_id == PLANNER_BACKEND_ID or backend_id == GENERATOR_BACKEND_ID:
                logger.info(f"CM: Routing completed LLM response from '{backend_id}' to MC.")
                self._modification_coordinator.process_llm_response(backend_id, completed_message)
                return
            else:
                logger.warning(
                    f"CM: MC is awaiting LLM, but response from unexpected backend '{backend_id}'. Ignoring for MC.")

        if backend_id == DEFAULT_CHAT_BACKEND_ID:
            logger.info(f"CM: Handling response from '{backend_id}' as normal chat completion.")
            self.stream_finished.emit()
            if self._project_context_manager:
                self._project_context_manager.add_message_to_active_project(completed_message)
                self.new_message_added.emit(completed_message)
                self._trigger_save_last_session_state()

            prompt_tokens = usage_stats_with_metadata.get("prompt_tokens")
            completion_tokens = usage_stats_with_metadata.get("completion_tokens")
            if prompt_tokens is not None and completion_tokens is not None:
                if not self._model_info_service:
                    logger.error("ModelInfoService not available to get max tokens.")
                    model_max_context = 0
                else:
                    model_max_context = self._model_info_service.get_max_tokens(self._current_chat_model_name)
                self.token_usage_updated.emit(backend_id, prompt_tokens, completion_tokens, model_max_context)
            return

        logger.warning(
            f"CM: Received unhandled completed response from backend '{backend_id}'. Message: {completed_message.text[:50]}...")
        if self._project_context_manager and self._backend_coordinator:
            current_model_for_backend = self._backend_coordinator.get_current_configured_model(backend_id)
            sys_msg_text = f"[Internal: Unhandled completion from '{backend_id}' for model '{current_model_for_backend or 'N/A'}']"
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[sys_msg_text], metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(sys_msg)
            self.new_message_added.emit(sys_msg)

    @pyqtSlot(str, str)
    def _handle_backend_response_error(self, backend_id: str, error_message_str: str):
        logger.error(f"CM: Received error from BC for backend '{backend_id}': {error_message_str}")
        if self._modification_coordinator and self._modification_coordinator.is_active():
            current_mc_phase = self._modification_coordinator._current_phase if ModPhase else "UNKNOWN"
            is_awaiting_planner = (
                                              current_mc_phase == ModPhase.AWAITING_PLAN or current_mc_phase == ModPhase.AWAITING_GEMINI_REFINEMENT) and backend_id == PLANNER_BACKEND_ID
            is_awaiting_generator = current_mc_phase == ModPhase.AWAITING_CODE_GENERATION and backend_id == GENERATOR_BACKEND_ID
            if is_awaiting_planner or is_awaiting_generator:
                logger.info(f"CM: Routing backend error from '{backend_id}' to MC for phase '{current_mc_phase}'.")
                self._modification_coordinator.process_llm_error(backend_id, error_message_str)
                return

        if backend_id == PLANNER_BACKEND_ID:
            logger.error(f"CM: Planner AI ('{PLANNER_BACKEND_ID}') reported an error (summary?): {error_message_str}")
            summary_error_display_msg = f"AvA (Planner) had an issue: {error_message_str}"
            self.error_occurred.emit(summary_error_display_msg, False)
            if self._project_context_manager:
                err_msg_text = f"[System: Error during an operation with AvA (Planner): {error_message_str}]"
                err_msg = ChatMessage(role=ERROR_ROLE, parts=[err_msg_text], metadata={"is_internal": False})
                self._project_context_manager.add_message_to_active_project(err_msg)
                self.new_message_added.emit(err_msg)
            return

        if backend_id == DEFAULT_CHAT_BACKEND_ID:
            logger.info(f"CM: Handling backend error from '{backend_id}' as normal chat error.")
            self.stream_finished.emit()
            if self._project_context_manager:
                err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Backend '{backend_id}' Error: {error_message_str}"])
                self._project_context_manager.add_message_to_active_project(err_obj)
                self.new_message_added.emit(err_obj)
                self.error_occurred.emit(f"Backend '{backend_id}' Error: {error_message_str}", False)
                self._trigger_save_last_session_state()
            return

        logger.warning(f"CM: Received unhandled error from backend '{backend_id}'.")
        self.error_occurred.emit(f"Error from backend '{backend_id}': {error_message_str}", False)
        if self._project_context_manager:
            sys_err_text = f"[Internal: Unhandled error from '{backend_id}': {error_message_str}]"
            sys_msg = ChatMessage(role=ERROR_ROLE, parts=[sys_err_text], metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(sys_msg)
            self.new_message_added.emit(sys_msg)

    @pyqtSlot(bool)
    def _handle_backend_busy_changed(self, backend_is_busy: bool):
        logger.debug(f"CM: BC overall busy state changed to: {backend_is_busy}")
        self._update_overall_busy_state()

    @pyqtSlot(str, str, bool, list)
    def _handle_backend_configuration_changed(self, backend_id: str, model_name: str, is_configured: bool,
                                              available_models: list):
        logger.info(
            f"CM: BC config changed for '{backend_id}'. Model: {model_name}, ConfigOK: {is_configured}, Avail: {len(available_models)}")
        if backend_id == DEFAULT_CHAT_BACKEND_ID:
            self._current_chat_model_name = model_name
            self._chat_backend_configured_successfully = is_configured
            self._available_chat_models = available_models[:]  # type: ignore
            if not is_configured and self._backend_coordinator:
                err = self._backend_coordinator.get_last_error_for_backend(
                    DEFAULT_CHAT_BACKEND_ID) or "Chat API configuration error."
                self.error_occurred.emit(f"Chat API Config Error: {err}", False)
            self.available_models_changed.emit(self._available_chat_models[:])
            self.config_state_changed.emit(self._current_chat_model_name,
                                           self._chat_backend_configured_successfully and bool(
                                               self._current_chat_personality_prompt))
            self.update_status_based_on_state()
            self._trigger_save_last_session_state()
        elif backend_id in [PLANNER_BACKEND_ID, GENERATOR_BACKEND_ID, OLLAMA_CHAT_BACKEND_ID]:
            name_map = {PLANNER_BACKEND_ID: "Planner", GENERATOR_BACKEND_ID: "Generator",
                        OLLAMA_CHAT_BACKEND_ID: "Ollama Chat"}
            d_name = name_map.get(backend_id, backend_id)
            if not is_configured and self._backend_coordinator:
                err = self._backend_coordinator.get_last_error_for_backend(backend_id) or f"{d_name} config error."
                self.error_occurred.emit(f"{d_name} ('{backend_id}') Config Error: {err}", False)
                self.status_update.emit(f"{d_name} ('{backend_id}') not configured.", "#e06c75", True, 5000)
            elif is_configured:
                self.status_update.emit(f"{d_name} ('{backend_id}') OK with {model_name}.", "#61afef", True, 3000)

    @pyqtSlot(bool, str)
    def _handle_upload_started(self, is_global: bool, item_description: str):
        active_project_name_str = "N/A"
        if self._project_context_manager:
            active_project_name_str = self._project_context_manager.get_active_project_name() or "Current"
        context_name = constants.GLOBAL_CONTEXT_DISPLAY_NAME if is_global else active_project_name_str
        self.status_update.emit(f"Uploading {item_description} to '{context_name}' context...", "#61afef", False, 0)
        self._update_overall_busy_state()

    @pyqtSlot(ChatMessage)
    def _handle_upload_summary(self, summary_message: ChatMessage):
        if not self._project_context_manager: return
        self._project_context_manager.add_message_to_active_project(summary_message)
        self.new_message_added.emit(summary_message)
        s_cid = summary_message.metadata.get("collection_id") if summary_message.metadata else None
        self._update_rag_initialized_state(emit_status=True, project_id=s_cid)
        self.update_status_based_on_state()
        self._trigger_save_last_session_state()

    @pyqtSlot(str)
    def _handle_upload_error(self, error_message_str: str):
        if not self._project_context_manager: return
        err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Upload System Error: {error_message_str}"])
        self._project_context_manager.add_message_to_active_project(err_obj)
        self.new_message_added.emit(err_obj)
        self.error_occurred.emit(f"Upload Error: {error_message_str}", False)
        self.update_status_based_on_state()

    @pyqtSlot(bool)
    def _handle_upload_busy_changed(self, upload_is_busy: bool):
        self._update_overall_busy_state()

    # --- _handle_code_generated_and_summary_needed - MODIFIED ---
    @pyqtSlot(str, str, str)  # generated_code, coder_instructions, target_filename
    def _handle_code_generated_and_summary_needed(self, generated_code: str, coder_instructions: str,
                                                  target_filename: str):
        logger.info(f"CM: Summary needed for '{target_filename}'. Delegating to CodeSummaryService.")
        if not self._code_summary_service:
            logger.error(f"Cannot request summary for '{target_filename}': CodeSummaryService not initialized.")
            self.error_occurred.emit(f"Internal error: Summary service unavailable for '{target_filename}'.", False)
            return
        if not self._backend_coordinator:
            logger.error(f"Cannot request summary for '{target_filename}': BackendCoordinator not available.")
            self.error_occurred.emit(f"Internal error: Backend system unavailable for summary of '{target_filename}'.",
                                     True)
            return
        self.status_update.emit(f"AvA is preparing to summarize changes for '{target_filename}'...", "#e5c07b", True,
                                4000)
        success = self._code_summary_service.request_code_summary(
            backend_coordinator=self._backend_coordinator,
            target_filename=target_filename,
            coder_instructions=coder_instructions,
            generated_code=generated_code
        )
        if not success:
            err_msg = f"Failed to dispatch summary request for '{target_filename}'. Check logs."
            logger.error(err_msg)
            if self._project_context_manager:
                sys_err_msg_text = f"[System: Could not initiate summary for '{target_filename}'. Possible configuration issue or internal error.]"
                sys_err_msg = ChatMessage(role=ERROR_ROLE, parts=[sys_err_msg_text], metadata={"is_internal": False})
                self._project_context_manager.add_message_to_active_project(sys_err_msg)
                self.new_message_added.emit(sys_err_msg)
            self.update_status_based_on_state()

    @pyqtSlot(list)
    def _handle_uih_normal_chat_request(self, history_containing_new_user_message: List[ChatMessage]):
        logger.info("CM: Handling normal_chat_request_ready from UIH.")
        if not self._backend_coordinator:
            self.error_occurred.emit("Cannot send chat: BC missing.", True)
            return

        if not self._project_context_manager:
            self.error_occurred.emit("Cannot process user chat: Project Manager missing.", True)
            return

        if not history_containing_new_user_message:
            logger.warning("CM: _handle_uih_normal_chat_request received empty history. Ignoring.")
            return

        new_user_message = history_containing_new_user_message[0]

        if new_user_message.role == USER_ROLE:
            self._project_context_manager.add_message_to_active_project(new_user_message)
            self.new_message_added.emit(new_user_message)
            self._trigger_save_last_session_state()
            logger.debug(f"CM: Added user message to history and emitted for display: {new_user_message.text[:50]}...")
        else:
            logger.warning(
                f"CM: Expected USER_ROLE message in _handle_uih_normal_chat_request, got {new_user_message.role}. Not displaying separately.")

        full_history_for_backend = self._project_context_manager.get_active_conversation_history()
        if not full_history_for_backend:
            logger.error("CM: History for backend is empty after adding user message. This is unexpected.")
            self.error_occurred.emit("Internal error preparing chat.", True)
            return

        self.stream_started.emit(MODEL_ROLE)
        request_options = {"temperature": self._current_chat_temperature}

        logger.debug(f"CM: Sending history of length {len(full_history_for_backend)} to backend.")
        self._backend_coordinator.request_response_stream(
            DEFAULT_CHAT_BACKEND_ID,
            full_history_for_backend,
            is_modification_response_expected=False,
            options=request_options
        )

    @pyqtSlot(str, str, str)
    def _handle_uih_mod_start_request(self, query: str, context: str, focus_prefix: str):
        logger.info("CM: Handling modification_sequence_start_requested from UIH.")
        if self._modification_coordinator:
            if self._modification_handler_instance:
                self._modification_handler_instance.activate_sequence()
            self._modification_coordinator.start_sequence(query, context, focus_prefix)
        else:
            self.error_occurred.emit("Modification feature unavailable.", False)

    @pyqtSlot(str, str)
    def _handle_uih_mod_user_input(self, user_command: str, action_type: str):
        logger.info(f"CM: Handling mod_user_input ('{user_command}', Type: '{action_type}') from UIH.")
        if self._modification_coordinator:
            self._modification_coordinator.process_user_input(user_command)
        else:
            self.error_occurred.emit("Modification feature unavailable.", False)

    @pyqtSlot(str)
    def _handle_uih_processing_error(self, error_message: str):
        logger.error(f"CM: UIH processing error: {error_message}")
        self.error_occurred.emit(f"Input Processing Error: {error_message}", False)
        if self._project_context_manager:
            err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Input Error: {error_message}"])
            self._project_context_manager.add_message_to_active_project(err_obj)
            self.new_message_added.emit(err_obj)

    @pyqtSlot(str, list)
    def _handle_mc_request_llm_call(self, target_backend_id: str, history_to_send: List[ChatMessage]):
        if self._backend_coordinator:
            mc_options = {"temperature": 0.5}
            if target_backend_id == GENERATOR_BACKEND_ID:
                mc_options = {"temperature": 0.2}
            self._backend_coordinator.request_response_stream(target_backend_id, history_to_send, True,
                                                              options=mc_options)
        elif self._modification_coordinator:
            self._modification_coordinator.process_llm_error(target_backend_id, "BC unavailable.")

    @pyqtSlot(str, str)
    def _handle_mc_file_ready(self, filename: str, content: str):
        self.code_file_updated.emit(filename, content)
        if self._project_context_manager:
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[f"[System: File '{filename}' updated. See Code Viewer.]"],
                                  metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(sys_msg)
            self.new_message_added.emit(sys_msg)

    @pyqtSlot(str)
    def _handle_mc_sequence_complete(self, reason: str):
        if self._project_context_manager:
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[f"[System: Code modification sequence ended ({reason}).]"],
                                  metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(sys_msg)
            self.new_message_added.emit(sys_msg)
            self._trigger_save_last_session_state()
        self.update_status_based_on_state()
        if self._modification_handler_instance:
            self._modification_handler_instance.cancel_modification()

    @pyqtSlot(str)
    def _handle_mc_error(self, error_message: str):
        if self._project_context_manager:
            err_msg_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Modification System Error: {error_message}"],
                                      metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(err_msg_obj)
            self.new_message_added.emit(err_msg_obj)
            self._trigger_save_last_session_state()
        self.error_occurred.emit(f"Modification Error: {error_message}", False)
        self.update_status_based_on_state()

    @pyqtSlot(str)
    def _handle_mc_status_update(self, message: str):
        if self._project_context_manager:
            status_msg = ChatMessage(role=SYSTEM_ROLE, parts=[message], metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(status_msg)
            self.new_message_added.emit(status_msg)

    def _cancel_active_tasks(self):
        if self._backend_coordinator: self._backend_coordinator.cancel_current_task()
        if self._upload_coordinator: self._upload_coordinator.cancel_current_upload()
        if self._modification_coordinator and self._modification_coordinator.is_active():
            self._modification_coordinator.cancel_sequence(reason="user cancel")

    def cleanup(self):
        self._cancel_active_tasks()
        self._trigger_save_last_session_state()

    def _update_rag_initialized_state(self, emit_status: bool = True, project_id: Optional[str] = None):
        if not self._project_context_manager: return
        target_pid = project_id or (self._project_context_manager.get_active_project_id())
        new_init_state = self.is_rag_context_initialized(target_pid)
        active_pid = self._project_context_manager.get_active_project_id()
        if target_pid == active_pid:
            if self._rag_initialized != new_init_state: self._rag_initialized = new_init_state
            if emit_status or (self._rag_initialized != new_init_state): self.update_status_based_on_state()
        elif emit_status:
            self.update_status_based_on_state()

    def is_rag_context_initialized(self, project_id: Optional[str]) -> bool:
        if not (self._vector_db_service and project_id):
            self._rag_available = False;
            return False
        self._rag_available = True
        is_vdb_ready = self._vector_db_service.is_ready(project_id)  # type: ignore
        size = self._vector_db_service.get_collection_size(project_id) if is_vdb_ready else 0  # type: ignore
        return is_vdb_ready and size > 0

    def get_project_history(self, project_id: str) -> List[ChatMessage]:
        return list(self._project_context_manager.get_project_history(
            project_id) or []) if self._project_context_manager else []

    def get_current_history(self) -> List[ChatMessage]:
        return list(
            self._project_context_manager.get_active_conversation_history() or []) if self._project_context_manager else []

    def get_current_model(self) -> str:
        return self._current_chat_model_name

    def get_current_personality(self) -> Optional[str]:
        return self._current_chat_personality_prompt

    def get_current_project_id(self) -> Optional[str]:
        return self._project_context_manager.get_active_project_id() if self._project_context_manager else None

    def is_api_ready(self) -> bool:
        return self._chat_backend_configured_successfully

    def is_overall_busy(self) -> bool:
        return self._overall_busy

    def is_rag_available(self) -> bool:
        return self._rag_available

    def get_rag_contents(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not (self._project_context_manager and self._vector_db_service): return []
        target_id = collection_id or (self._project_context_manager.get_active_project_id())
        if not target_id or not self._vector_db_service.is_ready(target_id): return []  # type: ignore
        try:
            return self._vector_db_service.get_all_metadata(target_id)  # type: ignore
        except Exception as e:
            logger.exception(f"Error RAG contents for '{target_id}': {e}"); return []

    def get_current_focus_paths(self) -> Optional[List[str]]:
        return self._current_chat_focus_paths

    def get_project_context_manager(self) -> Optional[ProjectContextManager]:
        return self._project_context_manager

    def get_backend_coordinator(self) -> Optional[BackendCoordinator]:
        return self._backend_coordinator

    def get_upload_coordinator(self) -> Optional[UploadCoordinator]:
        return self._upload_coordinator

    def get_modification_coordinator(self) -> Optional[ModificationCoordinator]:
        return self._modification_coordinator

    def get_session_flow_manager(self) -> Optional[SessionFlowManager]:
        return self._session_flow_manager

    def _trigger_save_last_session_state(self):
        if self._session_flow_manager:
            logger.debug("CM: Triggering save of last session state via SFM.")
            self._session_flow_manager.save_current_session_to_last_state(
                current_chat_model=self._current_chat_model_name,
                current_chat_personality=self._current_chat_personality_prompt
            )
        else:
            logger.error("CM: Cannot trigger save last session state, SFM is missing.")

    def set_model(self, model_name: str):
        logger.info(f"CM: Setting new model to: {model_name}")
        self._current_chat_model_name = model_name
        gemini_api_key = get_api_key()
        if self._backend_coordinator:
            if gemini_api_key:
                self._backend_coordinator.configure_backend(
                    DEFAULT_CHAT_BACKEND_ID, gemini_api_key, model_name, self._current_chat_personality_prompt
                )
            else:
                logger.warning(
                    f"Cannot reconfigure default chat backend for model {model_name}: Gemini API key missing.")
                self._backend_coordinator.configuration_changed.emit(DEFAULT_CHAT_BACKEND_ID, model_name, False, [])

    def set_personality(self, prompt: Optional[str]):
        logger.info(f"CM: Setting new personality (System Prompt). Present: {bool(prompt)}")
        self._current_chat_personality_prompt = prompt.strip() if prompt else None
        gemini_api_key = get_api_key()
        if self._backend_coordinator:
            if gemini_api_key:
                self._backend_coordinator.configure_backend(
                    DEFAULT_CHAT_BACKEND_ID, gemini_api_key, self._current_chat_model_name,
                    self._current_chat_personality_prompt
                )
            else:
                logger.warning(f"Cannot reconfigure default chat backend for new personality: Gemini API key missing.")
                self._backend_coordinator.configuration_changed.emit(DEFAULT_CHAT_BACKEND_ID,
                                                                     self._current_chat_model_name, False, [])

    def set_current_project(self, project_id: str):
        logger.info(f"CM: Setting current project to ID: {project_id}")
        if self._project_context_manager:
            if not self._project_context_manager.set_active_project(project_id):
                self.error_occurred.emit(f"Failed to set project '{project_id}'.", False)
        else:
            self.error_occurred.emit("Project manager not available.", True)

    def create_project_collection(self, project_name: str):
        logger.info(f"CM: Request to create project context: '{project_name}'")
        if self._project_context_manager:
            new_id = self._project_context_manager.create_project(project_name)
            if not new_id:
                self.error_occurred.emit(f"Failed to create project '{project_name}'. Name might exist.", False)
            else:
                self.status_update.emit(f"Project '{project_name}' created.", "#98c379", True, 3000)
        else:
            self.error_occurred.emit("Project manager not available.", True)

    def start_new_chat(self):
        logger.info("CM: Starting new chat session.")
        if self._session_flow_manager:
            self._session_flow_manager.start_new_chat_session(
                self._current_chat_model_name, self._current_chat_personality_prompt
            )
        else:
            self.error_occurred.emit("Session manager not available for new chat.", True)

    def load_chat_session(self, filepath: str):
        if self._session_flow_manager:
            self._session_flow_manager.load_named_session(filepath, DEFAULT_CHAT_BACKEND_ID)
        else:
            self.error_occurred.emit("Session manager not available for loading session.", True)

    def save_current_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager:
            return self._session_flow_manager.save_session_as(
                filepath, self._current_chat_model_name, self._current_chat_personality_prompt
            )
        self.error_occurred.emit("Session manager not available for saving session.", True);
        return False

    def delete_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager: return self._session_flow_manager.delete_named_session(filepath)
        self.error_occurred.emit("Session manager not available for deleting session.", True);
        return False

    def list_saved_sessions(self) -> List[str]:
        if self._session_flow_manager: return self._session_flow_manager.list_saved_sessions()
        self.error_occurred.emit("Session manager not available to list sessions.", False);
        return []

    def process_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        logger.info("CM: Processing user message...")
        if self._user_input_handler:
            self._user_input_handler.handle_user_message(
                text=text, image_data=image_data, focus_paths=self._current_chat_focus_paths,
                rag_available=self._rag_available,
                rag_initialized_for_project=self.is_rag_context_initialized(self.get_current_project_id())
            )
        else:
            self.error_occurred.emit("Input handler not available.", True)

    def update_status_based_on_state(self):
        if not self._chat_backend_configured_successfully:
            last_err = self._backend_coordinator.get_last_error_for_backend(
                DEFAULT_CHAT_BACKEND_ID) if self._backend_coordinator else "Chat configuration error"
            self.status_update.emit(f"API Not Configured: {last_err or 'Check settings.'}", "#e06c75", False, 0)
        elif self._overall_busy:
            self.status_update.emit("Processing...", "#e5c07b", False, 0)
        else:
            status_parts = ["Ready"]
            if self._project_context_manager:
                active_project_id = self._project_context_manager.get_active_project_id()
                active_project_name = self._project_context_manager.get_project_name(
                    active_project_id) or "Unknown Project"
                if active_project_id == constants.GLOBAL_COLLECTION_ID: active_project_name = constants.GLOBAL_CONTEXT_DISPLAY_NAME
                status_parts.append(f"(Ctx: {active_project_name})")
            if self.is_rag_context_initialized(
                    self._project_context_manager.get_active_project_id() if self._project_context_manager else None):
                status_parts.append("[RAG Active]")
            self.status_update.emit(" ".join(status_parts), "#98c379", False, 0)

    def set_chat_temperature(self, temperature: float):
        if 0.0 <= temperature <= 2.0:
            self._current_chat_temperature = temperature
            logger.info(f"CM: Chat temperature set to {self._current_chat_temperature:.2f}")
            self.status_update.emit(f"Temperature set to {self._current_chat_temperature:.2f}", "#61afef", True, 3000)
            self._trigger_save_last_session_state()
        else:
            logger.warning(f"CM: Invalid temperature value: {temperature}")

    def handle_file_upload(self, file_paths: List[str]):
        if self._upload_coordinator:
            self._upload_coordinator.upload_files_to_current_project(file_paths)
        else:
            self.error_occurred.emit("Upload service not available.", False)

    def handle_directory_upload(self, dir_path: str):
        if self._upload_coordinator:
            self._upload_coordinator.upload_directory_to_current_project(dir_path)
        else:
            self.error_occurred.emit("Upload service not available.", False)

    def handle_global_file_upload(self, file_paths: List[str]):
        if self._upload_coordinator:
            self._upload_coordinator.upload_files_to_global(file_paths)
        else:
            self.error_occurred.emit("Upload service not available.", False)

    def handle_global_directory_upload(self, dir_path: str):
        if self._upload_coordinator:
            self._upload_coordinator.upload_directory_to_global(dir_path)
        else:
            self.error_occurred.emit("Upload service not available.", False)

    def set_chat_focus(self, paths: List[str]):
        if not isinstance(paths, list): return
        self._current_chat_focus_paths = paths
        focus_display = ", ".join([os.path.basename(p) for p in paths])
        if len(focus_display) > 50: focus_display = focus_display[:47] + "..."
        self.status_update.emit(f"Focus set on: {focus_display}", "#61afef", True, 4000)

    def _update_overall_busy_state(self):
        backend_busy = self._backend_coordinator.is_processing_request() if self._backend_coordinator else False
        upload_busy = self._upload_coordinator.is_busy() if self._upload_coordinator else False
        new_overall_busy = backend_busy or upload_busy
        if self._overall_busy != new_overall_busy:
            self._overall_busy = new_overall_busy
            self.busy_state_changed.emit(self._overall_busy)
            self.update_status_based_on_state()