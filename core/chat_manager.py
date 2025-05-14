# core/chat_manager.py (Part 1: Imports, Init, Basic Setup)
# UPDATED: Connects to the modified modification_sequence_start_requested signal.

import logging
import asyncio
import os
import uuid
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtWidgets import QApplication

try:
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
except ImportError:
    class ChatMessage: pass
    from enum import Enum, auto
    class MessageLoadingState(Enum): IDLE=auto(); LOADING=auto(); COMPLETED=auto(); ERROR=auto()
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"
    logging.error("ChatManager: Failed to import ChatMessage or MessageLoadingState from core.")

from backend.interface import BackendInterface
from services.code_summary_service import CodeSummaryService
from services.model_info_service import ModelInfoService
from services.session_service import SessionService
from services.upload_service import UploadService, VECTOR_DB_SERVICE_AVAILABLE
from services.vector_db_service import VectorDBService
from .project_context_manager import ProjectContextManager
from .backend_coordinator import BackendCoordinator
from .session_flow_manager import SessionFlowManager
from .upload_coordinator import UploadCoordinator
from .rag_handler import RagHandler
from .user_input_handler import UserInputHandler # Corrected: UserInputHandler for signal connection
from .user_input_processor import UserInputProcessor
from .application_orchestrator import ApplicationOrchestrator

try:
    from .modification_handler import ModificationHandler
    MOD_HANDLER_AVAILABLE = True
except ImportError:
    ModificationHandler = None
    MOD_HANDLER_AVAILABLE = False
try:
    from .modification_coordinator import ModificationCoordinator, ModPhase
    MOD_COORDINATOR_AVAILABLE = True
except ImportError:
    ModificationCoordinator = None
    ModPhase = None
    MOD_COORDINATOR_AVAILABLE = False

from utils import constants
try:
    from config import get_api_key
except ImportError:
    def get_api_key(): return None
    logging.info("config.py not found or get_api_key not defined, using dummy for API key.")

logger = logging.getLogger(__name__)

DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
OLLAMA_CHAT_BACKEND_ID = "ollama_chat"
PLANNER_BACKEND_ID = "gemini_planner"
GENERATOR_BACKEND_ID = "ollama_generator"

class ChatManager(QObject):
    history_changed = pyqtSignal(list)
    new_message_added = pyqtSignal(object)
    status_update = pyqtSignal(str, str, bool, int)
    error_occurred = pyqtSignal(str, bool)
    busy_state_changed = pyqtSignal(bool)
    config_state_changed = pyqtSignal(str, bool)
    stream_started = pyqtSignal(str)
    stream_chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()
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
        self._rag_handler = self._orchestrator.get_rag_handler()
        self._modification_handler_instance = self._orchestrator.get_modification_handler_instance()
        if self._modification_handler_instance and isinstance(self._modification_handler_instance, QObject):
            if self._modification_handler_instance.parent() is None:
                self._modification_handler_instance.setParent(self)

        self._session_service: Optional[SessionService] = getattr(orchestrator, '_session_service', None)
        self._vector_db_service: Optional[VectorDBService] = getattr(orchestrator, '_vector_db_service', None)
        self._code_summary_service = CodeSummaryService()
        self._model_info_service = ModelInfoService()
        self._connect_component_signals()
        self._initialize_state_variables()
        logger.info("ChatManager core initialization using orchestrator complete.")

    def _connect_component_signals(self):
        logger.debug("ChatManager connecting component signals...")
        if self._project_context_manager:
            self._project_context_manager.project_list_updated.connect(self._handle_pcm_project_list_updated)
            self._project_context_manager.active_project_changed.connect(self._handle_pcm_active_project_changed)
        if self._backend_coordinator:
            self._backend_coordinator.stream_started.connect(self._handle_backend_stream_started)
            self._backend_coordinator.stream_chunk_received.connect(self._handle_backend_chunk_received)
            self._backend_coordinator.response_completed.connect(self._handle_backend_response_completed)
            self._backend_coordinator.response_error.connect(self._handle_backend_response_error)
            self._backend_coordinator.busy_state_changed.connect(self._handle_backend_busy_changed)
            self._backend_coordinator.configuration_changed.connect(self._handle_backend_configuration_changed)
        if self._upload_coordinator:
            self._upload_coordinator.upload_started.connect(self._handle_upload_started)
            self._upload_coordinator.upload_summary_received.connect(self._handle_upload_summary)
            self._upload_coordinator.upload_error.connect(self._handle_upload_error)
            self._upload_coordinator.busy_state_changed.connect(self._handle_upload_busy_changed)
        if self._session_flow_manager:
            self._session_flow_manager.session_loaded.connect(self._handle_sfm_session_loaded)
            self._session_flow_manager.active_history_cleared.connect(self._handle_sfm_active_history_cleared)
            self._session_flow_manager.status_update_requested.connect(self.status_update)
            self._session_flow_manager.error_occurred.connect(self.error_occurred)
            self._session_flow_manager.request_state_save.connect(self._handle_sfm_request_state_save)

        if self._user_input_handler: # Ensure UserInputHandler instance exists
            self._user_input_handler.normal_chat_request_ready.connect(self._handle_uih_normal_chat_request)
            # --- MODIFICATION START: Connect to the updated signal ---
            self._user_input_handler.modification_sequence_start_requested.connect(self._handle_uih_mod_start_request)
            # --- MODIFICATION END ---
            self._user_input_handler.modification_user_input_received.connect(self._handle_uih_mod_user_input)
            self._user_input_handler.processing_error_occurred.connect(self._handle_uih_processing_error)

        if self._modification_coordinator:
            self._modification_coordinator.request_llm_call.connect(self._handle_mc_request_llm_call)
            self._modification_coordinator.file_ready_for_display.connect(self._handle_mc_file_ready)
            self._modification_coordinator.modification_sequence_complete.connect(self._handle_mc_sequence_complete) # Already connected to the 2-arg version
            self._modification_coordinator.modification_error.connect(self._handle_mc_error)
            self._modification_coordinator.status_update.connect(self._handle_mc_status_update)
            self._modification_coordinator.codeGeneratedAndSummaryNeeded.connect(self._handle_code_generated_and_summary_needed)
        logger.debug("ChatManager component signal connection process finished.")

    def _initialize_state_variables(self):
        # This method content is from your combined.txt
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
        # This method content is from your combined.txt
        logger.info("ChatManager late initialization process starting...")
        if not (self._session_flow_manager and self._project_context_manager and self._user_input_handler and self._backend_coordinator):
            missing = [n for c, n in [(self._session_flow_manager, "SFM"), (self._project_context_manager, "PCM"), (self._user_input_handler, "UIH"), (self._backend_coordinator, "BC")] if not c]
            logger.critical(f"Cannot initialize ChatManager: Critical components missing: {missing}")
            self.error_occurred.emit(f"Critical error during init ({', '.join(missing)} missing).", True); return
        m, p, pd, apid = self._session_flow_manager.load_last_session_state_on_startup()
        if pd: self._project_context_manager.load_state(pd)
        else: self._project_context_manager.set_active_project(constants.GLOBAL_COLLECTION_ID)
        self._perform_orphan_cleanup(self._project_context_manager.save_state())
        self._set_initial_active_project(apid, None) # The None is for a legacy second arg.
        self._configure_initial_backends(m, p)
        self.update_status_based_on_state()
        caid = self._project_context_manager.get_active_project_id()
        self._update_rag_initialized_state(emit_status=False, project_id=caid)
        logger.info(f"ChatManager late init complete. Active project: {caid}, Chat Model: {self._current_chat_model_name}")

    def _perform_orphan_cleanup(self, project_context_data_from_pcm: Optional[Dict[str, Any]]):
        # This method content is from your combined.txt
        if not (self._project_context_manager and self._vector_db_service): return
        orphaned_ids = [pid for pid, name in self._project_context_manager.get_all_projects_info().items()
                        if name == constants.GLOBAL_CONTEXT_DISPLAY_NAME and pid != constants.GLOBAL_COLLECTION_ID]
        if orphaned_ids: logger.info(f"Attempting to delete {len(orphaned_ids)} orphaned entries...")
        # Actual deletion logic would be here if implemented

    def _set_initial_active_project(self, target_active_project_id: str, _): # Underscore for unused legacy arg
        # This method content is from your combined.txt
        if not self._project_context_manager: return
        if not self._project_context_manager.get_project_history(target_active_project_id):
            target_active_project_id = constants.GLOBAL_COLLECTION_ID
        self._project_context_manager.set_active_project(target_active_project_id)

    def _configure_initial_backends(self, loaded_chat_model: Optional[str], loaded_chat_personality: Optional[str]):
        # This method content is from your combined.txt
        if not self._backend_coordinator: return
        self._current_chat_model_name = loaded_chat_model or constants.DEFAULT_GEMINI_CHAT_MODEL
        self._current_chat_personality_prompt = loaded_chat_personality
        key = get_api_key()
        if key:
            self._backend_coordinator.configure_backend(DEFAULT_CHAT_BACKEND_ID, key, self._current_chat_model_name, self._current_chat_personality_prompt)
            self._backend_coordinator.configure_backend(PLANNER_BACKEND_ID, key, constants.DEFAULT_GEMINI_PLANNER_MODEL, "Expert planner.")
        self._backend_coordinator.configure_backend(GENERATOR_BACKEND_ID, None, constants.DEFAULT_OLLAMA_MODEL, "Code assistant.")
        self._backend_coordinator.configure_backend(OLLAMA_CHAT_BACKEND_ID, None, constants.DEFAULT_OLLAMA_MODEL, None) # Ollama chat needs no explicit personality here# core/chat_manager.py (Part 2: Slot Handlers, Actions, Getters/Setters)
# UPDATED: _handle_uih_mod_start_request now adds user message to history
#          before starting the modification sequence.

    # Continued from ChatManager Class Definition (Part 1)

    @pyqtSlot(dict)
    def _handle_pcm_project_list_updated(self, projects_dict: Dict[str, str]):
        # This method content is from your combined.txt
        self.project_inventory_updated.emit(projects_dict)
        if self._project_context_manager:
            current_active_id_in_pcm = self._project_context_manager.get_active_project_id()
            if current_active_id_in_pcm not in projects_dict and current_active_id_in_pcm != constants.GLOBAL_COLLECTION_ID:
                self.set_current_project(constants.GLOBAL_COLLECTION_ID)
            elif not projects_dict and current_active_id_in_pcm != constants.GLOBAL_COLLECTION_ID:
                self.set_current_project(constants.GLOBAL_COLLECTION_ID)

    @pyqtSlot(str)
    def _handle_pcm_active_project_changed(self, new_active_project_id: str):
        # This method content is from your combined.txt
        logger.info(f"CM: PCM active project changed to: {new_active_project_id}")
        active_history = self.get_project_history(new_active_project_id)
        self.history_changed.emit(active_history[:])
        self.current_project_changed.emit(new_active_project_id)
        self._update_rag_initialized_state(emit_status=True, project_id=new_active_project_id)
        self._trigger_save_last_session_state()

    @pyqtSlot(str, str, dict, str)
    def _handle_sfm_session_loaded(self, model_name: str, personality: Optional[str], proj_ctx_data: Dict[str, Any], active_pid: str):
        # This method content is from your combined.txt
        if not (self._project_context_manager and self._backend_coordinator): return
        self._project_context_manager.load_state(proj_ctx_data)
        self._current_chat_model_name = model_name or constants.DEFAULT_GEMINI_CHAT_MODEL
        self._current_chat_personality_prompt = personality
        self._configure_initial_backends(self._current_chat_model_name, self._current_chat_personality_prompt)
        self.current_project_changed.emit(active_pid) # This will trigger tab ensure
        self._update_rag_initialized_state(emit_status=True, project_id=active_pid)
        self.update_status_based_on_state()

    @pyqtSlot()
    def _handle_sfm_active_history_cleared(self):
        # This method content is from your combined.txt
        if self._project_context_manager:
            active_project_id = self._project_context_manager.get_active_project_id()
            if active_project_id:
                history = self._project_context_manager.get_project_history(active_project_id)
                if history is not None: history.clear() # Clears the list in-place
                self.history_changed.emit([]) # Emit empty list

    @pyqtSlot(str, str, dict)
    def _handle_sfm_request_state_save(self, model_name: str, personality: Optional[str], all_project_data: Dict[str, Any]):
        # This method content is from your combined.txt
        if self._session_flow_manager:
            # ChatManager is responsible for providing the latest data to SFM for saving
            self._session_flow_manager.save_current_session_to_last_state(model_name, personality)

    @pyqtSlot(str)
    def _handle_backend_stream_started(self, request_id: str):
        # This method content is from your combined.txt
        logger.info(f"CM: BackendCoordinator reported stream_started for request_id '{request_id}'. Emitting to UI.")
        self.stream_started.emit(request_id)

    @pyqtSlot(str, str)
    def _handle_backend_chunk_received(self, request_id: str, chunk: str):
        # This method content is from your combined.txt
        current_active_mc_task = self._modification_coordinator and self._modification_coordinator.is_awaiting_llm_response()
        if not current_active_mc_task:
            self.stream_chunk_received.emit(chunk)
        else:
            logger.debug(f"CM: Suppressing stream chunk from req_id '{request_id}' during MC LLM wait.")

    @pyqtSlot(str, ChatMessage, dict)
    def _handle_backend_response_completed(self, request_id: str, completed_message: ChatMessage,
                                           usage_stats_with_metadata: dict):
        # This method content is from your combined.txt (with existing logs)
        mc_current_phase_debug = "N/A"
        mc_is_awaiting_llm_debug = "N/A"
        if self._modification_coordinator:
            mc_current_phase_debug = self._modification_coordinator._current_phase
            mc_is_awaiting_llm_debug = self._modification_coordinator.is_awaiting_llm_response()

        logger.info(
            f"CM _handle_backend_response_completed TOP: ReqID='{request_id}', "
            f"Purpose='{usage_stats_with_metadata.get('purpose')}', "
            f"BackendForMC='{usage_stats_with_metadata.get('backend_id_for_mc')}', "
            f"MC_Phase='{mc_current_phase_debug}', MC_AwaitingLLM='{mc_is_awaiting_llm_debug}', "
            f"FullMeta='{usage_stats_with_metadata}'"
        )

        purpose = usage_stats_with_metadata.get("purpose")
        backend_id_for_mc = usage_stats_with_metadata.get("backend_id_for_mc")

        is_mc_related_purpose = purpose and isinstance(purpose, str) and purpose.startswith("mc_request_")

        if self._modification_coordinator and self._modification_coordinator.is_active() and is_mc_related_purpose:
            mc_is_expecting_this = False
            if self._modification_coordinator.is_awaiting_llm_response():
                mc_is_expecting_this = True
            elif (self._modification_coordinator._current_phase == ModPhase.AWAITING_CODE_GENERATION and backend_id_for_mc == GENERATOR_BACKEND_ID):
                mc_is_expecting_this = True
            elif (self._modification_coordinator._current_phase == ModPhase.AWAITING_PLAN and backend_id_for_mc == PLANNER_BACKEND_ID):
                mc_is_expecting_this = True
            elif (self._modification_coordinator._current_phase == ModPhase.AWAITING_GEMINI_REFINEMENT and backend_id_for_mc == PLANNER_BACKEND_ID):
                mc_is_expecting_this = True

            if mc_is_expecting_this:
                logger.info(f"CM: Routing completed LLM response for req_id '{request_id}' (Purpose: {purpose}) to MC.")
                self._modification_coordinator.process_llm_response(
                    backend_id_for_mc or PLANNER_BACKEND_ID, # Fallback to planner if not specified
                    completed_message
                )
                return
            else:
                logger.warning(f"CM: Response for req_id '{request_id}' has MC purpose ('{purpose}') but MC is not in a matching await state (Phase: {self._modification_coordinator._current_phase}, AwaitingLLM: {self._modification_coordinator.is_awaiting_llm_response()}). This response will NOT be displayed in chat.")
                return

        original_target_filename = usage_stats_with_metadata.get("original_target_filename")
        if purpose == "code_summary" and original_target_filename:
            logger.info(f"CM: Handling completed response as a CODE SUMMARY for '{original_target_filename}'.")
            self.status_update.emit(f"AvA's summary for '{original_target_filename}' is ready!", "#98c379", True, 3000)
            summary_msg_text = f"✨ **AvA's Summary for {original_target_filename}:** ✨\n\n{completed_message.text}"
            summary_chat_message = ChatMessage(role=SYSTEM_ROLE, parts=[summary_msg_text],
                                               metadata={"is_ava_summary": True, "target_file": original_target_filename, "is_internal": False})
            if self._project_context_manager:
                self._project_context_manager.add_message_to_active_project(summary_chat_message)
                self.new_message_added.emit(summary_chat_message)
                self._trigger_save_last_session_state()
            return

        logger.info(f"CM: Handling response for req_id '{request_id}' as normal chat completion (UI display).")
        message_updated_in_model = False
        if self._project_context_manager:
            active_history = self._project_context_manager.get_active_conversation_history()
            if active_history:
                for i, msg_in_history in enumerate(reversed(active_history)):
                    if msg_in_history.id == request_id and msg_in_history.role == MODEL_ROLE:
                        logger.debug(f"CM: Found placeholder message ID '{request_id}' at reverse index {i} to update for UI.")
                        msg_in_history.parts = completed_message.parts
                        if completed_message.metadata:
                            if msg_in_history.metadata is None: msg_in_history.metadata = {}
                            msg_in_history.metadata.update(completed_message.metadata)
                        msg_in_history.loading_state = MessageLoadingState.COMPLETED
                        self.new_message_added.emit(msg_in_history)
                        message_updated_in_model = True
                        break
            if not message_updated_in_model:
                logger.warning(f"CM: Could not find existing AI message with ID '{request_id}' to update for UI. Adding as new (unexpected).")
                if completed_message.metadata is None: completed_message.metadata = {}
                completed_message.metadata["request_id"] = request_id
                completed_message.loading_state = MessageLoadingState.COMPLETED
                self._project_context_manager.add_message_to_active_project(completed_message)
                self.new_message_added.emit(completed_message)
            self._trigger_save_last_session_state()

        self.stream_finished.emit()

        prompt_tokens = usage_stats_with_metadata.get("prompt_tokens")
        completion_tokens = usage_stats_with_metadata.get("completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None and self._model_info_service:
            model_max_context = self._model_info_service.get_max_tokens(self._current_chat_model_name)
            self.token_usage_updated.emit(DEFAULT_CHAT_BACKEND_ID, prompt_tokens, completion_tokens, model_max_context)
        return

    @pyqtSlot(str, str)
    def _handle_backend_response_error(self, request_id: str, error_message_str: str):
        # This method content is from your combined.txt
        logger.error(f"CM: Received ERROR from BC for request_id '{request_id}': {error_message_str}")
        if self._modification_coordinator and self._modification_coordinator.is_active():
            logger.info(f"CM: Routing backend error for req_id '{request_id}' to MC.")
            self._modification_coordinator.process_llm_error(DEFAULT_CHAT_BACKEND_ID, error_message_str)
            return

        message_updated_in_model = False
        if self._project_context_manager:
            active_history = self._project_context_manager.get_active_conversation_history()
            if active_history:
                for i, msg_in_history in enumerate(reversed(active_history)):
                     if msg_in_history.id == request_id and msg_in_history.role == MODEL_ROLE:
                        logger.debug(f"CM: Found placeholder message ID '{request_id}' to update with error info.")
                        msg_in_history.role = ERROR_ROLE
                        msg_in_history.parts = [f"Backend Error (Request ID: {request_id[:8]}...): {error_message_str}"]
                        msg_in_history.loading_state = MessageLoadingState.COMPLETED
                        self.new_message_added.emit(msg_in_history)
                        message_updated_in_model = True
                        break
            if not message_updated_in_model:
                logger.warning(f"CM: Error for req_id '{request_id}', but no placeholder found. Adding new error message.")
                err_obj = ChatMessage(id=request_id, role=ERROR_ROLE,
                                      parts=[f"Backend Error (Request ID: {request_id[:8]}...): {error_message_str}"],
                                      loading_state=MessageLoadingState.COMPLETED)
                self._project_context_manager.add_message_to_active_project(err_obj)
                self.new_message_added.emit(err_obj)
            self._trigger_save_last_session_state()

        self.stream_finished.emit()
        self.error_occurred.emit(f"Backend Error: {error_message_str}", False)
        return

    @pyqtSlot(bool)
    def _handle_backend_busy_changed(self, backend_is_busy: bool):
        # This method content is from your combined.txt
        logger.debug(f"CM: BC overall busy state changed to: {backend_is_busy}")
        self._update_overall_busy_state()

    @pyqtSlot(str, str, bool, list)
    def _handle_backend_configuration_changed(self, backend_id: str, model_name: str, is_configured: bool, available_models: list):
        # This method content is from your combined.txt
        logger.info(f"CM: BC config changed for '{backend_id}'. Model: {model_name}, ConfigOK: {is_configured}, Avail: {len(available_models)}")
        if backend_id == DEFAULT_CHAT_BACKEND_ID:
            self._current_chat_model_name = model_name
            self._chat_backend_configured_successfully = is_configured
            self._available_chat_models = available_models[:]
            if not is_configured and self._backend_coordinator:
                err = self._backend_coordinator.get_last_error_for_backend(DEFAULT_CHAT_BACKEND_ID) or "Chat API config error."
                self.error_occurred.emit(f"Chat API Config Error: {err}", False)
            self.available_models_changed.emit(self._available_chat_models[:])
            self.config_state_changed.emit(self._current_chat_model_name, self._chat_backend_configured_successfully and bool(self._current_chat_personality_prompt))
            self.update_status_based_on_state()
            self._trigger_save_last_session_state()
        elif backend_id in [PLANNER_BACKEND_ID, GENERATOR_BACKEND_ID, OLLAMA_CHAT_BACKEND_ID]:
            name_map = {PLANNER_BACKEND_ID: "Planner", GENERATOR_BACKEND_ID: "Generator", OLLAMA_CHAT_BACKEND_ID: "Ollama Chat"}
            d_name = name_map.get(backend_id, backend_id)
            if not is_configured and self._backend_coordinator:
                err = self._backend_coordinator.get_last_error_for_backend(backend_id) or f"{d_name} config error."
                self.error_occurred.emit(f"{d_name} ('{backend_id}') Config Error: {err}", False)
                self.status_update.emit(f"{d_name} ('{backend_id}') not configured.", "#e06c75", True, 5000)
            elif is_configured:
                self.status_update.emit(f"{d_name} ('{backend_id}') OK with {model_name}.", "#61afef", True, 3000)

    @pyqtSlot(bool, str)
    def _handle_upload_started(self, is_global: bool, item_description: str):
        # This method content is from your combined.txt
        active_project_name_str = "N/A"; pcm = self._project_context_manager
        if pcm: active_project_name_str = pcm.get_active_project_name() or "Current"
        context_name = constants.GLOBAL_CONTEXT_DISPLAY_NAME if is_global else active_project_name_str
        self.status_update.emit(f"Uploading {item_description} to '{context_name}' context...", "#61afef", False, 0)
        self._update_overall_busy_state()

    @pyqtSlot(ChatMessage)
    def _handle_upload_summary(self, summary_message: ChatMessage):
        # This method content is from your combined.txt
        if not self._project_context_manager: return
        self._project_context_manager.add_message_to_active_project(summary_message)
        self.new_message_added.emit(summary_message)
        s_cid = summary_message.metadata.get("collection_id") if summary_message.metadata else None
        self._update_rag_initialized_state(emit_status=True, project_id=s_cid)
        self.update_status_based_on_state(); self._trigger_save_last_session_state()

    @pyqtSlot(str)
    def _handle_upload_error(self, error_message_str: str):
        # This method content is from your combined.txt
        if not self._project_context_manager: return
        err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Upload System Error: {error_message_str}"])
        self._project_context_manager.add_message_to_active_project(err_obj); self.new_message_added.emit(err_obj)
        self.error_occurred.emit(f"Upload Error: {error_message_str}", False); self.update_status_based_on_state()

    @pyqtSlot(bool)
    def _handle_upload_busy_changed(self, upload_is_busy: bool):
        # This method content is from your combined.txt
        self._update_overall_busy_state()

    @pyqtSlot(list) # List[ChatMessage] where ChatMessage is the *clean* user message for UI
    def _handle_uih_normal_chat_request(self, new_user_message_list: List[ChatMessage]):
        # This method content is from your combined.txt (with existing log)
        logger.info("CM: Handling normal_chat_request_ready from UIH.")
        if not (self._backend_coordinator and self._project_context_manager):
            self.error_occurred.emit("Cannot send chat: Critical components missing.", True); return
        if not new_user_message_list or not isinstance(new_user_message_list[0], ChatMessage):
            logger.warning("CM: _handle_uih_normal_chat_request received invalid or empty message list. Ignoring."); return

        # This is the clean user message for UI display
        user_message_for_ui = new_user_message_list[0]
        self._project_context_manager.add_message_to_active_project(user_message_for_ui)
        self.new_message_added.emit(user_message_for_ui)
        self._trigger_save_last_session_state()
        logger.debug(f"CM: Added user message (ID: {user_message_for_ui.id}) to history for UI: {user_message_for_ui.text[:50]}...")

        QApplication.processEvents() # Allow UI to update with user message

        ai_request_id = str(uuid.uuid4())
        ai_placeholder_message = ChatMessage(id=ai_request_id, role=MODEL_ROLE, parts=[""], loading_state=MessageLoadingState.LOADING)
        self._project_context_manager.add_message_to_active_project(ai_placeholder_message)
        self.new_message_added.emit(ai_placeholder_message) # Show placeholder in UI
        logger.info(f"CM: Added AI placeholder (ID: {ai_request_id}) with LOADING state.")

        # The history for the backend should include the RAG-augmented version of the user's query.
        # UserInputProcessor._prepare_normal_chat_prompt creates this.
        # For now, we assume the history in PCM already contains the augmented message IF RAG was used.
        # This part needs to be robust: the message sent to the LLM should be the one processed by UIP.

        # Let's get the full history, which should now include the user_message_for_ui.
        # The _actual_ last message sent to the LLM (before the placeholder) needs to be the augmented one.
        # This means the UserInputProcessor's output (the augmented message) should have been what was
        # added to history if RAG was involved.
        # For this iteration, we assume the last message in history IS the augmented one if RAG applied.
        # This is a slight simplification of the ideal flow where UIP separates original and augmented.

        full_history_for_backend = self._project_context_manager.get_active_conversation_history()
        if not full_history_for_backend:
            logger.error("CM: History for backend is empty. This is unexpected."); self.error_occurred.emit("Internal error preparing chat.", True); return

        request_options = {"temperature": self._current_chat_temperature}
        # Metadata for BC to track the original request, not for the LLM.
        request_metadata_for_bc = {"original_user_message_id": user_message_for_ui.id }
        logger.debug(f"CM: Sending history (len {len(full_history_for_backend)}) to backend for request_id '{ai_request_id}'.")
        self._backend_coordinator.request_response_stream(
            target_backend_id=DEFAULT_CHAT_BACKEND_ID, request_id=ai_request_id,
            history_to_send=full_history_for_backend[:-1], # Send all but the AI placeholder itself
            is_modification_response_expected=False,
            options=request_options, request_metadata=request_metadata_for_bc
        )

    # --- UIH Slot Handlers (Modification Flow) ---
    # --- MODIFICATION START: Update slot signature and implementation ---
    @pyqtSlot(str, list, str, str) # original_query_text, image_data_list, context_for_mc, focus_prefix_for_mc
    def _handle_uih_mod_start_request(self,
                                      original_query_text: str,
                                      image_data_list: List[Dict[str, Any]],
                                      context_for_mc: str,
                                      focus_prefix_for_mc: str):
        logger.info(f"CM: Handling modification_sequence_start_requested from UIH. Query: '{original_query_text[:50]}...'")

        if not (self._modification_coordinator and self._project_context_manager):
            self.error_occurred.emit("Modification feature unavailable or PCM missing.", True)
            return

        # 1. Create and add the user's initial query message to history for UI display
        user_message_parts_for_ui = [original_query_text] + (image_data_list or [])
        user_chat_message_for_ui = ChatMessage(role=USER_ROLE, parts=user_message_parts_for_ui)

        self._project_context_manager.add_message_to_active_project(user_chat_message_for_ui)
        self.new_message_added.emit(user_chat_message_for_ui) # Show user's message in UI
        self._trigger_save_last_session_state()
        logger.debug(f"CM: Added user's modification request (ID: {user_chat_message_for_ui.id}) to history for UI display.")

        QApplication.processEvents() # Allow UI to update

        # 2. Activate ModificationHandler sequence (if available)
        if self._modification_handler_instance:
            self._modification_handler_instance.activate_sequence()

        # 3. Start the ModificationCoordinator sequence
        self._modification_coordinator.start_sequence(
            query=original_query_text,
            context=context_for_mc,
            focus_prefix=focus_prefix_for_mc
        )
        logger.info(f"CM: ModificationCoordinator sequence started for query: '{original_query_text[:50]}...'")
    # --- MODIFICATION END ---

    @pyqtSlot(str, str) # user_command, action_type
    def _handle_uih_mod_user_input(self, user_command: str, action_type: str):
        # This method content is from your combined.txt
        logger.info(f"CM: Handling mod_user_input ('{user_command}', Type: '{action_type}') from UIH.")
        if self._modification_coordinator:
            self._modification_coordinator.process_user_input(user_command)
        else:
            self.error_occurred.emit("Modification feature unavailable.", False)

    @pyqtSlot(str)
    def _handle_uih_processing_error(self, error_message: str):
        # This method content is from your combined.txt
        logger.error(f"CM: UIH processing error: {error_message}")
        self.error_occurred.emit(f"Input Processing Error: {error_message}", False)
        if self._project_context_manager:
            err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Input Error: {error_message}"])
            self._project_context_manager.add_message_to_active_project(err_obj)
            self.new_message_added.emit(err_obj)

    # --- ModificationCoordinator Slot Handlers ---
    @pyqtSlot(str, list)
    def _handle_mc_request_llm_call(self, target_backend_id: str, history_to_send: List[ChatMessage]):
        # This method content is from your combined.txt (with existing log)
        if self._backend_coordinator:
            mc_options = {"temperature": 0.5}
            if target_backend_id == GENERATOR_BACKEND_ID: mc_options = {"temperature": 0.2}

            mc_internal_request_id = f"mc_{target_backend_id}_{str(uuid.uuid4())[:8]}"
            request_metadata_for_mc = {
                "purpose": f"mc_request_{target_backend_id}",
                "mc_internal_id": mc_internal_request_id,
                "backend_id_for_mc": target_backend_id
            }
            logger.debug(f"CM MC LLM Call: Target='{target_backend_id}', ReqID='{mc_internal_request_id}', Meta='{request_metadata_for_mc}'")

            self._backend_coordinator.request_response_stream(
                target_backend_id=target_backend_id, request_id=mc_internal_request_id,
                history_to_send=history_to_send,
                is_modification_response_expected=True, # True for MC calls
                options=mc_options,
                request_metadata=request_metadata_for_mc
            )
        elif self._modification_coordinator:
            self._modification_coordinator.process_llm_error(target_backend_id, "BackendCoordinator unavailable.")

    @pyqtSlot(str, str, str)
    def _handle_code_generated_and_summary_needed(self, generated_code: str, coder_instructions: str, target_filename: str):
        # This method content is from your combined.txt
        logger.info(f"CM: Summary needed for '{target_filename}'. Delegating to CodeSummaryService.")
        if not (self._code_summary_service and self._backend_coordinator):
            self.error_occurred.emit(f"Internal error: Services unavailable for summary of '{target_filename}'.", True); return

        self.status_update.emit(f"AvA is preparing summary for '{target_filename}'...", "#e5c07b", True, 4000)
        success = self._code_summary_service.request_code_summary(
            self._backend_coordinator,
            target_filename,
            coder_instructions,
            generated_code
        )
        if not success:
            err_msg = f"Failed to dispatch summary request for '{target_filename}'."; logger.error(err_msg)
            if self._project_context_manager:
                sys_err_msg = ChatMessage(role=ERROR_ROLE, parts=[f"[System: Error initiating summary for '{target_filename}'.]"])
                self._project_context_manager.add_message_to_active_project(sys_err_msg); self.new_message_added.emit(sys_err_msg)
            self.update_status_based_on_state()

    @pyqtSlot(str, str)
    def _handle_mc_file_ready(self, filename: str, content: str):
        # This method content is from your combined.txt
        self.code_file_updated.emit(filename, content)
        if self._project_context_manager:
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[f"[System: File '{filename}' updated. See Code Viewer.]"], metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(sys_msg); self.new_message_added.emit(sys_msg)

    @pyqtSlot(str, str) # reason, original_query_summary (from previous fix)
    def _handle_mc_sequence_complete(self, reason: str, original_query_summary: str):
        # This method content is from your combined.txt (incorporating the previous fix)
        if self._project_context_manager:
            system_message_text = (
                f"[System: The multi-file code modification sequence by the Coder AI for "
                f"'{original_query_summary}' has ended ({reason}). "
                f"All generated code is available in the Code Viewer. "
                f"AvA, please respond conversationally to the user's next message. "
                f"Do not output full code blocks for the task that just completed.]"
            )
            logger.info(f"CM: Modification sequence complete. Adding guiding system message: {system_message_text}")
            sys_msg = ChatMessage(
                role=SYSTEM_ROLE,
                parts=[system_message_text],
                metadata={"is_internal": True}
            )
            self._project_context_manager.add_message_to_active_project(sys_msg)
            self._trigger_save_last_session_state()

        self.update_status_based_on_state()
        if self._modification_handler_instance:
            self._modification_handler_instance.cancel_modification()

    @pyqtSlot(str)
    def _handle_mc_error(self, error_message: str):
        # This method content is from your combined.txt
        if self._project_context_manager:
            err_msg_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Modification System Error: {error_message}"], metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(err_msg_obj); self.new_message_added.emit(err_msg_obj)
            self._trigger_save_last_session_state()
        self.error_occurred.emit(f"Modification Error: {error_message}", False); self.update_status_based_on_state()

    @pyqtSlot(str)
    def _handle_mc_status_update(self, message: str):
        # This method content is from your combined.txt
        if self._project_context_manager:
            status_msg = ChatMessage(role=SYSTEM_ROLE, parts=[message], metadata={"is_internal": False})
            self._project_context_manager.add_message_to_active_project(status_msg); self.new_message_added.emit(status_msg)

    # --- Action Methods & Getters/Setters (All from your combined.txt, no changes needed below this line for this fix) ---
    def _cancel_active_tasks(self):
        if self._backend_coordinator: self._backend_coordinator.cancel_current_task()
        if self._upload_coordinator: self._upload_coordinator.cancel_current_upload()
        if self._modification_coordinator and self._modification_coordinator.is_active():
            self._modification_coordinator.cancel_sequence(reason="user_cancel_all")

    def cleanup(self):
        self._cancel_active_tasks(); self._trigger_save_last_session_state()

    def _update_rag_initialized_state(self, emit_status: bool = True, project_id: Optional[str] = None):
        if not self._project_context_manager: return
        target_pid = project_id or (self._project_context_manager.get_active_project_id())
        new_init_state = self.is_rag_context_initialized(target_pid)
        active_pid = self._project_context_manager.get_active_project_id()
        if target_pid == active_pid:
            if self._rag_initialized != new_init_state: self._rag_initialized = new_init_state
            if emit_status or (self._rag_initialized != new_init_state) : self.update_status_based_on_state()
        elif emit_status:
            self.update_status_based_on_state()

    def is_rag_context_initialized(self, project_id: Optional[str]) -> bool:
        if not (self._vector_db_service and project_id): self._rag_available = False; return False
        self._rag_available = True
        is_vdb_ready = self._vector_db_service.is_ready(project_id)
        size = self._vector_db_service.get_collection_size(project_id) if is_vdb_ready else 0
        return is_vdb_ready and size > 0

    def get_project_history(self, project_id: str) -> List[ChatMessage]:
        return list(self._project_context_manager.get_project_history(project_id) or []) if self._project_context_manager else []
    def get_current_history(self) -> List[ChatMessage]:
        return list(self._project_context_manager.get_active_conversation_history() or []) if self._project_context_manager else []
    def get_current_model(self) -> str: return self._current_chat_model_name
    def get_current_personality(self) -> Optional[str]: return self._current_chat_personality_prompt
    def get_current_project_id(self) -> Optional[str]:
        return self._project_context_manager.get_active_project_id() if self._project_context_manager else None
    def is_api_ready(self) -> bool: return self._chat_backend_configured_successfully
    def is_overall_busy(self) -> bool: return self._overall_busy
    def is_rag_available(self) -> bool: return self._rag_available
    def get_rag_contents(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not (self._project_context_manager and self._vector_db_service): return []
        target_id = collection_id or (self._project_context_manager.get_active_project_id())
        if not target_id or not self._vector_db_service.is_ready(target_id): return []
        try: return self._vector_db_service.get_all_metadata(target_id)
        except Exception as e: logger.exception(f"Error RAG contents for '{target_id}': {e}"); return []
    def get_current_focus_paths(self) -> Optional[List[str]]: return self._current_chat_focus_paths
    def get_project_context_manager(self) -> Optional[ProjectContextManager]: return self._project_context_manager
    def get_backend_coordinator(self) -> Optional[BackendCoordinator]: return self._backend_coordinator
    def get_upload_coordinator(self) -> Optional[UploadCoordinator]: return self._upload_coordinator
    def get_modification_coordinator(self) -> Optional[ModificationCoordinator]: return self._modification_coordinator
    def get_session_flow_manager(self) -> Optional[SessionFlowManager]: return self._session_flow_manager

    def _trigger_save_last_session_state(self):
        if self._session_flow_manager:
            self._session_flow_manager.save_current_session_to_last_state(self._current_chat_model_name, self._current_chat_personality_prompt)

    def set_model(self, model_name: str):
        self._current_chat_model_name = model_name
        if self._backend_coordinator:
            self._backend_coordinator.configure_backend(DEFAULT_CHAT_BACKEND_ID, get_api_key(), model_name, self._current_chat_personality_prompt)
    def set_personality(self, prompt: Optional[str]):
        self._current_chat_personality_prompt = prompt.strip() if prompt else None
        if self._backend_coordinator:
            self._backend_coordinator.configure_backend(DEFAULT_CHAT_BACKEND_ID, get_api_key(), self._current_chat_model_name, self._current_chat_personality_prompt)
    def set_current_project(self, project_id: str):
        if self._project_context_manager:
            if not self._project_context_manager.set_active_project(project_id): self.error_occurred.emit(f"Failed to set project '{project_id}'.", False)
    def create_project_collection(self, project_name: str):
        if self._project_context_manager:
            if not self._project_context_manager.create_project(project_name): self.error_occurred.emit(f"Failed to create project '{project_name}'.", False)
            else: self.status_update.emit(f"Project '{project_name}' created.", "#98c379", True, 3000)

    def start_new_chat(self):
        if self._session_flow_manager: self._session_flow_manager.start_new_chat_session(self._current_chat_model_name, self._current_chat_personality_prompt)
    def load_chat_session(self, filepath: str):
        if self._session_flow_manager: self._session_flow_manager.load_named_session(filepath, DEFAULT_CHAT_BACKEND_ID)
    def save_current_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager: return self._session_flow_manager.save_session_as(filepath, self._current_chat_model_name, self._current_chat_personality_prompt)
        return False
    def delete_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager: return self._session_flow_manager.delete_named_session(filepath)
        return False
    def list_saved_sessions(self) -> List[str]:
        if self._session_flow_manager: return self._session_flow_manager.list_saved_sessions()
        return []

    def process_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        if self._user_input_handler:
            self._user_input_handler.handle_user_message(text=text, image_data=image_data, focus_paths=self._current_chat_focus_paths,
                                                         rag_available=self._rag_available,
                                                         rag_initialized_for_project=self.is_rag_context_initialized(self.get_current_project_id()))
    def update_status_based_on_state(self):
        if not self._chat_backend_configured_successfully:
            err = self._backend_coordinator.get_last_error_for_backend(DEFAULT_CHAT_BACKEND_ID) if self._backend_coordinator else "Config error"
            self.status_update.emit(f"API Not Configured: {err or 'Check settings.'}", "#e06c75", False, 0)
        elif self._overall_busy: self.status_update.emit("Processing...", "#e5c07b", False, 0)
        else:
            parts = ["Ready"]
            if self._project_context_manager:
                pid = self._project_context_manager.get_active_project_id()
                pname = self._project_context_manager.get_project_name(pid) or "Unknown"
                if pid == constants.GLOBAL_COLLECTION_ID: pname = constants.GLOBAL_CONTEXT_DISPLAY_NAME
                parts.append(f"(Ctx: {pname})")
            if self.is_rag_context_initialized(self._project_context_manager.get_active_project_id() if self._project_context_manager else None):
                parts.append("[RAG Active]")
            self.status_update.emit(" ".join(parts), "#98c379", False, 0)

    def set_chat_temperature(self, temperature: float):
        if 0.0 <= temperature <= 2.0:
            self._current_chat_temperature = temperature
            self.status_update.emit(f"Temperature set to {self._current_chat_temperature:.2f}", "#61afef", True, 3000)
            self._trigger_save_last_session_state()

    def handle_file_upload(self, file_paths: List[str]):
        if self._upload_coordinator: self._upload_coordinator.upload_files_to_current_project(file_paths)
    def handle_directory_upload(self, dir_path: str):
        if self._upload_coordinator: self._upload_coordinator.upload_directory_to_current_project(dir_path)
    def handle_global_file_upload(self, file_paths: List[str]):
        if self._upload_coordinator: self._upload_coordinator.upload_files_to_global(file_paths)
    def handle_global_directory_upload(self, dir_path: str):
        if self._upload_coordinator: self._upload_coordinator.upload_directory_to_global(dir_path)

    def set_chat_focus(self, paths: List[str]):
        self._current_chat_focus_paths = paths
        display = ", ".join([os.path.basename(p) for p in paths]);
        if len(display) > 50: display = display[:47] + "..."
        self.status_update.emit(f"Focus set on: {display}", "#61afef", True, 4000)

    def _update_overall_busy_state(self):
        be_busy = self._backend_coordinator.is_processing_request() if self._backend_coordinator else False
        ul_busy = self._upload_coordinator.is_busy() if self._upload_coordinator else False
        new_busy = be_busy or ul_busy
        if self._overall_busy != new_busy:
            self._overall_busy = new_busy
            self.busy_state_changed.emit(self._overall_busy)
            self.update_status_based_on_state()