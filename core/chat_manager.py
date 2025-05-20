import logging
import asyncio
import os
import uuid
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtWidgets import QApplication

from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from core.message_enums import MessageLoadingState
from backend.interface import BackendInterface
from services.code_summary_service import CodeSummaryService
from services.model_info_service import ModelInfoService
from services.session_service import SessionService
from services.upload_service import UploadService
from services.vector_db_service import VectorDBService
from .modification_coordinator import ModPhase, ModificationCoordinator
from .project_context_manager import ProjectContextManager
from .backend_coordinator import BackendCoordinator
from .project_summary_coordinator import ProjectSummaryCoordinator
from .session_flow_manager import SessionFlowManager
from .upload_coordinator import UploadCoordinator
from .rag_handler import RagHandler
from .user_input_handler import UserInputHandler
from .user_input_processor import UserInputProcessor
from .application_orchestrator import ApplicationOrchestrator


from utils import constants
from utils.constants import (
    DEFAULT_CHAT_BACKEND_ID, OLLAMA_CHAT_BACKEND_ID, GPT_CHAT_BACKEND_ID,
    PLANNER_BACKEND_ID, GENERATOR_BACKEND_ID, DEFAULT_GEMINI_CHAT_MODEL,
    DEFAULT_OLLAMA_MODEL, DEFAULT_GPT_MODEL, DEFAULT_GEMINI_PLANNER_MODEL
)

try:
    from config import APP_CONFIG


    def get_gemini_api_key():
        return APP_CONFIG.get("GEMINI_API_KEY")


    def get_openai_api_key():
        return os.getenv("OPENAI_API_KEY")
except ImportError:
    def get_gemini_api_key():
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


    def get_openai_api_key():
        return os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

CODER_AI_SYSTEM_PROMPT = """You are an expert Python code generation assistant. Your task is to generate or update the file specified, strictly adhering to the provided detailed instructions and any original file content.

**Key Requirements for Your Output:**

1.  **Accuracy:** Precisely implement the logic and features described in the instructions.
2.  **Completeness:** If updating, preserve unchanged original code. If new, generate all necessary components.
3.  **Standard Python Practices:**
    *   Follow PEP 8 style guidelines (e.g., line length, naming).
    *   Include type hints (PEP 484) for functions, methods, and important variables.
    *   Write clear docstrings (PEP 257) for modules, classes, functions, and methods.
    *   Add inline comments for complex or non-obvious logic.
4.  **Output Format:** Your *entire* response MUST be a single Markdown Python code block, starting with ```python path/to/filename.ext\\n and ending with ```. No other text or explanations.

Produce clean, readable, and correct Python code.
"""

USER_SELECTABLE_CHAT_BACKEND_DETAILS = [
    {"id": DEFAULT_CHAT_BACKEND_ID, "name": "Google Gemini"},
    {"id": OLLAMA_CHAT_BACKEND_ID, "name": "Ollama (Local Chat)"},
    {"id": GPT_CHAT_BACKEND_ID, "name": "OpenAI GPT"}
]

SPECIALIZED_BACKEND_DETAILS = [
    {"id": GENERATOR_BACKEND_ID, "name": "Ollama (Specialized)"}
]


class ChatManager(QObject):
    history_changed = pyqtSignal(list)
    new_message_added = pyqtSignal(object)
    status_update = pyqtSignal(str, str, bool, int)
    error_occurred = pyqtSignal(str, bool)
    busy_state_changed = pyqtSignal(bool)
    backend_config_state_changed = pyqtSignal(str, str, bool, bool)
    available_models_changed_for_backend = pyqtSignal(str, list)
    stream_started = pyqtSignal(str)
    stream_chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()
    code_file_updated = pyqtSignal(str, str)
    current_project_changed = pyqtSignal(str)
    project_inventory_updated = pyqtSignal(dict)
    token_usage_updated = pyqtSignal(str, int, int, int)

    def __init__(self, orchestrator: ApplicationOrchestrator, parent: Optional[QObject] = None):
        super().__init__(parent)
        if not isinstance(orchestrator, ApplicationOrchestrator):
            err_msg = "ChatManager requires a valid ApplicationOrchestrator instance."
            logger.critical(err_msg);
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
        self._project_summary_coordinator = self._orchestrator.get_project_summary_coordinator()
        if self._project_summary_coordinator: self._project_summary_coordinator.setParent(self)

        self._rag_handler = self._orchestrator.get_rag_handler()
        self._modification_handler_instance = self._orchestrator.get_modification_handler_instance()
        if self._modification_handler_instance and isinstance(self._modification_handler_instance, QObject):
            if self._modification_handler_instance.parent() is None:
                self._modification_handler_instance.setParent(self)
        self._session_service: Optional[SessionService] = getattr(orchestrator, '_session_service', None)
        self._vector_db_service: Optional[VectorDBService] = getattr(orchestrator, '_vector_db_service', None)
        self._code_summary_service = CodeSummaryService()
        self._model_info_service = ModelInfoService()

        self._initialize_state_variables()
        self._connect_component_signals()

    def _initialize_state_variables(self):
        self._overall_busy: bool = False
        self._current_active_chat_backend_id: str = DEFAULT_CHAT_BACKEND_ID
        all_backend_ids_from_adapters = sorted(list(self._backend_adapters_dict.keys()))

        self._current_model_names: Dict[str, str] = {bid: "" for bid in all_backend_ids_from_adapters}
        if DEFAULT_CHAT_BACKEND_ID in self._current_model_names: self._current_model_names[
            DEFAULT_CHAT_BACKEND_ID] = DEFAULT_GEMINI_CHAT_MODEL
        if OLLAMA_CHAT_BACKEND_ID in self._current_model_names: self._current_model_names[
            OLLAMA_CHAT_BACKEND_ID] = "llama3:latest"
        if GPT_CHAT_BACKEND_ID in self._current_model_names: self._current_model_names[
            GPT_CHAT_BACKEND_ID] = DEFAULT_GPT_MODEL
        if PLANNER_BACKEND_ID in self._current_model_names: self._current_model_names[
            PLANNER_BACKEND_ID] = DEFAULT_GEMINI_PLANNER_MODEL
        if GENERATOR_BACKEND_ID in self._current_model_names: self._current_model_names[
            GENERATOR_BACKEND_ID] = DEFAULT_OLLAMA_MODEL

        self._current_chat_personality_prompts: Dict[str, Optional[str]] = {bid: None for bid in
                                                                            all_backend_ids_from_adapters}
        if PLANNER_BACKEND_ID in self._current_chat_personality_prompts: self._current_chat_personality_prompts[
            PLANNER_BACKEND_ID] = "You are an expert planner and technical writer."
        if GENERATOR_BACKEND_ID in self._current_chat_personality_prompts:
            self._current_chat_personality_prompts[GENERATOR_BACKEND_ID] = CODER_AI_SYSTEM_PROMPT

        self._current_chat_temperature: float = 0.7
        self._chat_backend_configured_successfully: Dict[str, bool] = {bid: False for bid in
                                                                       all_backend_ids_from_adapters}
        self._available_models_per_backend: Dict[str, List[str]] = {bid: [] for bid in all_backend_ids_from_adapters}
        self._current_chat_focus_paths: Optional[List[str]] = None
        self._rag_available: bool = (self._vector_db_service is not None and hasattr(self._vector_db_service,
                                                                                     'is_ready') and self._vector_db_service.is_ready())
        self._rag_initialized: bool = self._rag_available

    def _connect_component_signals(self):
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
        if self._user_input_handler:
            self._user_input_handler.normal_chat_request_ready.connect(self._handle_uih_normal_chat_request)
            self._user_input_handler.modification_sequence_start_requested.connect(self._handle_uih_mod_start_request)
            self._user_input_handler.modification_user_input_received.connect(self._handle_uih_mod_user_input)
            self._user_input_handler.processing_error_occurred.connect(self._handle_uih_processing_error)
            self._user_input_handler.user_command_for_display_only.connect(self._handle_user_command_for_display_only)
        if self._modification_coordinator:
            self._modification_coordinator.request_llm_call.connect(self._handle_mc_request_llm_call)
            self._modification_coordinator.file_ready_for_display.connect(self._handle_mc_file_ready)
            self._modification_coordinator.modification_sequence_complete.connect(self._handle_mc_sequence_complete)
            self._modification_coordinator.modification_error.connect(self._handle_mc_error)
            self._modification_coordinator.status_update.connect(self._handle_mc_status_update)
        if self._project_summary_coordinator:
            self._project_summary_coordinator.summary_generated.connect(self._handle_project_summary_generated)
            self._project_summary_coordinator.summary_generation_failed.connect(self._handle_project_summary_failed)

    def initialize(self):
        if not (
                self._session_flow_manager and self._project_context_manager and self._user_input_handler and self._backend_coordinator):
            missing_deps = [name for comp, name in
                            [(self._session_flow_manager, "SFM"), (self._project_context_manager, "PCM"),
                             (self._user_input_handler, "UIH"), (self._backend_coordinator, "BC")] if not comp]
            self.error_occurred.emit(f"Critical error during init ({', '.join(missing_deps)} missing).", True);
            return

        loaded_state = self._session_flow_manager.load_last_session_state_on_startup()
        model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session = None, None, None, constants.GLOBAL_COLLECTION_ID
        active_backend_id_from_session, temperature_from_session, generator_model_from_session = None, None, None

        if loaded_state and len(loaded_state) == 7:
            model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session, active_backend_id_from_session, temperature_from_session, generator_model_from_session = loaded_state
        elif loaded_state and len(loaded_state) == 6:
            model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session, active_backend_id_from_session, temperature_from_session = loaded_state
        elif loaded_state and len(loaded_state) >= 4:
            model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session = loaded_state[:4]

        if active_backend_id_from_session and active_backend_id_from_session in self._current_model_names:
            self._current_active_chat_backend_id = active_backend_id_from_session
            if model_from_session: self._current_model_names[self._current_active_chat_backend_id] = model_from_session
            if pers_from_session: self._current_chat_personality_prompts[
                self._current_active_chat_backend_id] = pers_from_session
        elif model_from_session:
            self._current_model_names[DEFAULT_CHAT_BACKEND_ID] = model_from_session
            if pers_from_session: self._current_chat_personality_prompts[DEFAULT_CHAT_BACKEND_ID] = pers_from_session

        if temperature_from_session is not None:
            self._current_chat_temperature = temperature_from_session
        if generator_model_from_session and GENERATOR_BACKEND_ID in self._current_model_names:
            self._current_model_names[GENERATOR_BACKEND_ID] = generator_model_from_session

        if GENERATOR_BACKEND_ID in self._current_chat_personality_prompts and \
                self._current_chat_personality_prompts.get(GENERATOR_BACKEND_ID) is None:
            self._current_chat_personality_prompts[GENERATOR_BACKEND_ID] = CODER_AI_SYSTEM_PROMPT

        if proj_data_from_session:
            self._project_context_manager.load_state(proj_data_from_session)
        else:
            self._project_context_manager.set_active_project(constants.GLOBAL_COLLECTION_ID)

        self._perform_orphan_cleanup(self._project_context_manager.save_state())
        self._set_initial_active_project(active_pid_from_session, None)
        self._configure_all_initial_backends()
        self.update_status_based_on_state()
        current_active_project_id = self._project_context_manager.get_active_project_id()
        self._update_rag_initialized_state(emit_status=False, project_id=current_active_project_id)

    def _perform_orphan_cleanup(self, project_context_data_from_pcm: Optional[Dict[str, Any]]):
        if not (self._project_context_manager and self._vector_db_service): return

    def _set_initial_active_project(self, target_active_project_id: Optional[str], _):
        if not self._project_context_manager: return
        effective_target_id = target_active_project_id if target_active_project_id else constants.GLOBAL_COLLECTION_ID
        if not self._project_context_manager.get_project_history(effective_target_id):
            effective_target_id = constants.GLOBAL_COLLECTION_ID
        self._project_context_manager.set_active_project(effective_target_id)

    def _configure_all_initial_backends(self):
        if not self._backend_coordinator: return
        gemini_api_key, openai_api_key = get_gemini_api_key(), get_openai_api_key()

        for backend_id in self._backend_adapters_dict.keys():
            model_to_use = self._current_model_names.get(backend_id)
            if not model_to_use:
                if backend_id == DEFAULT_CHAT_BACKEND_ID:
                    model_to_use = DEFAULT_GEMINI_CHAT_MODEL
                elif backend_id == OLLAMA_CHAT_BACKEND_ID:
                    model_to_use = "llama3:latest"
                elif backend_id == GPT_CHAT_BACKEND_ID:
                    model_to_use = DEFAULT_GPT_MODEL
                elif backend_id == PLANNER_BACKEND_ID:
                    model_to_use = DEFAULT_GEMINI_PLANNER_MODEL
                elif backend_id == GENERATOR_BACKEND_ID:
                    model_to_use = DEFAULT_OLLAMA_MODEL
                else:
                    model_to_use = "default_model_placeholder"
                self._current_model_names[backend_id] = model_to_use

            personality_to_use = self._current_chat_personality_prompts.get(backend_id)
            if backend_id == GENERATOR_BACKEND_ID and personality_to_use is None:
                personality_to_use = CODER_AI_SYSTEM_PROMPT
                self._current_chat_personality_prompts[GENERATOR_BACKEND_ID] = personality_to_use

            api_key_for_this_backend = None
            if backend_id.startswith("gemini"):
                api_key_for_this_backend = gemini_api_key
            elif backend_id.startswith("gpt"):
                api_key_for_this_backend = openai_api_key
            self._backend_coordinator.configure_backend(backend_id, api_key_for_this_backend, model_to_use,
                                                        personality_to_use)

    def get_all_available_chat_models_with_details(self) -> List[Dict[str, Any]]:
        all_models_details = []
        for provider_detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS:
            backend_id, provider_name = provider_detail["id"], provider_detail["name"]
            if backend_id not in self._backend_adapters_dict: continue
            try:
                for model_name_from_adapter in self.get_models_for_backend(backend_id):
                    all_models_details.append(
                        {"display_name": f"{provider_name}: {model_name_from_adapter}", "backend_id": backend_id,
                         "model_name": model_name_from_adapter})
            except Exception as e:
                logger.error(f"Error fetching/processing models for chat backend {backend_id}: {e}")
        return all_models_details

    def get_all_available_specialized_models_with_details(self) -> List[Dict[str, Any]]:
        all_models_details = []
        try:
            for model_name_from_adapter in self.get_models_for_backend(GENERATOR_BACKEND_ID):
                all_models_details.append({"display_name": f"Ollama (Specialized): {model_name_from_adapter}",
                                           "backend_id": GENERATOR_BACKEND_ID, "model_name": model_name_from_adapter})
        except Exception as e:
            logger.error(f"Error fetching/processing models for specialized backend {GENERATOR_BACKEND_ID}: {e}")
        return all_models_details

    def get_available_backend_details(self) -> List[Dict[str, str]]:
        return [detail for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS if
                detail['id'] in self._backend_adapters_dict]

    def get_models_for_backend(self, backend_id: str) -> List[str]:
        if backend_id not in self._backend_adapters_dict: return self._available_models_per_backend.get(backend_id, [])
        if self._backend_coordinator:
            try:
                models = self._backend_coordinator.get_available_models_for_backend(backend_id)
                self._available_models_per_backend[backend_id] = models[:]
                return models
            except Exception as e:
                logger.exception(f"Error fetching models for backend '{backend_id}' via BackendCoordinator:")
        return self._available_models_per_backend.get(backend_id, [])

    def get_model_for_backend(self, backend_id: str) -> Optional[str]:
        return self._current_model_names.get(backend_id)

    def get_current_active_chat_backend_id(self) -> str:
        return self._current_active_chat_backend_id

    @pyqtSlot(dict)
    def _handle_pcm_project_list_updated(self, projects_dict: Dict[str, str]):
        self.project_inventory_updated.emit(projects_dict)
        if self._project_context_manager:
            current_active_id_in_pcm = self._project_context_manager.get_active_project_id()
            if current_active_id_in_pcm not in projects_dict and current_active_id_in_pcm != constants.GLOBAL_COLLECTION_ID:
                self.set_current_project(constants.GLOBAL_COLLECTION_ID)
            elif not projects_dict and current_active_id_in_pcm != constants.GLOBAL_COLLECTION_ID:
                self.set_current_project(constants.GLOBAL_COLLECTION_ID)

    @pyqtSlot(str)
    def _handle_pcm_active_project_changed(self, new_active_project_id: str):
        active_history = self.get_project_history(new_active_project_id)
        self.history_changed.emit(active_history[:])
        self.current_project_changed.emit(new_active_project_id)
        self._update_rag_initialized_state(emit_status=True, project_id=new_active_project_id)
        self._trigger_save_last_session_state()

    @pyqtSlot(str, str, dict, str)
    def _handle_sfm_session_loaded(self, model_name: str, personality: Optional[str], proj_ctx_data: Dict[str, Any],
                                   active_pid_from_session: str):
        if not (self._project_context_manager and self._backend_coordinator): return
        session_extra_data = proj_ctx_data.pop("session_extra_data_on_load", None)
        active_backend_id_from_session, temperature_from_session, generator_model_from_session = DEFAULT_CHAT_BACKEND_ID, None, None
        if session_extra_data and isinstance(session_extra_data, dict):
            active_backend_id_from_session = session_extra_data.get("active_chat_backend_id", DEFAULT_CHAT_BACKEND_ID)
            if (temp_val := session_extra_data.get("chat_temperature")) is not None:
                try:
                    temperature_from_session = float(temp_val)
                except (ValueError, TypeError):
                    pass
            generator_model_from_session = session_extra_data.get("generator_model_name")

        if active_backend_id_from_session in self._current_model_names:
            self._current_active_chat_backend_id = active_backend_id_from_session
            self._current_model_names[
                self._current_active_chat_backend_id] = model_name or self._current_model_names.get(
                self._current_active_chat_backend_id, "")
            self._current_chat_personality_prompts[self._current_active_chat_backend_id] = personality
        else:
            self._current_model_names[DEFAULT_CHAT_BACKEND_ID] = model_name or DEFAULT_GEMINI_CHAT_MODEL
            self._current_chat_personality_prompts[DEFAULT_CHAT_BACKEND_ID] = personality
            self._current_active_chat_backend_id = DEFAULT_CHAT_BACKEND_ID
        if temperature_from_session is not None: self._current_chat_temperature = temperature_from_session
        if generator_model_from_session and GENERATOR_BACKEND_ID in self._current_model_names:
            self._current_model_names[GENERATOR_BACKEND_ID] = generator_model_from_session

        if GENERATOR_BACKEND_ID in self._current_chat_personality_prompts and \
                self._current_chat_personality_prompts.get(GENERATOR_BACKEND_ID) is None:
            self._current_chat_personality_prompts[GENERATOR_BACKEND_ID] = CODER_AI_SYSTEM_PROMPT

        self._project_context_manager.load_state(proj_ctx_data)
        self._configure_all_initial_backends()
        self.set_current_project(active_pid_from_session)
        self._update_rag_initialized_state(emit_status=True, project_id=active_pid_from_session)
        self.update_status_based_on_state()

    @pyqtSlot()
    def _handle_sfm_active_history_cleared(self):
        if self._project_context_manager:
            active_project_id = self._project_context_manager.get_active_project_id()
            if active_project_id and (history := self._project_context_manager.get_project_history(active_project_id)):
                history.clear();
                self.history_changed.emit([])

    @pyqtSlot(str, str, dict, dict)
    def _handle_sfm_request_state_save(self, model_name: str, personality: Optional[str],
                                       all_project_data: Dict[str, Any], session_extra_data: Dict[str, Any]):
        if self._session_flow_manager:
            session_extra_data["generator_model_name"] = self._current_model_names.get(GENERATOR_BACKEND_ID)
            self._session_flow_manager.save_current_session_to_last_state(model_name, personality, session_extra_data)

    @pyqtSlot(str)
    def _handle_backend_stream_started(self, request_id: str):
        self.stream_started.emit(request_id)

    @pyqtSlot(str, str)
    def _handle_backend_chunk_received(self, request_id: str, chunk: str):
        if self._modification_coordinator and self._modification_coordinator.is_active():
            purpose = None
            if self._backend_coordinator:
                task_key = None
                for tk in self._backend_coordinator._active_backend_tasks.keys():
                    if tk[1] == request_id:
                        task_key = tk
                        break
                if task_key:
                    task = self._backend_coordinator._active_backend_tasks.get(task_key)
                    if task and hasattr(task, 'get_coro') and hasattr(task.get_coro(),
                                                                      'cr_frame') and task.get_coro().cr_frame:
                        f_locals = task.get_coro().cr_frame.f_locals
                        if 'request_metadata' in f_locals and isinstance(f_locals['request_metadata'], dict):
                            purpose = f_locals['request_metadata'].get("purpose")

            if purpose and purpose.startswith("mc_request_code_generation_"):
                return

        self.stream_chunk_received.emit(chunk)

    @pyqtSlot(str, ChatMessage, dict)
    def _handle_backend_response_completed(self, request_id: str, completed_message: ChatMessage,
                                           usage_stats_with_metadata: dict):
        purpose = usage_stats_with_metadata.get("purpose")

        if purpose and isinstance(purpose, str) and purpose.startswith("mc_request_code_generation_"):
            return

        backend_id_for_mc = usage_stats_with_metadata.get("backend_id_for_mc")
        if purpose and isinstance(purpose, str) and purpose.startswith("psc_"): return

        is_mc_planner_response = purpose and isinstance(purpose, str) and purpose.startswith(
            "mc_request_") and backend_id_for_mc == PLANNER_BACKEND_ID

        if self._modification_coordinator and self._modification_coordinator.is_active() and is_mc_planner_response:
            if self._modification_coordinator.is_awaiting_llm_response() and \
                    self._modification_coordinator._current_phase == ModPhase.AWAITING_PLAN_AND_ALL_CODER_INSTRUCTIONS:
                self._modification_coordinator.process_llm_response(backend_id_for_mc, completed_message)
                return
            else:
                return

        if purpose == "code_summary" and (
                original_target_filename := usage_stats_with_metadata.get("original_target_filename")):
            self.status_update.emit(f"Ava's summary for '{original_target_filename}' is ready!", "#98c379", True, 3000)
            summary_msg_text = f"✨ **Ava's Summary for {original_target_filename}:** ✨\n\n{completed_message.text}"
            summary_chat_message = ChatMessage(role=SYSTEM_ROLE, parts=[summary_msg_text],
                                               metadata={"is_ava_summary": True,
                                                         "target_file": original_target_filename, "is_internal": False})
            if self._project_context_manager: self._project_context_manager.add_message_to_active_project(
                summary_chat_message); self.new_message_added.emit(
                summary_chat_message); self._trigger_save_last_session_state()
            return

        message_updated_in_model = False
        if self._project_context_manager and (
                active_history := self._project_context_manager.get_active_conversation_history()):
            for msg_in_history in reversed(active_history):
                if msg_in_history.id == request_id and msg_in_history.role == MODEL_ROLE:
                    msg_in_history.parts = completed_message.parts
                    if completed_message.metadata:
                        if msg_in_history.metadata is None: msg_in_history.metadata = {}
                        msg_in_history.metadata.update(completed_message.metadata)
                    msg_in_history.loading_state = MessageLoadingState.COMPLETED
                    self.new_message_added.emit(msg_in_history);
                    message_updated_in_model = True;
                    break
        if not message_updated_in_model and self._project_context_manager:
            if completed_message.metadata is None: completed_message.metadata = {}
            completed_message.metadata["request_id"] = request_id;
            completed_message.loading_state = MessageLoadingState.COMPLETED
            self._project_context_manager.add_message_to_active_project(completed_message);
            self.new_message_added.emit(completed_message)

        if self._project_context_manager: self._trigger_save_last_session_state()
        self.stream_finished.emit()
        prompt_tokens, completion_tokens = usage_stats_with_metadata.get(
            "prompt_tokens"), usage_stats_with_metadata.get("completion_tokens")

        backend_id_for_tokens = backend_id_for_mc if backend_id_for_mc else (
            GENERATOR_BACKEND_ID if "technical_summary" in (purpose or "") else (
                DEFAULT_CHAT_BACKEND_ID if "friendly_summary" in (
                        purpose or "") else self._current_active_chat_backend_id))
        if prompt_tokens is not None and completion_tokens is not None and self._model_info_service:
            model_name_for_token_calc = self._current_model_names.get(backend_id_for_tokens, "")
            model_max_context = self._model_info_service.get_max_tokens(model_name_for_token_calc)
            self.token_usage_updated.emit(backend_id_for_tokens, prompt_tokens, completion_tokens, model_max_context)

    @pyqtSlot(str, str)
    def _handle_backend_response_error(self, request_id: str, error_message_str: str):
        purpose = None
        if self._backend_coordinator:
            task_key = None
            for tk in self._backend_coordinator._active_backend_tasks.keys():
                if tk[1] == request_id:
                    task_key = tk
                    break
            if task_key:
                task = self._backend_coordinator._active_backend_tasks.get(task_key)
                if task and hasattr(task, 'get_coro') and hasattr(task.get_coro(),
                                                                  'cr_frame') and task.get_coro().cr_frame:
                    f_locals = task.get_coro().cr_frame.f_locals
                    if 'request_metadata' in f_locals and isinstance(f_locals['request_metadata'], dict):
                        purpose = f_locals['request_metadata'].get("purpose")

        if purpose and isinstance(purpose, str) and purpose.startswith("mc_request_code_generation_"):
            return

        is_psc_related_error = False
        if self._project_summary_coordinator and ((hasattr(self._project_summary_coordinator,
                                                           '_current_request_id_tech_summary') and self._project_summary_coordinator._current_request_id_tech_summary == request_id) or (
                                                          hasattr(self._project_summary_coordinator,
                                                                  '_current_request_id_friendly_summary') and self._project_summary_coordinator._current_request_id_friendly_summary == request_id)):
            is_psc_related_error = True
        if is_psc_related_error: self.stream_finished.emit(); return

        if self._modification_coordinator and self._modification_coordinator.is_active() and \
                self._modification_coordinator._current_phase == ModPhase.AWAITING_PLAN_AND_ALL_CODER_INSTRUCTIONS and \
                request_id.startswith("mc_"):
            backend_id_for_mc_error = PLANNER_BACKEND_ID
            if hasattr(self._modification_coordinator, 'process_llm_error'):
                self._modification_coordinator.process_llm_error(backend_id_for_mc_error, error_message_str)
            self.stream_finished.emit()
            return

        message_updated_in_model = False
        if self._project_context_manager and (
                active_history := self._project_context_manager.get_active_conversation_history()):
            for msg_in_history in reversed(active_history):
                if msg_in_history.id == request_id and msg_in_history.role == MODEL_ROLE:
                    msg_in_history.role = ERROR_ROLE;
                    msg_in_history.parts = [f"Backend Error (Request ID: {request_id[:8]}...): {error_message_str}"];
                    msg_in_history.loading_state = MessageLoadingState.COMPLETED
                    self.new_message_added.emit(msg_in_history);
                    message_updated_in_model = True;
                    break
        if not message_updated_in_model and self._project_context_manager:
            err_obj = ChatMessage(id=request_id, role=ERROR_ROLE,
                                  parts=[f"Backend Error (Request ID: {request_id[:8]}...): {error_message_str}"],
                                  loading_state=MessageLoadingState.COMPLETED)
            self._project_context_manager.add_message_to_active_project(err_obj);
            self.new_message_added.emit(err_obj)

        if self._project_context_manager: self._trigger_save_last_session_state()
        self.stream_finished.emit();
        self.error_occurred.emit(f"Backend Error: {error_message_str}", False)

    @pyqtSlot(bool)
    def _handle_backend_busy_changed(self, backend_is_busy: bool):
        self._update_overall_busy_state()

    @pyqtSlot(str, str, bool, list)
    def _handle_backend_configuration_changed(self, backend_id: str, model_name: str, is_configured: bool,
                                              available_models: list):
        self._chat_backend_configured_successfully[backend_id] = is_configured
        self._available_models_per_backend[backend_id] = available_models[:]
        self._current_model_names[backend_id] = model_name
        if not is_configured and self._backend_coordinator:
            err = self._backend_coordinator.get_last_error_for_backend(backend_id) or f"{backend_id} config error."
            self.error_occurred.emit(f"Config Error ({backend_id} - {model_name}): {err}", False)
        self.available_models_changed_for_backend.emit(backend_id,
                                                       self._available_models_per_backend.get(backend_id, []))
        self.backend_config_state_changed.emit(backend_id, model_name, is_configured,
                                               bool(self._current_chat_personality_prompts.get(backend_id)))
        if backend_id == self._current_active_chat_backend_id:
            self.update_status_based_on_state()
        elif backend_id in [PLANNER_BACKEND_ID, GENERATOR_BACKEND_ID]:
            d_name = {PLANNER_BACKEND_ID: "Planner", GENERATOR_BACKEND_ID: "Specialized"}.get(backend_id, backend_id)
            status_msg = f"{d_name} ({backend_id}) OK with {model_name}." if is_configured else f"{d_name} ({backend_id}) not configured ({model_name})."
            self.status_update.emit(status_msg, "#98c379" if is_configured else "#e06c75", True,
                                    3000 if is_configured else 5000)
        self._trigger_save_last_session_state()

    @pyqtSlot(bool, str)
    def _handle_upload_started(self, is_global: bool, item_description: str):
        active_project_name_str = (
                    self._project_context_manager.get_active_project_name() or "Current") if self._project_context_manager else "N/A"
        context_name = constants.GLOBAL_CONTEXT_DISPLAY_NAME if is_global else active_project_name_str
        self.status_update.emit(f"Uploading {item_description} to '{context_name}' context...", "#61afef", False, 0)
        self._update_overall_busy_state()

    @pyqtSlot(ChatMessage)
    def _handle_upload_summary(self, summary_message: ChatMessage):
        if not self._project_context_manager: return
        self._project_context_manager.add_message_to_active_project(summary_message);
        self.new_message_added.emit(summary_message)
        s_cid = summary_message.metadata.get("collection_id") if summary_message.metadata else None
        self._update_rag_initialized_state(emit_status=True, project_id=s_cid)
        self.update_status_based_on_state();
        self._trigger_save_last_session_state()

    @pyqtSlot(str)
    def _handle_upload_error(self, error_message_str: str):
        if not self._project_context_manager: return
        err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Upload System Error: {error_message_str}"])
        self._project_context_manager.add_message_to_active_project(err_obj);
        self.new_message_added.emit(err_obj)
        self.error_occurred.emit(f"Upload Error: {error_message_str}", False);
        self.update_status_based_on_state()

    @pyqtSlot(bool)
    def _handle_upload_busy_changed(self, upload_is_busy: bool):
        self._update_overall_busy_state()

    @pyqtSlot(list)
    def _handle_uih_normal_chat_request(self, new_user_message_list: List[ChatMessage]):
        if not (self._backend_coordinator and self._project_context_manager): self.error_occurred.emit(
            "Cannot send chat: Critical components missing.", True); return
        if not new_user_message_list or not isinstance(new_user_message_list[0], ChatMessage): return
        user_message_for_ui = new_user_message_list[0]
        self._project_context_manager.add_message_to_active_project(user_message_for_ui);
        self.new_message_added.emit(user_message_for_ui)
        self._trigger_save_last_session_state();
        QApplication.processEvents()
        ai_request_id = str(uuid.uuid4())
        ai_placeholder_message = ChatMessage(id=ai_request_id, role=MODEL_ROLE, parts=[""],
                                             loading_state=MessageLoadingState.LOADING)
        self._project_context_manager.add_message_to_active_project(ai_placeholder_message);
        self.new_message_added.emit(ai_placeholder_message)
        full_history_for_backend = self._project_context_manager.get_active_conversation_history()
        if not full_history_for_backend: self.error_occurred.emit("Internal error preparing chat.", True); return
        self._backend_coordinator.request_response_stream(
            target_backend_id=self._current_active_chat_backend_id, request_id=ai_request_id,
            history_to_send=full_history_for_backend[:-1], is_modification_response_expected=False,
            options={"temperature": self._current_chat_temperature},
            request_metadata={"original_user_message_id": user_message_for_ui.id})

    @pyqtSlot(str, list, str, str)
    def _handle_uih_mod_start_request(self, original_query_text: str, image_data_list: List[Dict[str, Any]],
                                      context_for_mc: str, focus_prefix_for_mc: str):
        if not (self._modification_coordinator and self._project_context_manager): self.error_occurred.emit(
            "Modification feature unavailable or PCM missing.", True); return
        user_chat_message_for_ui = ChatMessage(role=USER_ROLE, parts=[original_query_text] + (image_data_list or []))
        self._project_context_manager.add_message_to_active_project(user_chat_message_for_ui);
        self.new_message_added.emit(user_chat_message_for_ui)
        self._trigger_save_last_session_state();
        QApplication.processEvents()
        if self._modification_handler_instance: self._modification_handler_instance.activate_sequence()
        self._modification_coordinator.start_sequence(query=original_query_text, context_from_rag=context_for_mc,
                                                      focus_prefix=focus_prefix_for_mc)

    @pyqtSlot(str, str)
    def _handle_uih_mod_user_input(self, user_command: str, action_type: str):
        if self._modification_coordinator:
            self._modification_coordinator.process_user_input(user_command)
        else:
            self.error_occurred.emit("Modification feature unavailable.", False)

    @pyqtSlot(str)
    def _handle_uih_processing_error(self, error_message: str):
        self.error_occurred.emit(f"Input Processing Error: {error_message}", False)
        if self._project_context_manager:
            err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Input Error: {error_message}"])
            self._project_context_manager.add_message_to_active_project(err_obj);
            self.new_message_added.emit(err_obj)

    @pyqtSlot(ChatMessage)
    def _handle_user_command_for_display_only(self, user_message: ChatMessage):
        if not self._project_context_manager: return
        self._project_context_manager.add_message_to_active_project(user_message);
        self.new_message_added.emit(user_message)
        self._trigger_save_last_session_state()

    @pyqtSlot(str, list)
    def _handle_mc_request_llm_call(self, target_backend_id: str, history_to_send: List[ChatMessage]):
        if self._backend_coordinator:
            mc_options = {}
            if target_backend_id == GENERATOR_BACKEND_ID:
                mc_options = {"temperature": 0.2}
            elif target_backend_id == PLANNER_BACKEND_ID:
                mc_options = {"temperature": 0.5}

            mc_internal_request_id = f"mc_planner_initial_{str(uuid.uuid4())[:8]}"
            request_metadata_for_mc = {"purpose": f"mc_request_{target_backend_id}",
                                       "mc_internal_id": mc_internal_request_id, "backend_id_for_mc": target_backend_id}

            self._backend_coordinator.request_response_stream(
                target_backend_id=target_backend_id,
                request_id=mc_internal_request_id,
                history_to_send=history_to_send,
                is_modification_response_expected=True,
                options=mc_options,
                request_metadata=request_metadata_for_mc
            )
        elif self._modification_coordinator:
            self._modification_coordinator.process_llm_error(target_backend_id, "BackendCoordinator unavailable.")

    @pyqtSlot(str, str, str)
    def _handle_code_generated_and_summary_needed(self, generated_code: str, coder_instructions: str,
                                                  target_filename: str):
        if not (self._code_summary_service and self._backend_coordinator): self.error_occurred.emit(
            f"Internal error: Services unavailable for summary of '{target_filename}'.", True); return
        self.status_update.emit(f"Ava is preparing summary for '{target_filename}'...", "#e5c07b", True, 4000)
        if not self._code_summary_service.request_code_summary(self._backend_coordinator, target_filename,
                                                               coder_instructions, generated_code):
            if self._project_context_manager:
                sys_err_msg = ChatMessage(role=ERROR_ROLE,
                                          parts=[f"[System: Error initiating summary for '{target_filename}'.]"]);
                self._project_context_manager.add_message_to_active_project(sys_err_msg);
                self.new_message_added.emit(sys_err_msg)
            self.update_status_based_on_state()

    @pyqtSlot(str, str)
    def _handle_mc_file_ready(self, filename: str, content: str):
        self.code_file_updated.emit(filename, content)
        if self._project_context_manager:
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[f"[System: File '{filename}' updated. See Code Viewer.]"],
                                  metadata={"is_internal": False});
            self._project_context_manager.add_message_to_active_project(sys_msg);
            self.new_message_added.emit(sys_msg)

    @pyqtSlot(str, str)
    def _handle_mc_sequence_complete(self, reason: str, original_query_summary: str):
        if self._project_context_manager:
            strengthened_system_message_text = (
                f"[System: Task '{original_query_summary}' (multi-file code modification) is now FULLY COMPLETED and CLOSED. "
                f"Reason for completion: {reason}. All generated code is in the Code Viewer. "
                f"IMPORTANT: The user is now expected to start a NEW, unrelated conversation or ask a new question. "
                f"Ava, your role is to transition to a fresh, open-ended conversational state. "
                f"Acknowledge completion if the user mentions it, then wait for their NEXT independent query. "
                f"DO NOT re-describe, re-summarize, or offer to continue the '{original_query_summary}' task. "
                f"Consider the previous task context as concluded.]"
            )
            internal_sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[strengthened_system_message_text],
                                           metadata={"is_internal": True})
            self._project_context_manager.add_message_to_active_project(internal_sys_msg)

            if reason.startswith("completed"):
                priming_model_message_text = ""
                if reason == "completed_no_files_in_plan":
                    priming_model_message_text = f"Okay, it looks like no changes were actually needed for '{original_query_summary}'. What can I help you with now? ✨"
                elif reason == "completed_by_user_acceptance":
                    priming_model_message_text = f"Great! All changes for '{original_query_summary}' are accepted. What's next on your list? 🚀"
                elif reason == "completed_no_files":
                    priming_model_message_text = f"Okay, it looks like no changes were actually needed for '{original_query_summary}'. What can I help you with now? ✨"
                elif reason == "completed":
                    priming_model_message_text = f"Alright, all done with the '{original_query_summary}' task! What's next on your list? 🚀"
                else:
                    priming_model_message_text = f"The task '{original_query_summary}' has finished ({reason}). What shall we do next?"
                if priming_model_message_text:
                    priming_model_msg = ChatMessage(role=MODEL_ROLE, parts=[priming_model_message_text])
                    self._project_context_manager.add_message_to_active_project(priming_model_msg)
                    self.new_message_added.emit(priming_model_msg)
            self._trigger_save_last_session_state()
        self.update_status_based_on_state()
        if self._modification_handler_instance: self._modification_handler_instance.cancel_modification()

    @pyqtSlot(str)
    def _handle_mc_error(self, error_message: str):
        if self._project_context_manager:
            err_msg_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Modification System Error: {error_message}"],
                                      metadata={"is_internal": False});
            self._project_context_manager.add_message_to_active_project(err_msg_obj);
            self.new_message_added.emit(err_msg_obj);
            self._trigger_save_last_session_state()
        self.error_occurred.emit(f"Modification Error: {error_message}", False);
        self.update_status_based_on_state()

    @pyqtSlot(str)
    def _handle_mc_status_update(self, message: str):
        if self._project_context_manager:
            status_msg = ChatMessage(role=SYSTEM_ROLE, parts=[message], metadata={"is_internal": False});
            self._project_context_manager.add_message_to_active_project(status_msg);
            self.new_message_added.emit(status_msg)

    @pyqtSlot(str, str)
    def _handle_project_summary_generated(self, project_id: str, friendly_summary_text: str):
        if not self._project_context_manager: return
        project_name = self._project_context_manager.get_project_name(project_id) or project_id
        summary_chat_message = ChatMessage(role=MODEL_ROLE, parts=[
            f"✨ **Ava's Project Insights for '{project_name}'!** ✨\n\n{friendly_summary_text}"],
                                           metadata={"is_project_summary": True, "project_id": project_id,
                                                     "is_internal": False})
        if (target_history := self._project_context_manager.get_project_history(project_id)) is not None:
            target_history.append(summary_chat_message)
            if self._project_context_manager.get_active_project_id() == project_id:
                self.new_message_added.emit(summary_chat_message)
            else:
                self.status_update.emit(f"Project summary for project '{project_name}' is ready!", "#98c379", True,
                                        5000)
        self._trigger_save_last_session_state();
        self.update_status_based_on_state()

    @pyqtSlot(str, str)
    def _handle_project_summary_failed(self, project_id: str, error_message: str):
        if not self._project_context_manager: return
        project_name = self._project_context_manager.get_project_name(project_id) or project_id
        error_chat_message = ChatMessage(role=ERROR_ROLE, parts=[
            f"[System Error: Could not generate summary for project '{project_name}'. Reason: {error_message}]"],
                                         metadata={"is_project_summary_error": True, "project_id": project_id,
                                                   "is_internal": False})
        if (target_history := self._project_context_manager.get_project_history(project_id)) is not None:
            target_history.append(error_chat_message)
            if self._project_context_manager.get_active_project_id() == project_id: self.new_message_added.emit(
                error_chat_message)
        self.error_occurred.emit(f"Summary failed for '{project_name}': {error_message}", False);
        self._trigger_save_last_session_state();
        self.update_status_based_on_state()

    def _cancel_active_tasks(self):
        if self._backend_coordinator: self._backend_coordinator.cancel_current_task()
        if self._upload_coordinator: self._upload_coordinator.cancel_current_upload()
        if self._modification_coordinator and self._modification_coordinator.is_active(): self._modification_coordinator.cancel_sequence(
            reason="user_cancel_all")

    def cleanup(self):
        self._cancel_active_tasks(); self._trigger_save_last_session_state()

    def _update_rag_initialized_state(self, emit_status: bool = True, project_id: Optional[str] = None):
        if not self._project_context_manager: return
        target_pid = project_id or (self._project_context_manager.get_active_project_id())
        new_init_state = self.is_rag_context_initialized(target_pid)
        if target_pid == self._project_context_manager.get_active_project_id() and self._rag_initialized != new_init_state: self._rag_initialized = new_init_state
        if emit_status or (
                target_pid == self._project_context_manager.get_active_project_id() and self._rag_initialized != new_init_state): self.update_status_based_on_state()

    def is_rag_context_initialized(self, project_id: Optional[str]) -> bool:
        if not (self._vector_db_service and project_id): self._rag_available = False; return False
        self._rag_available = True
        return (self._vector_db_service.is_ready(project_id) and self._vector_db_service.get_collection_size(
            project_id) > 0)

    def get_project_history(self, project_id: str) -> List[ChatMessage]:
        return list(self._project_context_manager.get_project_history(
            project_id) or []) if self._project_context_manager else []

    def get_current_history(self) -> List[ChatMessage]:
        return list(
            self._project_context_manager.get_active_conversation_history() or []) if self._project_context_manager else []

    def get_current_project_id(self) -> Optional[str]:
        return self._project_context_manager.get_active_project_id() if self._project_context_manager else None

    def is_overall_busy(self) -> bool:
        return self._overall_busy

    def is_rag_available(self) -> bool:
        return self._rag_available

    def get_rag_contents(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not (self._project_context_manager and self._vector_db_service): return []
        target_id = collection_id or (self._project_context_manager.get_active_project_id())
        if not target_id or not self._vector_db_service.is_ready(target_id): return []
        try:
            return self._vector_db_service.get_all_metadata(target_id)
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

    def get_project_summary_coordinator(self) -> Optional[ProjectSummaryCoordinator]:
        return self._project_summary_coordinator

    def _trigger_save_last_session_state(self):
        if self._session_flow_manager:
            active_chat_backend_id = self._current_active_chat_backend_id
            session_extra_data = {"active_chat_backend_id": active_chat_backend_id,
                                  "chat_temperature": self._current_chat_temperature,
                                  "generator_model_name": self._current_model_names.get(GENERATOR_BACKEND_ID)}
            self._session_flow_manager.save_current_session_to_last_state(
                self._current_model_names.get(active_chat_backend_id),
                self._current_chat_personality_prompts.get(active_chat_backend_id),
                {k: v for k, v in session_extra_data.items() if v is not None})

    def get_current_chat_model(self) -> str:
        return self._current_model_names.get(self._current_active_chat_backend_id, "Unknown Model")

    def get_current_chat_personality(self) -> Optional[str]:
        return self._current_chat_personality_prompts.get(self._current_active_chat_backend_id)

    def set_active_chat_backend(self, backend_id: str):
        if backend_id not in self._current_model_names and not any(
            detail["id"] == backend_id for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS): self.error_occurred.emit(
            f"Invalid chat backend type selected: {backend_id}", False); return
        if self._current_active_chat_backend_id != backend_id:
            self._current_active_chat_backend_id = backend_id
            api_key_to_use = get_gemini_api_key() if backend_id.startswith("gemini") else (
                get_openai_api_key() if backend_id.startswith("gpt") else None)
            if self._backend_coordinator: self._backend_coordinator.configure_backend(backend_id, api_key_to_use,
                                                                                      self._current_model_names.get(
                                                                                          backend_id, ""),
                                                                                      self._current_chat_personality_prompts.get(
                                                                                          backend_id))
            self.update_status_based_on_state();
            self._trigger_save_last_session_state()

    def set_model_for_backend(self, backend_id: str, model_name: str):
        if backend_id not in self._current_model_names: self.error_occurred.emit(
            f"Cannot set model for invalid backend: {backend_id}", False); return
        if not model_name: self.error_occurred.emit(f"Model name cannot be empty for backend: {backend_id}",
                                                    False); return
        self._current_model_names[backend_id] = model_name
        api_key_to_use = get_gemini_api_key() if backend_id.startswith("gemini") else (
            get_openai_api_key() if backend_id.startswith("gpt") else None)
        personality_to_use = self._current_chat_personality_prompts.get(backend_id)
        if backend_id == GENERATOR_BACKEND_ID and personality_to_use is None:
            personality_to_use = CODER_AI_SYSTEM_PROMPT
            self._current_chat_personality_prompts[GENERATOR_BACKEND_ID] = personality_to_use

        if self._backend_coordinator: self._backend_coordinator.configure_backend(backend_id, api_key_to_use,
                                                                                  model_name, personality_to_use)
        if backend_id == self._current_active_chat_backend_id: self.update_status_based_on_state()
        self._trigger_save_last_session_state()

    def set_personality_for_backend(self, backend_id: str, prompt: Optional[str]):
        if backend_id not in self._current_chat_personality_prompts: self.error_occurred.emit(
            f"Cannot set personality for invalid backend: {backend_id}", False); return

        effective_prompt = prompt.strip() if prompt else None
        if backend_id == GENERATOR_BACKEND_ID and not effective_prompt:
            effective_prompt = CODER_AI_SYSTEM_PROMPT

        self._current_chat_personality_prompts[backend_id] = effective_prompt
        api_key_to_use = get_gemini_api_key() if backend_id.startswith("gemini") else (
            get_openai_api_key() if backend_id.startswith("gpt") else None)
        if self._backend_coordinator: self._backend_coordinator.configure_backend(backend_id, api_key_to_use,
                                                                                  self._current_model_names.get(
                                                                                      backend_id, ""),
                                                                                  self._current_chat_personality_prompts[
                                                                                      backend_id])
        if backend_id == self._current_active_chat_backend_id: self.update_status_based_on_state()
        self._trigger_save_last_session_state()

    def set_current_project(self, project_id: str):
        if self._project_context_manager and not self._project_context_manager.set_active_project(
            project_id): self.error_occurred.emit(f"Failed to set project '{project_id}'.", False)

    def create_project_collection(self, project_name: str):
        if self._project_context_manager:
            if not self._project_context_manager.create_project(project_name):
                self.error_occurred.emit(f"Failed to create project '{project_name}'.", False)
            else:
                self.status_update.emit(f"Project '{project_name}' created.", "#98c379", True, 3000)

    def start_new_chat(self):
        if self._session_flow_manager:
            session_extra_data = {"active_chat_backend_id": self._current_active_chat_backend_id,
                                  "chat_temperature": self._current_chat_temperature,
                                  "generator_model_name": self._current_model_names.get(GENERATOR_BACKEND_ID)}
            self._session_flow_manager.start_new_chat_session(
                self._current_model_names.get(self._current_active_chat_backend_id),
                self._current_chat_personality_prompts.get(self._current_active_chat_backend_id),
                {k: v for k, v in session_extra_data.items() if v is not None})

    def load_chat_session(self, filepath: str):
        if self._session_flow_manager: self._session_flow_manager.load_named_session(filepath,
                                                                                     self._current_active_chat_backend_id)

    def save_current_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager:
            session_extra_data = {"active_chat_backend_id": self._current_active_chat_backend_id,
                                  "chat_temperature": self._current_chat_temperature,
                                  "generator_model_name": self._current_model_names.get(GENERATOR_BACKEND_ID)}
            return self._session_flow_manager.save_session_as(filepath, self._current_model_names.get(
                self._current_active_chat_backend_id), self._current_chat_personality_prompts.get(
                self._current_active_chat_backend_id), {k: v for k, v in session_extra_data.items() if v is not None})
        return False

    def delete_chat_session(self, filepath: str) -> bool:
        return self._session_flow_manager.delete_named_session(filepath) if self._session_flow_manager else False

    def list_saved_sessions(self) -> List[str]:
        return self._session_flow_manager.list_saved_sessions() if self._session_flow_manager else []

    def process_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        if self._user_input_handler: self._user_input_handler.handle_user_message(text=text, image_data=image_data,
                                                                                  focus_paths=self._current_chat_focus_paths,
                                                                                  rag_available=self._rag_available,
                                                                                  rag_initialized_for_project=self.is_rag_context_initialized(
                                                                                      self.get_current_project_id()))

    def update_status_based_on_state(self):
        active_backend_display_name = next((detail["name"] for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS if
                                            detail['id'] == self._current_active_chat_backend_id),
                                           self._current_active_chat_backend_id)
        if not self._chat_backend_configured_successfully.get(self._current_active_chat_backend_id, False):
            err_msg = f"API Not Configured ({active_backend_display_name})"
            if self._backend_coordinator and (err := self._backend_coordinator.get_last_error_for_backend(
                self._current_active_chat_backend_id)): err_msg = f"API Error ({active_backend_display_name}): {err}"
            self.status_update.emit(f"{err_msg}. Check settings.", "#e06c75", False, 0)
        elif self._overall_busy:
            self.status_update.emit(f"Processing with {active_backend_display_name}...", "#e5c07b", False, 0)
        else:
            parts = [f"Ready ({active_backend_display_name})"]
            if self._project_context_manager and (pid := self._project_context_manager.get_active_project_id()) and (
            pname := self._project_context_manager.get_project_name(pid) or "Unknown"):
                parts.append(
                    f"(Ctx: {constants.GLOBAL_CONTEXT_DISPLAY_NAME if pid == constants.GLOBAL_COLLECTION_ID else pname})")
            if self.is_rag_context_initialized(
                self._project_context_manager.get_active_project_id() if self._project_context_manager else None): parts.append(
                "[RAG Active]")
            self.status_update.emit(" ".join(parts), "#98c379", False, 0)

    def set_chat_temperature(self, temperature: float):
        if 0.0 <= temperature <= 2.0:
            self._current_chat_temperature = temperature
            active_backend_display_name = next((detail["name"] for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS if
                                                detail['id'] == self._current_active_chat_backend_id),
                                               self._current_active_chat_backend_id)
            self.status_update.emit(
                f"Temperature for '{active_backend_display_name}' set to {self._current_chat_temperature:.2f}",
                "#61afef", True, 3000);
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
        display_paths = [os.path.basename(p) for p in paths];
        display = ", ".join(display_paths[:3]) + (
            f", ... ({len(display_paths) - 3} more)" if len(display_paths) > 3 else "")
        self.status_update.emit(f"Focus set on: {display}", "#61afef", True, 4000)

    def _update_overall_busy_state(self):
        new_busy = (self._backend_coordinator.is_processing_request() if self._backend_coordinator else False) or (
            self._upload_coordinator.is_busy() if self._upload_coordinator else False)
        if self._overall_busy != new_busy: self._overall_busy = new_busy; self.busy_state_changed.emit(
            self._overall_busy); self.update_status_based_on_state()

    def is_api_ready(self) -> bool:
        return self._chat_backend_configured_successfully.get(self._current_active_chat_backend_id, False)