# === core/chat_manager.py (Part 1/3) ===
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
    class ChatMessage:  # type: ignore
        def __init__(self, role="", parts=None, timestamp=None, metadata=None, id="", loading_state=None): pass

    from enum import Enum, auto

    class MessageLoadingState(Enum):
        IDLE = auto(); LOADING = auto(); COMPLETED = auto(); ERROR = auto()  # type: ignore
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"  # type: ignore
    logging.error("ChatManager: Failed to import ChatMessage or MessageLoadingState from core.")

from backend.interface import BackendInterface
from services.code_summary_service import CodeSummaryService
from services.model_info_service import ModelInfoService
from services.session_service import SessionService
from services.upload_service import UploadService
from services.vector_db_service import VectorDBService
from .project_context_manager import ProjectContextManager
from .backend_coordinator import BackendCoordinator
from .session_flow_manager import SessionFlowManager
from .upload_coordinator import UploadCoordinator
from .rag_handler import RagHandler
from .user_input_handler import UserInputHandler
from .user_input_processor import UserInputProcessor
from .application_orchestrator import ApplicationOrchestrator

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

try:
    from .project_summary_coordinator import ProjectSummaryCoordinator
    PROJECT_SUMMARY_COORDINATOR_AVAILABLE = True
except ImportError:
    ProjectSummaryCoordinator = None  # type: ignore
    PROJECT_SUMMARY_COORDINATOR_AVAILABLE = False
    logging.warning("ChatManager: ProjectSummaryCoordinator not available or not imported.")

from utils import constants
from utils.constants import (
    DEFAULT_CHAT_BACKEND_ID,
    OLLAMA_CHAT_BACKEND_ID,
    GPT_CHAT_BACKEND_ID,
    PLANNER_BACKEND_ID,
    GENERATOR_BACKEND_ID,
    DEFAULT_GEMINI_CHAT_MODEL,
    DEFAULT_OLLAMA_MODEL, # This might represent the default *generator* model if Ollama is the generator
    DEFAULT_GPT_MODEL,
    DEFAULT_GEMINI_PLANNER_MODEL
)

try:
    from config import APP_CONFIG

    def get_gemini_api_key():
        return APP_CONFIG.get("GEMINI_API_KEY")

    def get_openai_api_key(): # Ensure this exists or use os.getenv("OPENAI_API_KEY")
        return os.getenv("OPENAI_API_KEY") # Changed to getenv as it's not in your config.py example
except ImportError:
    def get_gemini_api_key():
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # type: ignore

    def get_openai_api_key():
        return os.getenv("OPENAI_API_KEY")  # type: ignore
    logging.warning(
        "ChatManager: config.py's APP_CONFIG or API key getters not found. Using environment variables as fallback.")

logger = logging.getLogger(__name__)

# This defines which backends are user-selectable *for chat*
USER_SELECTABLE_CHAT_BACKEND_DETAILS = [
    {"id": DEFAULT_CHAT_BACKEND_ID, "name": "Google Gemini"},
    {"id": OLLAMA_CHAT_BACKEND_ID, "name": "Ollama (Local Chat)"}, # Clarified name
    {"id": GPT_CHAT_BACKEND_ID, "name": "OpenAI GPT"}
]

# Define details for the specialized/generator backend (currently fixed to Ollama)
# This could be expanded if we support other generator types later.
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
        logger.info("ChatManager initializing with ApplicationOrchestrator...")
        if not isinstance(orchestrator, ApplicationOrchestrator):
            err_msg = "ChatManager requires a valid ApplicationOrchestrator instance."
            logger.critical(err_msg);
            raise TypeError(err_msg)

        self._orchestrator = orchestrator
        self._backend_adapters_dict = self._orchestrator.get_all_backend_adapters_dict()
        logger.debug(f"ChatManager received adapter dict keys: {list(self._backend_adapters_dict.keys())}")

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
        logger.info("ChatManager core initialization using orchestrator complete.")

    def _initialize_state_variables(self):
        self._overall_busy: bool = False
        self._current_active_chat_backend_id: str = DEFAULT_CHAT_BACKEND_ID # For the main chat

        all_backend_ids_from_adapters = sorted(
            list(self._backend_adapters_dict.keys()))
        logger.debug(
            f"ChatManager _initialize_state_variables: Initializing state for adapter IDs: {all_backend_ids_from_adapters}")

        # This dictionary stores the currently selected model for *each* backend ID
        # including DEFAULT_CHAT_BACKEND_ID, OLLAMA_CHAT_BACKEND_ID, GPT_CHAT_BACKEND_ID,
        # PLANNER_BACKEND_ID, and crucially GENERATOR_BACKEND_ID.
        self._current_model_names: Dict[str, str] = {
            bid: "" for bid in all_backend_ids_from_adapters
        }
        # Set defaults for Chat-related backends
        if DEFAULT_CHAT_BACKEND_ID in self._current_model_names:
            self._current_model_names[DEFAULT_CHAT_BACKEND_ID] = DEFAULT_GEMINI_CHAT_MODEL
        if OLLAMA_CHAT_BACKEND_ID in self._current_model_names:
            # This default is for if Ollama is chosen for *chat*.
            # If you don't have chat models in Ollama, this won't be used often for chat.
            self._current_model_names[OLLAMA_CHAT_BACKEND_ID] = "llama3:latest" # Example chat model
        if GPT_CHAT_BACKEND_ID in self._current_model_names:
            self._current_model_names[GPT_CHAT_BACKEND_ID] = DEFAULT_GPT_MODEL

        # Set defaults for specific purpose backends
        if PLANNER_BACKEND_ID in self._current_model_names:
            self._current_model_names[PLANNER_BACKEND_ID] = DEFAULT_GEMINI_PLANNER_MODEL
        if GENERATOR_BACKEND_ID in self._current_model_names:
            # This is the default model for the *Specialized LLM* (Generator)
            self._current_model_names[GENERATOR_BACKEND_ID] = DEFAULT_OLLAMA_MODEL # e.g., "codellama:13b"

        self._current_chat_personality_prompts: Dict[str, Optional[str]] = {
            bid: None for bid in all_backend_ids_from_adapters
        }
        if PLANNER_BACKEND_ID in self._current_chat_personality_prompts:
            self._current_chat_personality_prompts[
                PLANNER_BACKEND_ID] = "You are an expert planner and technical writer."
        # Personality for GENERATOR_BACKEND_ID is often embedded in the prompt by ModificationHandler/Coordinator
        # But we can set a base one here if desired.
        if GENERATOR_BACKEND_ID in self._current_chat_personality_prompts:
            self._current_chat_personality_prompts[
                GENERATOR_BACKEND_ID] = "You are an expert Python coding assistant. Your responses must be only code, in a single markdown code block."

        self._current_chat_temperature: float = 0.7
        self._chat_backend_configured_successfully: Dict[str, bool] = {
            bid: False for bid in all_backend_ids_from_adapters
        }
        self._available_models_per_backend: Dict[str, List[str]] = {
            bid: [] for bid in all_backend_ids_from_adapters
        }
        self._current_chat_focus_paths: Optional[List[str]] = None
        self._rag_available: bool = (self._vector_db_service is not None and
                                     hasattr(self._vector_db_service, 'is_ready') and
                                     self._vector_db_service.is_ready())
        self._rag_initialized: bool = self._rag_available
        logger.debug(f"ChatManager state variables initialized. Model names dict: {self._current_model_names}")

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
            self._modification_coordinator.codeGeneratedAndSummaryNeeded.connect(
                self._handle_code_generated_and_summary_needed)
        if self._project_summary_coordinator:
            self._project_summary_coordinator.summary_generated.connect(self._handle_project_summary_generated)
            self._project_summary_coordinator.summary_generation_failed.connect(self._handle_project_summary_failed)
            logger.info("ChatManager connected to ProjectSummaryCoordinator output signals.")
        logger.debug("ChatManager component signal connection process finished.")
# === core/chat_manager.py (Part 2/3) ===
    def initialize(self):
        logger.info("ChatManager late initialization process starting...")
        if not (self._session_flow_manager and self._project_context_manager and
                self._user_input_handler and self._backend_coordinator):
            missing_deps = [name for comp, name in [
                (self._session_flow_manager, "SFM"), (self._project_context_manager, "PCM"),
                (self._user_input_handler, "UIH"), (self._backend_coordinator, "BC")] if not comp]
            logger.critical(f"Cannot initialize ChatManager: Critical components missing: {missing_deps}")
            self.error_occurred.emit(f"Critical error during init ({', '.join(missing_deps)} missing).", True)
            return

        # session_extra_data from SessionService now also includes 'generator_model_name'
        loaded_state = self._session_flow_manager.load_last_session_state_on_startup()
        model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session = None, None, None, constants.GLOBAL_COLLECTION_ID
        active_backend_id_from_session = None
        temperature_from_session = None
        generator_model_from_session = None # New state variable

        if loaded_state and len(loaded_state) == 7: # Expecting 7 items now
            model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session, \
                active_backend_id_from_session, temperature_from_session, generator_model_from_session = loaded_state
            logger.info(
                f"CM Init: Loaded full session state (7 items). ActivePID: {active_pid_from_session}, "
                f"ActiveChatBackend: {active_backend_id_from_session}, GeneratorModel: {generator_model_from_session}")
        elif loaded_state and len(loaded_state) == 6: # Backward compatibility for 6 items
            model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session, \
                active_backend_id_from_session, temperature_from_session = loaded_state
            logger.warning(
                "CM Init: Older session format detected (6 items). Generator model will use default.")
        elif loaded_state and len(loaded_state) >= 4: # Backward compatibility for 4 items
            model_from_session, pers_from_session, proj_data_from_session, active_pid_from_session = loaded_state[:4]
            logger.warning(
                "CM Init: Very old session format detected (4 items). Active backend, temperature, and generator model will use defaults.")
        else:
            logger.error(
                "CM Init: SessionFlowManager.load_last_session_state_on_startup returned invalid data. Using defaults.")
            active_pid_from_session = constants.GLOBAL_COLLECTION_ID

        # Set Chat AI properties
        if active_backend_id_from_session and active_backend_id_from_session in self._current_model_names:
            self._current_active_chat_backend_id = active_backend_id_from_session
            logger.info(f"Loaded active chat backend ID from session: {self._current_active_chat_backend_id}")
            if model_from_session: self._current_model_names[
                self._current_active_chat_backend_id] = model_from_session
            if pers_from_session: self._current_chat_personality_prompts[
                self._current_active_chat_backend_id] = pers_from_session
        elif model_from_session: # Fallback for older sessions if only model_name was present
            self._current_model_names[DEFAULT_CHAT_BACKEND_ID] = model_from_session
            if pers_from_session: self._current_chat_personality_prompts[DEFAULT_CHAT_BACKEND_ID] = pers_from_session

        if temperature_from_session is not None:
            self._current_chat_temperature = temperature_from_session
            logger.info(f"Loaded temperature from session: {self._current_chat_temperature}")

        # Set Generator AI model (Specialized LLM)
        if generator_model_from_session and GENERATOR_BACKEND_ID in self._current_model_names:
            self._current_model_names[GENERATOR_BACKEND_ID] = generator_model_from_session
            logger.info(f"Loaded generator model from session: {generator_model_from_session} for backend {GENERATOR_BACKEND_ID}")
        else:
            logger.info(f"No generator model in session or backend not found, using default: {self._current_model_names.get(GENERATOR_BACKEND_ID)}")


        if proj_data_from_session:
            self._project_context_manager.load_state(proj_data_from_session)
        else:
            self._project_context_manager.set_active_project(constants.GLOBAL_COLLECTION_ID)

        self._perform_orphan_cleanup(self._project_context_manager.save_state())
        self._set_initial_active_project(active_pid_from_session, None)

        self._configure_all_initial_backends() # This will now use the loaded generator model too

        self.update_status_based_on_state()
        current_active_project_id = self._project_context_manager.get_active_project_id()
        self._update_rag_initialized_state(emit_status=False, project_id=current_active_project_id)

        logger.info(
            f"ChatManager late init complete. Active project: {current_active_project_id}. "
            f"Active Chat Backend ID: {self._current_active_chat_backend_id}. "
            f"Model for active chat backend: {self._current_model_names.get(self._current_active_chat_backend_id, 'N/A')}. "
            f"Model for generator backend ({GENERATOR_BACKEND_ID}): {self._current_model_names.get(GENERATOR_BACKEND_ID, 'N/A')}"
        )

    def _perform_orphan_cleanup(self, project_context_data_from_pcm: Optional[Dict[str, Any]]):
        if not (self._project_context_manager and self._vector_db_service): return
        logger.debug("Orphan cleanup check (currently simplified).")

    def _set_initial_active_project(self, target_active_project_id: Optional[str], _): # _ for legacy arg
        if not self._project_context_manager: return
        effective_target_id = target_active_project_id if target_active_project_id else constants.GLOBAL_COLLECTION_ID
        if not self._project_context_manager.get_project_history(effective_target_id):
            logger.warning(f"Initial target project '{effective_target_id}' not found in PCM, defaulting to Global.")
            effective_target_id = constants.GLOBAL_COLLECTION_ID
        self._project_context_manager.set_active_project(effective_target_id)

    def _configure_all_initial_backends(self):
        if not self._backend_coordinator: return
        logger.info("ChatManager: Configuring all initial backends...")
        gemini_api_key = get_gemini_api_key()
        openai_api_key = get_openai_api_key()

        for backend_id in self._backend_adapters_dict.keys():
            model_to_use = self._current_model_names.get(backend_id)

            if not model_to_use: # Fallback if model name is empty for a known adapter
                if backend_id == DEFAULT_CHAT_BACKEND_ID: model_to_use = DEFAULT_GEMINI_CHAT_MODEL
                elif backend_id == OLLAMA_CHAT_BACKEND_ID: model_to_use = "llama3:latest" # Example chat default
                elif backend_id == GPT_CHAT_BACKEND_ID: model_to_use = DEFAULT_GPT_MODEL
                elif backend_id == PLANNER_BACKEND_ID: model_to_use = DEFAULT_GEMINI_PLANNER_MODEL
                elif backend_id == GENERATOR_BACKEND_ID: model_to_use = DEFAULT_OLLAMA_MODEL # Specialist default
                else:
                    logger.warning(f"No default model defined for existing backend_id '{backend_id}'. Configuration might be incomplete.")
                    model_to_use = "default_model_placeholder"
                self._current_model_names[backend_id] = model_to_use

            personality_to_use = self._current_chat_personality_prompts.get(backend_id)
            api_key_for_this_backend = None
            if backend_id.startswith("gemini"): api_key_for_this_backend = gemini_api_key
            elif backend_id.startswith("gpt"): api_key_for_this_backend = openai_api_key
            # Ollama doesn't use an API key in this context

            logger.info(
                f"  Configuring backend '{backend_id}' with model '{model_to_use}' and personality: {'Set' if personality_to_use else 'None'}")
            self._backend_coordinator.configure_backend(
                backend_id, api_key_for_this_backend, model_to_use, personality_to_use
            )
        logger.info("ChatManager: All initial backend configurations dispatched.")

    # --- NEW METHOD for LeftPanel Chat LLM ComboBox ---
    def get_all_available_chat_models_with_details(self) -> List[Dict[str, Any]]:
        all_models_details = []
        for provider_detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS:
            backend_id = provider_detail["id"]
            provider_name = provider_detail["name"]
            if backend_id not in self._backend_adapters_dict:
                logger.warning(f"Chat backend '{backend_id}' defined in details but no adapter found. Skipping.")
                continue

            try:
                models_for_backend = self.get_models_for_backend(backend_id)
                for model_name_from_adapter in models_for_backend:
                    all_models_details.append({
                        "display_name": f"{provider_name}: {model_name_from_adapter}",
                        "backend_id": backend_id,
                        "model_name": model_name_from_adapter
                    })
            except Exception as e:
                logger.error(f"Error fetching/processing models for chat backend {backend_id} ({provider_name}): {e}")
        logger.debug(f"CM: get_all_available_chat_models_with_details returning {len(all_models_details)} items.")
        return all_models_details

    # --- NEW METHOD for LeftPanel Specialized LLM ComboBox ---
    def get_all_available_specialized_models_with_details(self) -> List[Dict[str, Any]]:
        all_models_details = []
        # Currently, GENERATOR_BACKEND_ID is assumed to be Ollama
        # This could be made more dynamic if other backends are used for generation
        generator_provider_name = "Ollama (Specialized)" # Or derive from SPECIALIZED_BACKEND_DETAILS
        try:
            models_for_generator = self.get_models_for_backend(GENERATOR_BACKEND_ID)
            for model_name_from_adapter in models_for_generator:
                all_models_details.append({
                    "display_name": f"{generator_provider_name}: {model_name_from_adapter}",
                    "backend_id": GENERATOR_BACKEND_ID, # This backend_id is for configuration
                    "model_name": model_name_from_adapter
                })
        except Exception as e:
            logger.error(f"Error fetching/processing models for specialized backend {GENERATOR_BACKEND_ID}: {e}")
        logger.debug(f"CM: get_all_available_specialized_models_with_details returning {len(all_models_details)} items.")
        return all_models_details

    # This method is now used by the new flat model dropdown population logic
    def get_available_backend_details(self) -> List[Dict[str, str]]:
        # This method returns details for CHAT backends, used by the old LCP logic.
        # For the new flat Chat LLM dropdown, get_all_available_chat_models_with_details is preferred.
        # However, it can remain for other potential uses or backward compatibility if needed.
        valid_details = []
        for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS: # Still refers to chat backends
            if detail['id'] in self._backend_adapters_dict:
                valid_details.append(detail)
            else:
                logger.warning(f"Backend ID '{detail['id']}' (Name: '{detail['name']}') defined in USER_SELECTABLE_CHAT_BACKEND_DETAILS "
                               f"but no adapter found. It will not be available in UI for direct provider selection.")
        logger.debug(f"CM: get_available_backend_details returning: {valid_details} (for chat providers)")
        return valid_details

    def get_models_for_backend(self, backend_id: str) -> List[str]:
        if backend_id not in self._backend_adapters_dict:
            logger.error(f"CM: Cannot get models for '{backend_id}': No adapter instance exists for this ID.")
            return self._available_models_per_backend.get(backend_id, [])

        if self._backend_coordinator:
            try:
                # This now correctly fetches models for *any* backend_id, including GENERATOR_BACKEND_ID
                models = self._backend_coordinator.get_available_models_for_backend(backend_id)
                self._available_models_per_backend[backend_id] = models[:]
                logger.debug(f"CM: Fetched {len(models)} models for backend '{backend_id}': {models}")
                return models
            except Exception as e:
                logger.exception(f"Error fetching models for backend '{backend_id}' via BackendCoordinator:")
        else:
            logger.warning(f"CM: BackendCoordinator not available, cannot fetch models for '{backend_id}'.")
        return self._available_models_per_backend.get(backend_id, [])

    def get_model_for_backend(self, backend_id: str) -> Optional[str]:
        # This method now also serves to get the model for GENERATOR_BACKEND_ID
        model = self._current_model_names.get(backend_id)
        logger.debug(f"CM: get_model_for_backend('{backend_id}') returning '{model}'")
        return model

    def get_current_active_chat_backend_id(self) -> str:
        logger.debug(f"CM: get_current_active_chat_backend_id returning '{self._current_active_chat_backend_id}'")
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
        logger.info(f"CM: PCM active project changed to: {new_active_project_id}")
        active_history = self.get_project_history(new_active_project_id)
        self.history_changed.emit(active_history[:])
        self.current_project_changed.emit(new_active_project_id)
        self._update_rag_initialized_state(emit_status=True, project_id=new_active_project_id)
        self._trigger_save_last_session_state()

    @pyqtSlot(str, str, dict, str)
    def _handle_sfm_session_loaded(self, model_name: str, personality: Optional[str],
                                   proj_ctx_data: Dict[str, Any], active_pid_from_session: str):
        if not (self._project_context_manager and self._backend_coordinator): return

        session_extra_data = proj_ctx_data.pop("session_extra_data_on_load", None)
        active_backend_id_from_session = DEFAULT_CHAT_BACKEND_ID
        temperature_from_session = None
        generator_model_from_session = None # New for specialized model

        if session_extra_data and isinstance(session_extra_data, dict):
            active_backend_id_from_session = session_extra_data.get("active_chat_backend_id", DEFAULT_CHAT_BACKEND_ID)
            temp_val = session_extra_data.get("chat_temperature")
            if temp_val is not None:
                try: temperature_from_session = float(temp_val)
                except (ValueError, TypeError): pass
            generator_model_from_session = session_extra_data.get("generator_model_name") # Load generator model

        # Configure Chat AI
        if active_backend_id_from_session in self._current_model_names:
            self._current_active_chat_backend_id = active_backend_id_from_session
            self._current_model_names[self._current_active_chat_backend_id] = model_name or self._current_model_names.get(self._current_active_chat_backend_id, "")
            self._current_chat_personality_prompts[self._current_active_chat_backend_id] = personality
        else: # Fallback for older sessions or unknown backend ID
            logger.warning(f"Loaded active_backend_id '{active_backend_id_from_session}' from session is not a known chat backend. Applying to default chat backend.")
            self._current_model_names[DEFAULT_CHAT_BACKEND_ID] = model_name or DEFAULT_GEMINI_CHAT_MODEL
            self._current_chat_personality_prompts[DEFAULT_CHAT_BACKEND_ID] = personality
            self._current_active_chat_backend_id = DEFAULT_CHAT_BACKEND_ID

        if temperature_from_session is not None:
            self._current_chat_temperature = temperature_from_session

        # Configure Specialized/Generator AI Model
        if generator_model_from_session and GENERATOR_BACKEND_ID in self._current_model_names:
            self._current_model_names[GENERATOR_BACKEND_ID] = generator_model_from_session
            logger.info(f"Loaded generator model '{generator_model_from_session}' for backend {GENERATOR_BACKEND_ID} from session.")
        else:
            logger.info(f"No generator model in session or backend {GENERATOR_BACKEND_ID} not found, will use default.")


        self._project_context_manager.load_state(proj_ctx_data)
        self._configure_all_initial_backends() # This will pick up the loaded generator model
        self.set_current_project(active_pid_from_session)
        self._update_rag_initialized_state(emit_status=True, project_id=active_pid_from_session)
        self.update_status_based_on_state()

    @pyqtSlot()
    def _handle_sfm_active_history_cleared(self):
        if self._project_context_manager:
            active_project_id = self._project_context_manager.get_active_project_id()
            if active_project_id:
                history = self._project_context_manager.get_project_history(active_project_id)
                if history is not None: history.clear()
                self.history_changed.emit([])
# === core/chat_manager.py (Part 3/3) ===
    @pyqtSlot(str, str, dict, dict)
    def _handle_sfm_request_state_save(self, model_name: str, personality: Optional[str], all_project_data: Dict[str, Any], session_extra_data: Dict[str, Any]):
        if self._session_flow_manager:
            # session_extra_data already contains active_chat_backend_id and chat_temperature
            # We ensure generator_model_name is also included before saving
            session_extra_data["generator_model_name"] = self._current_model_names.get(GENERATOR_BACKEND_ID)
            self._session_flow_manager.save_current_session_to_last_state(model_name, personality, session_extra_data)

    @pyqtSlot(str)
    def _handle_backend_stream_started(self, request_id: str):
        logger.info(f"CM: BackendCoordinator reported stream_started for request_id '{request_id}'. Emitting to UI.")
        self.stream_started.emit(request_id)

    @pyqtSlot(str, str)
    def _handle_backend_chunk_received(self, request_id: str, chunk: str):
        current_active_mc_task = self._modification_coordinator and self._modification_coordinator.is_awaiting_llm_response()
        if not current_active_mc_task:
            self.stream_chunk_received.emit(chunk)
        else:
            logger.debug(f"CM: Suppressing stream chunk from req_id '{request_id}' during MC LLM wait.")

    @pyqtSlot(str, ChatMessage, dict)
    def _handle_backend_response_completed(self, request_id: str, completed_message: ChatMessage, usage_stats_with_metadata: dict):
        mc_current_phase_debug = "N/A"; mc_is_awaiting_llm_debug = "N/A"
        if self._modification_coordinator: mc_current_phase_debug = self._modification_coordinator._current_phase; mc_is_awaiting_llm_debug = self._modification_coordinator.is_awaiting_llm_response()
        logger.info(f"CM _handle_backend_response_completed TOP: ReqID='{request_id}', Purpose='{usage_stats_with_metadata.get('purpose')}', BackendForMC='{usage_stats_with_metadata.get('backend_id_for_mc')}', MC_Phase='{mc_current_phase_debug}', MC_AwaitingLLM='{mc_is_awaiting_llm_debug}', FullMeta='{usage_stats_with_metadata}'")
        purpose = usage_stats_with_metadata.get("purpose"); backend_id_for_mc = usage_stats_with_metadata.get("backend_id_for_mc")

        if purpose and isinstance(purpose, str) and purpose.startswith("psc_"):
            logger.debug(f"CM: Response for ReqID '{request_id}' (Purpose: '{purpose}') is for ProjectSummaryCoordinator. PSC will handle."); return
        is_mc_related_purpose = purpose and isinstance(purpose, str) and purpose.startswith("mc_request_")
        if self._modification_coordinator and self._modification_coordinator.is_active() and is_mc_related_purpose:
            mc_is_expecting_this = False
            if self._modification_coordinator.is_awaiting_llm_response(): mc_is_expecting_this = True
            elif (self._modification_coordinator._current_phase == ModPhase.AWAITING_CODE_GENERATION and backend_id_for_mc == GENERATOR_BACKEND_ID): mc_is_expecting_this = True
            elif (self._modification_coordinator._current_phase == ModPhase.AWAITING_PLAN and backend_id_for_mc == PLANNER_BACKEND_ID): mc_is_expecting_this = True
            elif (self._modification_coordinator._current_phase == ModPhase.AWAITING_GEMINI_REFINEMENT and backend_id_for_mc == PLANNER_BACKEND_ID): mc_is_expecting_this = True
            if mc_is_expecting_this: logger.info(f"CM: Routing completed LLM response for req_id '{request_id}' (Purpose: {purpose}) to MC."); self._modification_coordinator.process_llm_response(backend_id_for_mc or PLANNER_BACKEND_ID, completed_message); return
            else: logger.warning(f"CM: Response for req_id '{request_id}' has MC purpose ('{purpose}') but MC is not in a matching await state (Phase: {self._modification_coordinator._current_phase}, AwaitingLLM: {self._modification_coordinator.is_awaiting_llm_response()}). This response will NOT be displayed in chat."); return
        original_target_filename = usage_stats_with_metadata.get("original_target_filename")
        if purpose == "code_summary" and original_target_filename:
            logger.info(f"CM: Handling completed response as a CODE SUMMARY for '{original_target_filename}'.")
            self.status_update.emit(f"Ava's summary for '{original_target_filename}' is ready!", "#98c379", True, 3000)
            summary_msg_text = f"✨ **Ava's Summary for {original_target_filename}:** ✨\n\n{completed_message.text}"
            summary_chat_message = ChatMessage(role=SYSTEM_ROLE, parts=[summary_msg_text], metadata={"is_ava_summary": True, "target_file": original_target_filename, "is_internal": False})
            if self._project_context_manager: self._project_context_manager.add_message_to_active_project(summary_chat_message); self.new_message_added.emit(summary_chat_message); self._trigger_save_last_session_state()
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
                        self.new_message_added.emit(msg_in_history); message_updated_in_model = True; break
            if not message_updated_in_model:
                logger.warning(f"CM: Could not find existing AI message with ID '{request_id}' to update for UI. Adding as new (unexpected).")
                if completed_message.metadata is None: completed_message.metadata = {}
                completed_message.metadata["request_id"] = request_id; completed_message.loading_state = MessageLoadingState.COMPLETED
                self._project_context_manager.add_message_to_active_project(completed_message); self.new_message_added.emit(completed_message)
            self._trigger_save_last_session_state()
        self.stream_finished.emit()
        prompt_tokens = usage_stats_with_metadata.get("prompt_tokens"); completion_tokens = usage_stats_with_metadata.get("completion_tokens")
        backend_id_for_tokens = self._current_active_chat_backend_id
        if backend_id_for_mc : backend_id_for_tokens = backend_id_for_mc
        elif purpose and purpose.startswith("psc_"):
            if "technical_summary" in purpose: backend_id_for_tokens = GENERATOR_BACKEND_ID
            elif "friendly_summary" in purpose: backend_id_for_tokens = DEFAULT_CHAT_BACKEND_ID
        if prompt_tokens is not None and completion_tokens is not None and self._model_info_service:
            model_name_for_token_calc = self._current_model_names.get(backend_id_for_tokens, "")
            model_max_context = self._model_info_service.get_max_tokens(model_name_for_token_calc)
            self.token_usage_updated.emit(backend_id_for_tokens, prompt_tokens, completion_tokens, model_max_context)
        return

    @pyqtSlot(str, str)
    def _handle_backend_response_error(self, request_id: str, error_message_str: str):
        logger.error(f"CM: Received ERROR from BC for request_id '{request_id}': {error_message_str}")
        is_psc_related_error = False
        if self._project_summary_coordinator:
            # Check if the request_id matches those tracked by PSC
            if hasattr(self._project_summary_coordinator, '_current_request_id_tech_summary') and \
               self._project_summary_coordinator._current_request_id_tech_summary == request_id:
                is_psc_related_error = True
            elif hasattr(self._project_summary_coordinator, '_current_request_id_friendly_summary') and \
                 self._project_summary_coordinator._current_request_id_friendly_summary == request_id:
                is_psc_related_error = True

        if is_psc_related_error:
            logger.info(f"CM: Error for req_id '{request_id}' belongs to ProjectSummaryCoordinator. It will handle or has handled it via direct BC signal connection.")
            # PSC connects directly to BC.response_error, so it should handle its own errors.
            # We just need to make sure the UI doesn't show a duplicate/confusing error.
            # The stream_finished signal ensures loading indicators stop.
            self.stream_finished.emit()
            return

        is_mc_related_error = False
        if self._modification_coordinator and self._modification_coordinator.is_active():
             # Check if the request_id looks like one generated by MC
             # (e.g., starts with "mc_") or if MC is in a state expecting a response.
             if request_id.startswith("mc_"): # A simple check, MC might need more robust tracking
                 is_mc_related_error = True

        if is_mc_related_error:
            logger.info(f"CM: Error for req_id '{request_id}' appears to be for ModificationCoordinator. MC will handle via direct BC signal.")
            self.stream_finished.emit()
            return

        message_updated_in_model = False
        if self._project_context_manager:
            active_history = self._project_context_manager.get_active_conversation_history()
            if active_history:
                for i, msg_in_history in enumerate(reversed(active_history)):
                    if msg_in_history.id == request_id and msg_in_history.role == MODEL_ROLE:
                        logger.debug(f"CM: Found placeholder message ID '{request_id}' to update with error info.")
                        msg_in_history.role = ERROR_ROLE; msg_in_history.parts = [f"Backend Error (Request ID: {request_id[:8]}...): {error_message_str}"]; msg_in_history.loading_state = MessageLoadingState.COMPLETED # Or .ERROR
                        self.new_message_added.emit(msg_in_history); message_updated_in_model = True; break
            if not message_updated_in_model:
                logger.warning(f"CM: Error for req_id '{request_id}', but no placeholder found. Adding new error message.")
                err_obj = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"Backend Error (Request ID: {request_id[:8]}...): {error_message_str}"], loading_state=MessageLoadingState.COMPLETED) # Or .ERROR
                self._project_context_manager.add_message_to_active_project(err_obj); self.new_message_added.emit(err_obj)
            self._trigger_save_last_session_state()
        self.stream_finished.emit(); self.error_occurred.emit(f"Backend Error: {error_message_str}", False); return

    @pyqtSlot(bool)
    def _handle_backend_busy_changed(self, backend_is_busy: bool):
        logger.debug(f"CM: BC overall busy state changed to: {backend_is_busy}")
        self._update_overall_busy_state()

    @pyqtSlot(str, str, bool, list)
    def _handle_backend_configuration_changed(self, backend_id: str, model_name: str, is_configured: bool, available_models: list):
        logger.info(
            f"CM: BC config changed for '{backend_id}'. Model: {model_name}, ConfigOK: {is_configured}, Avail: {len(available_models)}")
        self._chat_backend_configured_successfully[backend_id] = is_configured
        self._available_models_per_backend[backend_id] = available_models[:]
        self._current_model_names[backend_id] = model_name # Ensure our state matches what BC configured

        if not is_configured and self._backend_coordinator:
            err = self._backend_coordinator.get_last_error_for_backend(backend_id) or f"{backend_id} config error."
            self.error_occurred.emit(f"Config Error ({backend_id} - {model_name}): {err}", False)

        self.available_models_changed_for_backend.emit(backend_id, self._available_models_per_backend.get(backend_id, []))
        personality_active = bool(self._current_chat_personality_prompts.get(backend_id))
        # This signal is now generic; LeftPanel will decide if it's for chat or specialized LLM
        self.backend_config_state_changed.emit(backend_id, model_name, is_configured, personality_active)

        if backend_id == self._current_active_chat_backend_id: self.update_status_based_on_state()
        elif backend_id in [PLANNER_BACKEND_ID, GENERATOR_BACKEND_ID]: # Non-chat backends status
            name_map = {PLANNER_BACKEND_ID: "Planner", GENERATOR_BACKEND_ID: "Specialized"} # Updated name
            d_name = name_map.get(backend_id, backend_id)
            status_color = "#98c379" if is_configured else "#e06c75"
            status_msg = f"{d_name} ({backend_id}) OK with {model_name}." if is_configured else f"{d_name} ({backend_id}) not configured ({model_name})."
            self.status_update.emit(status_msg, status_color, True, 3000 if is_configured else 5000)
        self._trigger_save_last_session_state()

    # ... (Upload and UIH handlers remain the same as they don't directly depend on the specifics of the two comboboxes,
    #      but rather on the configured active_chat_backend_id and GENERATOR_BACKEND_ID) ...
    # Omitting _handle_upload_... and _handle_uih_... for brevity here, assume they are unchanged from combined.txt

    @pyqtSlot(bool, str)
    def _handle_upload_started(self, is_global: bool, item_description: str):
        active_project_name_str = "N/A"; pcm = self._project_context_manager
        if pcm: active_project_name_str = pcm.get_active_project_name() or "Current"
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
        self.update_status_based_on_state(); self._trigger_save_last_session_state()

    @pyqtSlot(str)
    def _handle_upload_error(self, error_message_str: str):
        if not self._project_context_manager: return
        err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Upload System Error: {error_message_str}"])
        self._project_context_manager.add_message_to_active_project(err_obj); self.new_message_added.emit(err_obj)
        self.error_occurred.emit(f"Upload Error: {error_message_str}", False); self.update_status_based_on_state()

    @pyqtSlot(bool)
    def _handle_upload_busy_changed(self, upload_is_busy: bool): self._update_overall_busy_state()

    @pyqtSlot(list)
    def _handle_uih_normal_chat_request(self, new_user_message_list: List[ChatMessage]):
        logger.info(f"CM: Handling normal_chat_request_ready from UIH for backend '{self._current_active_chat_backend_id}'.")
        if not (self._backend_coordinator and self._project_context_manager):
            self.error_occurred.emit("Cannot send chat: Critical components missing.", True); return
        if not new_user_message_list or not isinstance(new_user_message_list[0], ChatMessage):
            logger.warning("CM: _handle_uih_normal_chat_request received invalid or empty message list. Ignoring."); return

        user_message_for_ui = new_user_message_list[0]
        self._project_context_manager.add_message_to_active_project(user_message_for_ui)
        self.new_message_added.emit(user_message_for_ui)
        self._trigger_save_last_session_state()
        logger.debug(f"CM: Added user message (ID: {user_message_for_ui.id}) to history for UI: {user_message_for_ui.text[:50]}...")
        QApplication.processEvents()

        ai_request_id = str(uuid.uuid4())
        ai_placeholder_message = ChatMessage(id=ai_request_id, role=MODEL_ROLE, parts=[""], loading_state=MessageLoadingState.LOADING)
        self._project_context_manager.add_message_to_active_project(ai_placeholder_message)
        self.new_message_added.emit(ai_placeholder_message)
        logger.info(f"CM: Added AI placeholder (ID: {ai_request_id}) with LOADING state for backend '{self._current_active_chat_backend_id}'.")

        full_history_for_backend = self._project_context_manager.get_active_conversation_history()
        if not full_history_for_backend:
            logger.error("CM: History for backend is empty. This is unexpected.");
            self.error_occurred.emit("Internal error preparing chat.", True); return

        request_options = {"temperature": self._current_chat_temperature}
        request_metadata_for_bc = {"original_user_message_id": user_message_for_ui.id}
        logger.debug(f"CM: Sending history (len {len(full_history_for_backend)}) to backend '{self._current_active_chat_backend_id}' for request_id '{ai_request_id}'.")

        self._backend_coordinator.request_response_stream(
            target_backend_id=self._current_active_chat_backend_id,
            request_id=ai_request_id,
            history_to_send=full_history_for_backend[:-1], # Exclude AI placeholder
            is_modification_response_expected=False,
            options=request_options,
            request_metadata=request_metadata_for_bc
        )

    @pyqtSlot(str, list, str, str)
    def _handle_uih_mod_start_request(self, original_query_text: str, image_data_list: List[Dict[str, Any]], context_for_mc: str, focus_prefix_for_mc: str):
        logger.info(f"CM: Handling modification_sequence_start_requested from UIH. Query: '{original_query_text[:50]}...'")
        if not (self._modification_coordinator and self._project_context_manager): self.error_occurred.emit("Modification feature unavailable or PCM missing.", True); return
        user_message_parts_for_ui = [original_query_text] + (image_data_list or [])
        user_chat_message_for_ui = ChatMessage(role=USER_ROLE, parts=user_message_parts_for_ui)
        self._project_context_manager.add_message_to_active_project(user_chat_message_for_ui); self.new_message_added.emit(user_chat_message_for_ui)
        self._trigger_save_last_session_state(); logger.debug(f"CM: Added user's modification request (ID: {user_chat_message_for_ui.id}) to history for UI display.")
        QApplication.processEvents()
        if self._modification_handler_instance: self._modification_handler_instance.activate_sequence()
        self._modification_coordinator.start_sequence(query=original_query_text, context=context_for_mc, focus_prefix=focus_prefix_for_mc)
        logger.info(f"CM: ModificationCoordinator sequence started for query: '{original_query_text[:50]}...'")

    @pyqtSlot(str, str)
    def _handle_uih_mod_user_input(self, user_command: str, action_type: str):
        logger.info(f"CM: Handling mod_user_input ('{user_command}', Type: '{action_type}') from UIH.")
        if self._modification_coordinator: self._modification_coordinator.process_user_input(user_command)
        else: self.error_occurred.emit("Modification feature unavailable.", False)

    @pyqtSlot(str)
    def _handle_uih_processing_error(self, error_message: str):
        logger.error(f"CM: UIH processing error: {error_message}")
        self.error_occurred.emit(f"Input Processing Error: {error_message}", False)
        if self._project_context_manager:
            err_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Input Error: {error_message}"])
            self._project_context_manager.add_message_to_active_project(err_obj); self.new_message_added.emit(err_obj)

    @pyqtSlot(ChatMessage)
    def _handle_user_command_for_display_only(self, user_message: ChatMessage):
        if not self._project_context_manager: logger.error("Cannot display user command: ProjectContextManager is missing."); return
        logger.debug(f"ChatManager: Adding user command (ID: {user_message.id}, Text: '{user_message.text[:50]}...') to history for display only.")
        self._project_context_manager.add_message_to_active_project(user_message); self.new_message_added.emit(user_message)
        self._trigger_save_last_session_state()


    # --- MC, PSC Handlers (Assume unchanged unless direct impact from model selection change) ---
    # Omitting _handle_mc_... and _handle_project_summary_... for brevity, assume they are unchanged from combined.txt
    @pyqtSlot(str, list)
    def _handle_mc_request_llm_call(self, target_backend_id: str, history_to_send: List[ChatMessage]):
        if self._backend_coordinator:
            mc_options = {"temperature": 0.5};
            if target_backend_id == GENERATOR_BACKEND_ID: mc_options = {"temperature": 0.2}
            mc_internal_request_id = f"mc_{target_backend_id}_{str(uuid.uuid4())[:8]}"
            request_metadata_for_mc = {"purpose": f"mc_request_{target_backend_id}", "mc_internal_id": mc_internal_request_id, "backend_id_for_mc": target_backend_id}
            logger.debug(f"CM MC LLM Call: Target='{target_backend_id}', ReqID='{mc_internal_request_id}', Meta='{request_metadata_for_mc}'")
            self._backend_coordinator.request_response_stream(target_backend_id=target_backend_id, request_id=mc_internal_request_id, history_to_send=history_to_send, is_modification_response_expected=True, options=mc_options, request_metadata=request_metadata_for_mc)
        elif self._modification_coordinator: self._modification_coordinator.process_llm_error(target_backend_id, "BackendCoordinator unavailable.")

    @pyqtSlot(str, str, str)
    def _handle_code_generated_and_summary_needed(self, generated_code: str, coder_instructions: str, target_filename: str):
        logger.info(f"CM: Summary needed for '{target_filename}'. Delegating to CodeSummaryService.")
        if not (self._code_summary_service and self._backend_coordinator): self.error_occurred.emit(f"Internal error: Services unavailable for summary of '{target_filename}'.", True); return
        self.status_update.emit(f"Ava is preparing summary for '{target_filename}'...", "#e5c07b", True, 4000)
        success = self._code_summary_service.request_code_summary(self._backend_coordinator, target_filename, coder_instructions, generated_code)
        if not success:
            err_msg = f"Failed to dispatch summary request for '{target_filename}'."; logger.error(err_msg)
            if self._project_context_manager: sys_err_msg = ChatMessage(role=ERROR_ROLE, parts=[f"[System: Error initiating summary for '{target_filename}'.]"]); self._project_context_manager.add_message_to_active_project(sys_err_msg); self.new_message_added.emit(sys_err_msg)
            self.update_status_based_on_state()

    @pyqtSlot(str, str)
    def _handle_mc_file_ready(self, filename: str, content: str):
        self.code_file_updated.emit(filename, content)
        if self._project_context_manager: sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[f"[System: File '{filename}' updated. See Code Viewer.]"], metadata={"is_internal": False}); self._project_context_manager.add_message_to_active_project(sys_msg); self.new_message_added.emit(sys_msg)

    @pyqtSlot(str, str)
    def _handle_mc_sequence_complete(self, reason: str, original_query_summary: str):
        if self._project_context_manager:
            system_message_text = (f"[System: The multi-file code modification sequence by the Coder AI for '{original_query_summary}' has ended ({reason}). All generated code is available in the Code Viewer. Ava, please respond conversationally to the user's next message. Do not output full code blocks for the task that just completed.]")
            logger.info(f"CM: Modification sequence complete. Adding guiding system message: {system_message_text}")
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[system_message_text], metadata={"is_internal": True}); self._project_context_manager.add_message_to_active_project(sys_msg); self._trigger_save_last_session_state()
        self.update_status_based_on_state();
        if self._modification_handler_instance: self._modification_handler_instance.cancel_modification()

    @pyqtSlot(str)
    def _handle_mc_error(self, error_message: str):
        if self._project_context_manager: err_msg_obj = ChatMessage(role=ERROR_ROLE, parts=[f"Modification System Error: {error_message}"], metadata={"is_internal": False}); self._project_context_manager.add_message_to_active_project(err_msg_obj); self.new_message_added.emit(err_msg_obj); self._trigger_save_last_session_state()
        self.error_occurred.emit(f"Modification Error: {error_message}", False); self.update_status_based_on_state()

    @pyqtSlot(str)
    def _handle_mc_status_update(self, message: str):
        if self._project_context_manager: status_msg = ChatMessage(role=SYSTEM_ROLE, parts=[message], metadata={"is_internal": False}); self._project_context_manager.add_message_to_active_project(status_msg); self.new_message_added.emit(status_msg)

    @pyqtSlot(str, str)
    def _handle_project_summary_generated(self, project_id: str, friendly_summary_text: str):
        logger.info(f"ChatManager: Project summary generated for '{project_id}'.")
        if not self._project_context_manager: return
        project_name = self._project_context_manager.get_project_name(project_id) or project_id
        summary_message_text = f"✨ **Ava's Project Insights for '{project_name}'!** ✨\n\n{friendly_summary_text}"
        summary_chat_message = ChatMessage(role=MODEL_ROLE, parts=[summary_message_text], metadata={"is_project_summary": True, "project_id": project_id, "is_internal": False})
        target_history = self._project_context_manager.get_project_history(project_id)
        if target_history is not None:
            target_history.append(summary_chat_message)
            if self._project_context_manager.get_active_project_id() == project_id: self.new_message_added.emit(summary_chat_message); logger.debug(f"Added project summary for active project '{project_id}' to UI.")
            else: logger.info(f"Project summary for '{project_id}' generated but it's not the active project. User will see it when they switch."); self.status_update.emit(f"Project summary for project '{project_name}' is ready!", "#98c379", True, 5000)
        else: logger.error(f"Could not find history for project '{project_id}' to add summary message.")
        self._trigger_save_last_session_state(); self.update_status_based_on_state()

    @pyqtSlot(str, str)
    def _handle_project_summary_failed(self, project_id: str, error_message: str):
        logger.error(f"ChatManager: Project summary generation failed for '{project_id}': {error_message}")
        if not self._project_context_manager: return
        project_name = self._project_context_manager.get_project_name(project_id) or project_id
        error_chat_message = ChatMessage(role=ERROR_ROLE, parts=[f"[System Error: Could not generate summary for project '{project_name}'. Reason: {error_message}]"], metadata={"is_project_summary_error": True, "project_id": project_id, "is_internal": False})
        target_history = self._project_context_manager.get_project_history(project_id)
        if target_history is not None:
            target_history.append(error_chat_message)
            if self._project_context_manager.get_active_project_id() == project_id: self.new_message_added.emit(error_chat_message); logger.debug(f"Added project summary error for active project '{project_id}' to UI.")
            else: logger.info(f"Project summary error for '{project_id}' occurred but it's not the active project.")
        else: logger.error(f"Could not find history for project '{project_id}' to add summary error message.")
        self.error_occurred.emit(f"Summary failed for '{project_name}': {error_message}", False); self._trigger_save_last_session_state(); self.update_status_based_on_state()

    def _cancel_active_tasks(self):
        if self._backend_coordinator: self._backend_coordinator.cancel_current_task()
        if self._upload_coordinator: self._upload_coordinator.cancel_current_upload()
        if self._modification_coordinator and self._modification_coordinator.is_active():
            self._modification_coordinator.cancel_sequence(reason="user_cancel_all")

    def cleanup(self):
        self._cancel_active_tasks()
        self._trigger_save_last_session_state()

    def _update_rag_initialized_state(self, emit_status: bool = True, project_id: Optional[str] = None):
        if not self._project_context_manager: return
        target_pid = project_id or (self._project_context_manager.get_active_project_id())
        new_init_state = self.is_rag_context_initialized(target_pid)
        active_pid = self._project_context_manager.get_active_project_id()
        if target_pid == active_pid:
            if self._rag_initialized != new_init_state:
                self._rag_initialized = new_init_state
        if emit_status or (target_pid == active_pid and self._rag_initialized != new_init_state):
            self.update_status_based_on_state()

    def is_rag_context_initialized(self, project_id: Optional[str]) -> bool:
        if not (self._vector_db_service and project_id):
            self._rag_available = False
            return False
        self._rag_available = True
        is_vdb_ready = self._vector_db_service.is_ready(project_id)
        size = self._vector_db_service.get_collection_size(project_id) if is_vdb_ready else 0
        return is_vdb_ready and size > 0

    def get_project_history(self, project_id: str) -> List[ChatMessage]:
        return list(self._project_context_manager.get_project_history(project_id) or []) if self._project_context_manager else []

    def get_current_history(self) -> List[ChatMessage]:
        return list(self._project_context_manager.get_active_conversation_history() or []) if self._project_context_manager else []

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
        try: return self._vector_db_service.get_all_metadata(target_id)
        except Exception as e: logger.exception(f"Error RAG contents for '{target_id}': {e}"); return []

    def get_current_focus_paths(self) -> Optional[List[str]]:
        return self._current_chat_focus_paths

    def get_project_context_manager(self) -> Optional[ProjectContextManager]: return self._project_context_manager
    def get_backend_coordinator(self) -> Optional[BackendCoordinator]: return self._backend_coordinator
    def get_upload_coordinator(self) -> Optional[UploadCoordinator]: return self._upload_coordinator
    def get_modification_coordinator(self) -> Optional[ModificationCoordinator]: return self._modification_coordinator
    def get_session_flow_manager(self) -> Optional[SessionFlowManager]: return self._session_flow_manager
    def get_project_summary_coordinator(self) -> Optional[ProjectSummaryCoordinator]: return self._project_summary_coordinator

    def _trigger_save_last_session_state(self):
        if self._session_flow_manager:
            active_chat_backend_id = self._current_active_chat_backend_id
            active_chat_model = self._current_model_names.get(active_chat_backend_id)
            active_chat_pers = self._current_chat_personality_prompts.get(active_chat_backend_id)

            # Prepare session_extra_data
            session_extra_data = {
                "active_chat_backend_id": active_chat_backend_id,
                "chat_temperature": self._current_chat_temperature,
                "generator_model_name": self._current_model_names.get(GENERATOR_BACKEND_ID) # Save specialized model
                # Add other specific backend model selections here if they become configurable (e.g., planner)
            }
            # Remove None values from session_extra_data to keep saved file clean
            session_extra_data = {k: v for k, v in session_extra_data.items() if v is not None}

            self._session_flow_manager.save_current_session_to_last_state(active_chat_model, active_chat_pers, session_extra_data)

    def get_current_chat_model(self) -> str: # Gets model for ACTIVE CHAT backend
        return self._current_model_names.get(self._current_active_chat_backend_id, "Unknown Model")
    def get_current_chat_personality(self) -> Optional[str]: # Gets personality for ACTIVE CHAT backend
        return self._current_chat_personality_prompts.get(self._current_active_chat_backend_id)

    def set_active_chat_backend(self, backend_id: str):
        # This method is now solely for setting the CHAT backend.
        # The specialized/generator backend model is set via set_model_for_backend(GENERATOR_BACKEND_ID, ...)
        if backend_id not in self._current_model_names: # Should check against USER_SELECTABLE_CHAT_BACKEND_DETAILS
            is_valid_chat_backend = any(detail["id"] == backend_id for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS)
            if not is_valid_chat_backend:
                logger.error(f"Attempted to set unknown/non-chat backend_id '{backend_id}' as active chat backend.")
                self.error_occurred.emit(f"Invalid chat backend type selected: {backend_id}", False)
                return

        if self._current_active_chat_backend_id != backend_id:
            logger.info(f"Switching active CHAT backend from '{self._current_active_chat_backend_id}' to '{backend_id}'.")
            self._current_active_chat_backend_id = backend_id
            # Ensure the newly active chat backend is configured with its currently selected model & personality
            model_to_ensure = self._current_model_names.get(backend_id, "") # Get its current model setting
            pers_to_ensure = self._current_chat_personality_prompts.get(backend_id)
            api_key_to_use = None
            if backend_id.startswith("gemini"): api_key_to_use = get_gemini_api_key()
            elif backend_id.startswith("gpt"): api_key_to_use = get_openai_api_key()

            if self._backend_coordinator:
                self._backend_coordinator.configure_backend(backend_id, api_key_to_use, model_to_ensure, pers_to_ensure)
            self.update_status_based_on_state()
            self._trigger_save_last_session_state()

    def set_model_for_backend(self, backend_id: str, model_name: str):
        # This method can now be used for ANY backend_id, including GENERATOR_BACKEND_ID.
        if backend_id not in self._current_model_names:
            logger.error(f"Cannot set model for unknown backend_id: {backend_id}")
            self.error_occurred.emit(f"Cannot set model for invalid backend: {backend_id}", False)
            return
        if not model_name:
            logger.error(f"Cannot set empty model name for backend_id: {backend_id}")
            self.error_occurred.emit(f"Model name cannot be empty for backend: {backend_id}", False)
            return

        logger.info(f"Setting model for backend '{backend_id}' to '{model_name}'.")
        self._current_model_names[backend_id] = model_name
        api_key_to_use = None
        if backend_id.startswith("gemini"): api_key_to_use = get_gemini_api_key()
        elif backend_id.startswith("gpt"): api_key_to_use = get_openai_api_key()

        if self._backend_coordinator:
            self._backend_coordinator.configure_backend(
                backend_id, api_key_to_use, model_name,
                self._current_chat_personality_prompts.get(backend_id) # Use existing personality for this backend
            )
        # If the backend_id is the currently active chat backend, update status
        if backend_id == self._current_active_chat_backend_id:
             self.update_status_based_on_state()
        self._trigger_save_last_session_state()

    def set_personality_for_backend(self, backend_id: str, prompt: Optional[str]):
        if backend_id not in self._current_chat_personality_prompts:
            logger.error(f"Cannot set personality for unknown backend_id: {backend_id}")
            self.error_occurred.emit(f"Cannot set personality for invalid backend: {backend_id}", False)
            return

        new_prompt_val = prompt.strip() if prompt else None
        logger.info(f"Setting personality for backend '{backend_id}'. New prompt: {'Set' if new_prompt_val else 'None'}")
        self._current_chat_personality_prompts[backend_id] = new_prompt_val
        api_key_to_use = None
        if backend_id.startswith("gemini"): api_key_to_use = get_gemini_api_key()
        elif backend_id.startswith("gpt"): api_key_to_use = get_openai_api_key()

        if self._backend_coordinator:
            self._backend_coordinator.configure_backend(
                backend_id, api_key_to_use,
                self._current_model_names.get(backend_id, ""), # Use current model for this backend
                new_prompt_val
            )
        # If the backend_id is the currently active chat backend, update status
        if backend_id == self._current_active_chat_backend_id:
             self.update_status_based_on_state()
        self._trigger_save_last_session_state()

    def set_model(self, model_name: str): # DEPRECATED
        logger.warning("DEPRECATED ChatManager.set_model() called. Use set_model_for_backend(active_chat_backend_id, model_name) instead.")
        self.set_model_for_backend(self._current_active_chat_backend_id, model_name)

    def set_personality(self, prompt: Optional[str]): # DEPRECATED
        logger.warning("DEPRECATED ChatManager.set_personality() called. Use set_personality_for_backend(active_chat_backend_id, prompt) instead.")
        self.set_personality_for_backend(self._current_active_chat_backend_id, prompt)

    def set_current_project(self, project_id: str):
        if self._project_context_manager:
            if not self._project_context_manager.set_active_project(project_id):
                self.error_occurred.emit(f"Failed to set project '{project_id}'.", False)

    def create_project_collection(self, project_name: str):
        if self._project_context_manager:
            if not self._project_context_manager.create_project(project_name):
                self.error_occurred.emit(f"Failed to create project '{project_name}'.", False)
            else: self.status_update.emit(f"Project '{project_name}' created.", "#98c379", True, 3000)

    def start_new_chat(self):
        if self._session_flow_manager:
            active_chat_model = self._current_model_names.get(self._current_active_chat_backend_id)
            active_chat_pers = self._current_chat_personality_prompts.get(self._current_active_chat_backend_id)
            session_extra_data = {
                "active_chat_backend_id": self._current_active_chat_backend_id,
                "chat_temperature": self._current_chat_temperature,
                "generator_model_name": self._current_model_names.get(GENERATOR_BACKEND_ID)
            }
            session_extra_data = {k: v for k, v in session_extra_data.items() if v is not None}
            self._session_flow_manager.start_new_chat_session(active_chat_model, active_chat_pers, session_extra_data)

    def load_chat_session(self, filepath: str):
        if self._session_flow_manager:
            # Pass current active chat backend as a default if session doesn't specify one.
            self._session_flow_manager.load_named_session(filepath, self._current_active_chat_backend_id)

    def save_current_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager:
            active_chat_model = self._current_model_names.get(self._current_active_chat_backend_id)
            active_chat_pers = self._current_chat_personality_prompts.get(self._current_active_chat_backend_id)
            session_extra_data = {
                "active_chat_backend_id": self._current_active_chat_backend_id,
                "chat_temperature": self._current_chat_temperature,
                "generator_model_name": self._current_model_names.get(GENERATOR_BACKEND_ID)
            }
            session_extra_data = {k: v for k, v in session_extra_data.items() if v is not None}
            return self._session_flow_manager.save_session_as(filepath, active_chat_model, active_chat_pers, session_extra_data)
        return False

    def delete_chat_session(self, filepath: str) -> bool:
        if self._session_flow_manager: return self._session_flow_manager.delete_named_session(filepath)
        return False

    def list_saved_sessions(self) -> List[str]:
        if self._session_flow_manager: return self._session_flow_manager.list_saved_sessions()
        return []

    def process_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        if self._user_input_handler:
            self._user_input_handler.handle_user_message(text=text, image_data=image_data,
                focus_paths=self._current_chat_focus_paths,
                rag_available=self._rag_available,
                rag_initialized_for_project=self.is_rag_context_initialized(self.get_current_project_id()))

    def update_status_based_on_state(self):
        active_backend_is_ready = self._chat_backend_configured_successfully.get(self._current_active_chat_backend_id, False)
        active_backend_display_name = self._current_active_chat_backend_id
        # Find display name for the active CHAT backend
        for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS: # Iterate CHAT backend details
            if detail['id'] == self._current_active_chat_backend_id:
                active_backend_display_name = detail['name']; break

        if not active_backend_is_ready:
            err_msg = f"API Not Configured ({active_backend_display_name})"
            if self._backend_coordinator:
                err = self._backend_coordinator.get_last_error_for_backend(self._current_active_chat_backend_id)
                if err: err_msg = f"API Error ({active_backend_display_name}): {err}"
            self.status_update.emit(f"{err_msg}. Check settings.", "#e06c75", False, 0)
        elif self._overall_busy:
            self.status_update.emit(f"Processing with {active_backend_display_name}...", "#e5c07b", False, 0)
        else:
            parts = [f"Ready ({active_backend_display_name})"]
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
            active_backend_display_name = self._current_active_chat_backend_id # Name of the active CHAT backend
            for detail in USER_SELECTABLE_CHAT_BACKEND_DETAILS:
                if detail['id'] == self._current_active_chat_backend_id:
                    active_backend_display_name = detail['name']; break
            self.status_update.emit(f"Temperature for '{active_backend_display_name}' set to {self._current_chat_temperature:.2f}", "#61afef", True, 3000)
            self._trigger_save_last_session_state()
        else: logger.warning(f"Attempt to set invalid temperature: {temperature}")

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

    def is_api_ready(self) -> bool: # Helper for MainWindow to check general readiness OF CHAT API
        return self._chat_backend_configured_successfully.get(self._current_active_chat_backend_id, False)

# --- End of Part 3/3 ---