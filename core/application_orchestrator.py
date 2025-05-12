# core/application_orchestrator.py
import logging
from typing import Dict, Optional

# --- Backend Adapters ---
from backend.interface import BackendInterface
from backend.gemini_adapter import GeminiAdapter
from backend.ollama_adapter import OllamaAdapter

# --- Core Components ---
from core.project_context_manager import ProjectContextManager
from core.backend_coordinator import BackendCoordinator
from core.session_flow_manager import SessionFlowManager
from core.upload_coordinator import UploadCoordinator
from core.rag_handler import RagHandler
from core.user_input_processor import UserInputProcessor
from core.user_input_handler import UserInputHandler

# Import ModificationHandler and ModificationCoordinator with try-except
try:
    from core.modification_handler import ModificationHandler
    MOD_HANDLER_AVAILABLE = True
except ImportError as e:
    ModificationHandler = None # type: ignore
    MOD_HANDLER_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ModificationHandler: {e}.")

try:
    from core.modification_coordinator import ModificationCoordinator, ModPhase
    MOD_COORDINATOR_AVAILABLE = True
except ImportError as e:
    ModificationCoordinator = None # type: ignore
    ModPhase = None # type: ignore
    MOD_COORDINATOR_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ModificationCoordinator or ModPhase: {e}.")


# --- Services (passed in) ---
from services.session_service import SessionService
from services.upload_service import UploadService, VECTOR_DB_SERVICE_AVAILABLE
from services.vector_db_service import VectorDBService


logger = logging.getLogger(__name__)

# --- Backend ID Constants (Mirrored from ChatManager, consider moving to utils.constants later) ---
DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
OLLAMA_CHAT_BACKEND_ID = "ollama_chat"
PLANNER_BACKEND_ID = "gemini_planner"
GENERATOR_BACKEND_ID = "ollama_generator"
# --- End Backend ID Constants ---

class ApplicationOrchestrator:
    def __init__(self, session_service: SessionService, upload_service: UploadService):
        logger.info("ApplicationOrchestrator initializing...")
        self._session_service = session_service
        self._upload_service = upload_service
        self._vector_db_service = getattr(upload_service, '_vector_db_service', None)
        if not isinstance(self._vector_db_service, VectorDBService):
            self._vector_db_service = None # type: ignore
            logger.warning("ApplicationOrchestrator: VectorDBService instance not available from UploadService!")


        # --- 1. Create Adapters ---
        self.gemini_chat_default_adapter = GeminiAdapter()
        self.ollama_chat_adapter = OllamaAdapter()
        self.gemini_planner_adapter = GeminiAdapter()
        # Assuming ollama_generator_adapter can use the same instance as ollama_chat_adapter
        self.ollama_generator_adapter: BackendInterface = self.ollama_chat_adapter

        self._all_backend_adapters_dict: Dict[str, BackendInterface] = {
            DEFAULT_CHAT_BACKEND_ID: self.gemini_chat_default_adapter,
            OLLAMA_CHAT_BACKEND_ID: self.ollama_chat_adapter,
            PLANNER_BACKEND_ID: self.gemini_planner_adapter,
            GENERATOR_BACKEND_ID: self.ollama_generator_adapter,
        }
        logger.debug("Adapters instantiated.")

        # --- 2. Create Core Services & Coordinators (respecting dependencies) ---

        # ProjectContextManager (No external dependencies for its __init__)
        self.project_context_manager = ProjectContextManager()
        logger.debug("ProjectContextManager instantiated.")

        # BackendCoordinator (Needs adapters)
        self.backend_coordinator = BackendCoordinator(self._all_backend_adapters_dict)
        logger.debug("BackendCoordinator instantiated.")

        # RagHandler (Needs UploadService and its VectorDBService)
        self.rag_handler: Optional[RagHandler] = None
        if self._upload_service and self._vector_db_service:
            self.rag_handler = RagHandler(self._upload_service, self._vector_db_service)
            logger.debug("RagHandler instantiated.")
        else:
            logger.warning("ApplicationOrchestrator: RagHandler cannot be instantiated (UploadService or VectorDBService missing).")

        # ModificationHandler (No external dependencies for its __init__)
        self.modification_handler_instance: Optional[ModificationHandler] = None
        if MOD_HANDLER_AVAILABLE and ModificationHandler is not None:
            try:
                self.modification_handler_instance = ModificationHandler() # Parent will be set by ChatManager
                logger.debug("ModificationHandler instantiated.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ModificationHandler: {e}", exc_info=True)
        else:
            logger.info("ApplicationOrchestrator: ModificationHandler not available or not imported, skipping instantiation.")

        # UserInputProcessor (Needs RagHandler, ModificationHandler)
        self.user_input_processor_instance: Optional[UserInputProcessor] = None
        if self.rag_handler: # Check if RagHandler was successfully created
            try:
                self.user_input_processor_instance = UserInputProcessor(
                    self.rag_handler,
                    self.modification_handler_instance # Pass even if None, UIP can handle it
                )
                logger.debug("UserInputProcessor instantiated.")
            except Exception as e:
                logger.critical(f"ApplicationOrchestrator: Failed to instantiate UserInputProcessor: {e}", exc_info=True)
        else:
            logger.critical("ApplicationOrchestrator: Cannot instantiate UserInputProcessor, RagHandler missing.")

        # ModificationCoordinator (Needs ModHandler, BackendCoordinator, ProjectContextManager)
        self.modification_coordinator: Optional[ModificationCoordinator] = None
        if MOD_COORDINATOR_AVAILABLE and ModificationCoordinator is not None and \
           self.modification_handler_instance and self.backend_coordinator and self.project_context_manager:
            try:
                self.modification_coordinator = ModificationCoordinator(
                    modification_handler=self.modification_handler_instance,
                    backend_coordinator=self.backend_coordinator,
                    project_context_manager=self.project_context_manager
                    # Parent will be set by ChatManager
                )
                logger.debug("ModificationCoordinator instantiated.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ModificationCoordinator: {e}", exc_info=True)
        else:
            logger.warning("ApplicationOrchestrator: ModificationCoordinator cannot be instantiated (dependencies missing or import failed).")

        # SessionFlowManager (Needs SessionService, ProjectContextManager, BackendCoordinator)
        self.session_flow_manager: Optional[SessionFlowManager] = None
        if self._session_service and self.project_context_manager and self.backend_coordinator:
            self.session_flow_manager = SessionFlowManager(
                session_service=self._session_service,
                project_context_manager=self.project_context_manager,
                backend_coordinator=self.backend_coordinator
                # Parent will be set by ChatManager
            )
            logger.debug("SessionFlowManager instantiated.")
        else:
            logger.critical("ApplicationOrchestrator: SessionFlowManager could not be initialized due to missing dependencies.")

        # UploadCoordinator (Needs UploadService, ProjectContextManager)
        self.upload_coordinator: Optional[UploadCoordinator] = None
        if self._upload_service and self.project_context_manager:
            self.upload_coordinator = UploadCoordinator(
                self._upload_service,
                self.project_context_manager
                # Parent will be set by ChatManager
            )
            logger.debug("UploadCoordinator instantiated.")
        else:
            logger.error("ApplicationOrchestrator: Cannot initialize UploadCoordinator due to missing dependencies.")

        # UserInputHandler (Needs UserInputProcessor, ProjectContextManager, ModificationCoordinator)
        self.user_input_handler: Optional[UserInputHandler] = None
        if self.user_input_processor_instance and self.project_context_manager:
            try:
                self.user_input_handler = UserInputHandler(
                    user_input_processor=self.user_input_processor_instance,
                    project_context_manager=self.project_context_manager,
                    modification_coordinator=self.modification_coordinator # Pass even if None
                    # Parent will be set by ChatManager
                )
                logger.debug("UserInputHandler instantiated.")
            except Exception as e:
                logger.critical(f"ApplicationOrchestrator: Failed to initialize UserInputHandler: {e}", exc_info=True)
        else:
            logger.critical("ApplicationOrchestrator: UserInputHandler cannot be initialized (dependencies missing).")

        logger.info("ApplicationOrchestrator core components instantiation process complete.")

    # --- Public Getters for Components ---
    def get_all_backend_adapters_dict(self) -> Dict[str, BackendInterface]:
        return self._all_backend_adapters_dict

    def get_project_context_manager(self) -> ProjectContextManager:
        # This should ideally not return None if orchestration is successful
        # and PCM has no critical instantiation deps beyond itself.
        if self.project_context_manager is None:
            logger.critical("get_project_context_manager called but instance is None!")
            # This would be a critical error, consider raising or handling robustly
            raise RuntimeError("ProjectContextManager not instantiated in Orchestrator.")
        return self.project_context_manager

    def get_backend_coordinator(self) -> BackendCoordinator:
        if self.backend_coordinator is None:
            logger.critical("get_backend_coordinator called but instance is None!")
            raise RuntimeError("BackendCoordinator not instantiated in Orchestrator.")
        return self.backend_coordinator

    def get_session_flow_manager(self) -> Optional[SessionFlowManager]:
        return self.session_flow_manager

    def get_upload_coordinator(self) -> Optional[UploadCoordinator]:
        return self.upload_coordinator

    def get_user_input_handler(self) -> Optional[UserInputHandler]:
        return self.user_input_handler

    def get_modification_coordinator(self) -> Optional[ModificationCoordinator]:
        return self.modification_coordinator

    # Optional getters for components ChatManager might not directly own but still queries
    def get_rag_handler(self) -> Optional[RagHandler]:
        return self.rag_handler

    def get_modification_handler_instance(self) -> Optional[ModificationHandler]:
        return self.modification_handler_instance

    def get_user_input_processor_instance(self) -> Optional[UserInputProcessor]:
        return self.user_input_processor_instance