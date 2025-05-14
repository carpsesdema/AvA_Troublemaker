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
from core.upload_coordinator import UploadCoordinator  # Will receive PSC
from core.rag_handler import RagHandler
from core.user_input_processor import UserInputProcessor
from core.user_input_handler import UserInputHandler  # Will receive PSC

# Import ModificationHandler and ModificationCoordinator with try-except
try:
    from core.modification_handler import ModificationHandler

    MOD_HANDLER_AVAILABLE = True
except ImportError as e:
    ModificationHandler = None  # type: ignore
    MOD_HANDLER_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ModificationHandler: {e}.")

try:
    from core.modification_coordinator import ModificationCoordinator, ModPhase

    MOD_COORDINATOR_AVAILABLE = True
except ImportError as e:
    ModificationCoordinator = None  # type: ignore
    ModPhase = None  # type: ignore
    MOD_COORDINATOR_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ModificationCoordinator or ModPhase: {e}.")

# --- Imports for Project Summary Feature ---
try:
    from services.project_intelligence_service import ProjectIntelligenceService

    PROJECT_INTEL_SERVICE_AVAILABLE = True
except ImportError as e:
    ProjectIntelligenceService = None  # type: ignore
    PROJECT_INTEL_SERVICE_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ProjectIntelligenceService: {e}.")

try:
    from core.project_summary_coordinator import ProjectSummaryCoordinator

    PROJECT_SUMMARY_COORDINATOR_AVAILABLE = True
except ImportError as e:
    ProjectSummaryCoordinator = None  # type: ignore
    PROJECT_SUMMARY_COORDINATOR_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ProjectSummaryCoordinator: {e}.")
# --- END Imports ---


# --- Services (passed in) ---
from services.session_service import SessionService
from services.upload_service import UploadService, VECTOR_DB_SERVICE_AVAILABLE
from services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)

# --- Backend ID Constants ---
DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
OLLAMA_CHAT_BACKEND_ID = "ollama_chat"
PLANNER_BACKEND_ID = "gemini_planner"
GENERATOR_BACKEND_ID = "ollama_generator"


# --- End Backend ID Constants ---

class ApplicationOrchestrator:
    def __init__(self, session_service: SessionService, upload_service: UploadService):
        logger.info("ApplicationOrchestrator initializing...")
        self._session_service = session_service
        self._upload_service = upload_service  # Stored for UploadCoordinator
        self._vector_db_service = getattr(upload_service, '_vector_db_service', None)
        if not isinstance(self._vector_db_service, VectorDBService):
            self._vector_db_service = None  # type: ignore
            logger.warning("ApplicationOrchestrator: VectorDBService instance not available from UploadService!")

        # --- 1. Create Adapters ---
        self.gemini_chat_default_adapter = GeminiAdapter()
        self.ollama_chat_adapter = OllamaAdapter()
        self.gemini_planner_adapter = GeminiAdapter()
        self.ollama_generator_adapter: BackendInterface = self.ollama_chat_adapter
        self._all_backend_adapters_dict: Dict[str, BackendInterface] = {
            DEFAULT_CHAT_BACKEND_ID: self.gemini_chat_default_adapter,
            OLLAMA_CHAT_BACKEND_ID: self.ollama_chat_adapter,
            PLANNER_BACKEND_ID: self.gemini_planner_adapter,
            GENERATOR_BACKEND_ID: self.ollama_generator_adapter,
        }
        logger.debug("Adapters instantiated.")

        # --- 2. Create Core Services & Coordinators (respecting dependencies) ---
        self.project_context_manager = ProjectContextManager()
        logger.debug("ProjectContextManager instantiated.")

        self.backend_coordinator = BackendCoordinator(self._all_backend_adapters_dict)
        logger.debug("BackendCoordinator instantiated.")

        self.rag_handler: Optional[RagHandler] = None
        if self._upload_service and self._vector_db_service:  # Use the instance var for upload_service
            self.rag_handler = RagHandler(self._upload_service, self._vector_db_service)
            logger.debug("RagHandler instantiated.")
        else:
            logger.warning(
                "ApplicationOrchestrator: RagHandler cannot be instantiated (UploadService or VectorDBService missing).")

        self.modification_handler_instance: Optional[ModificationHandler] = None
        if MOD_HANDLER_AVAILABLE and ModificationHandler is not None:
            try:
                self.modification_handler_instance = ModificationHandler()
                logger.debug("ModificationHandler instantiated.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ModificationHandler: {e}", exc_info=True)
        else:
            logger.info(
                "ApplicationOrchestrator: ModificationHandler not available or not imported, skipping instantiation.")

        self.user_input_processor_instance: Optional[UserInputProcessor] = None
        if self.rag_handler:
            try:
                self.user_input_processor_instance = UserInputProcessor(
                    self.rag_handler,
                    self.modification_handler_instance
                )
                logger.debug("UserInputProcessor instantiated.")
            except Exception as e:
                logger.critical(f"ApplicationOrchestrator: Failed to instantiate UserInputProcessor: {e}",
                                exc_info=True)
        else:
            logger.critical("ApplicationOrchestrator: Cannot instantiate UserInputProcessor, RagHandler missing.")

        self.modification_coordinator: Optional[ModificationCoordinator] = None
        if MOD_COORDINATOR_AVAILABLE and ModificationCoordinator is not None and \
                self.modification_handler_instance and self.backend_coordinator and self.project_context_manager:
            try:
                self.modification_coordinator = ModificationCoordinator(
                    modification_handler=self.modification_handler_instance,
                    backend_coordinator=self.backend_coordinator,
                    project_context_manager=self.project_context_manager
                )
                logger.debug("ModificationCoordinator instantiated.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ModificationCoordinator: {e}",
                             exc_info=True)
        else:
            logger.warning(
                "ApplicationOrchestrator: ModificationCoordinator cannot be instantiated (dependencies missing or import failed).")

        self.session_flow_manager: Optional[SessionFlowManager] = None
        if self._session_service and self.project_context_manager and self.backend_coordinator:
            self.session_flow_manager = SessionFlowManager(
                session_service=self._session_service,
                project_context_manager=self.project_context_manager,
                backend_coordinator=self.backend_coordinator
            )
            logger.debug("SessionFlowManager instantiated.")
        else:
            logger.critical(
                "ApplicationOrchestrator: SessionFlowManager could not be initialized due to missing dependencies.")

        # --- Instantiate ProjectIntelligenceService (before PSC) ---
        self.project_intelligence_service: Optional[ProjectIntelligenceService] = None
        if PROJECT_INTEL_SERVICE_AVAILABLE and ProjectIntelligenceService is not None and self._vector_db_service:
            try:
                self.project_intelligence_service = ProjectIntelligenceService(
                    vector_db_service=self._vector_db_service)
                logger.debug("ProjectIntelligenceService instantiated.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ProjectIntelligenceService: {e}",
                             exc_info=True)
        else:
            logger.warning(
                "ApplicationOrchestrator: ProjectIntelligenceService cannot be instantiated (VectorDBService or import failed).")

        # --- Instantiate ProjectSummaryCoordinator (before UIH and UC) ---
        self.project_summary_coordinator: Optional[ProjectSummaryCoordinator] = None
        if PROJECT_SUMMARY_COORDINATOR_AVAILABLE and ProjectSummaryCoordinator is not None and \
                self.project_intelligence_service and self.backend_coordinator and self.project_context_manager:
            try:
                self.project_summary_coordinator = ProjectSummaryCoordinator(
                    project_intelligence_service=self.project_intelligence_service,
                    backend_coordinator=self.backend_coordinator,
                    project_context_manager=self.project_context_manager
                )
                logger.debug("ProjectSummaryCoordinator instantiated.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ProjectSummaryCoordinator: {e}",
                             exc_info=True)
        else:
            logger.warning(
                "ApplicationOrchestrator: ProjectSummaryCoordinator cannot be instantiated (dependencies or import failed).")

        # --- Instantiate UploadCoordinator (Pass PSC) ---
        self.upload_coordinator: Optional[UploadCoordinator] = None
        if self._upload_service and self.project_context_manager:  # Use the instance var for upload_service
            try:
                self.upload_coordinator = UploadCoordinator(
                    upload_service=self._upload_service,
                    project_context_manager=self.project_context_manager,
                    project_summary_coordinator=self.project_summary_coordinator  # Pass PSC
                )
                logger.debug("UploadCoordinator instantiated with ProjectSummaryCoordinator.")
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate UploadCoordinator: {e}", exc_info=True)
        else:
            logger.error(
                "ApplicationOrchestrator: Cannot initialize UploadCoordinator (UploadService or ProjectContextManager missing).")

        # --- Instantiate UserInputHandler (Pass PSC) ---
        self.user_input_handler: Optional[UserInputHandler] = None
        if self.user_input_processor_instance and self.project_context_manager:
            try:
                self.user_input_handler = UserInputHandler(
                    user_input_processor=self.user_input_processor_instance,
                    project_context_manager=self.project_context_manager,
                    modification_coordinator=self.modification_coordinator,
                    project_summary_coordinator=self.project_summary_coordinator  # Pass PSC
                )
                logger.debug("UserInputHandler instantiated with ProjectSummaryCoordinator.")
            except Exception as e:
                logger.critical(f"ApplicationOrchestrator: Failed to initialize UserInputHandler: {e}", exc_info=True)
        else:
            logger.critical(
                "ApplicationOrchestrator: UserInputHandler cannot be initialized (UserInputProcessor or ProjectContextManager missing).")

        logger.info("ApplicationOrchestrator core components instantiation process complete.")

    # --- Public Getters (no changes needed to the getters themselves) ---
    def get_all_backend_adapters_dict(self) -> Dict[str, BackendInterface]:
        return self._all_backend_adapters_dict

    def get_project_context_manager(self) -> ProjectContextManager:
        if self.project_context_manager is None:
            logger.critical("get_project_context_manager called but instance is None!")
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

    def get_project_summary_coordinator(self) -> Optional[ProjectSummaryCoordinator]:
        return self.project_summary_coordinator

    def get_rag_handler(self) -> Optional[RagHandler]:
        return self.rag_handler

    def get_modification_handler_instance(self) -> Optional[ModificationHandler]:
        return self.modification_handler_instance

    def get_user_input_processor_instance(self) -> Optional[UserInputProcessor]:
        return self.user_input_processor_instance

    def get_project_intelligence_service(self) -> Optional[ProjectIntelligenceService]:
        return self.project_intelligence_service