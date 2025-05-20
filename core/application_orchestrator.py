# core/application_orchestrator.py
import logging
from typing import Dict, Optional

from backend.interface import BackendInterface
from backend.gemini_adapter import GeminiAdapter
from backend.ollama_adapter import OllamaAdapter
from backend.gpt_adapter import GPTAdapter

from utils.constants import (
    DEFAULT_CHAT_BACKEND_ID,
    OLLAMA_CHAT_BACKEND_ID,
    GPT_CHAT_BACKEND_ID,
    PLANNER_BACKEND_ID,
    GENERATOR_BACKEND_ID
)

from core.project_context_manager import ProjectContextManager
from core.backend_coordinator import BackendCoordinator
from core.session_flow_manager import SessionFlowManager
from core.upload_coordinator import UploadCoordinator
from core.rag_handler import RagHandler
from core.user_input_processor import UserInputProcessor
from core.user_input_handler import UserInputHandler

try:
    from core.modification_handler import ModificationHandler

    MOD_HANDLER_AVAILABLE = True
except ImportError as e:
    ModificationHandler = None
    MOD_HANDLER_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ModificationHandler: {e}.")

try:
    from core.modification_coordinator import ModificationCoordinator, ModPhase

    MOD_COORDINATOR_AVAILABLE = True
except ImportError as e:
    ModificationCoordinator = None
    ModPhase = None
    MOD_COORDINATOR_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ModificationCoordinator or ModPhase: {e}.")

try:
    from services.project_intelligence_service import ProjectIntelligenceService

    PROJECT_INTEL_SERVICE_AVAILABLE = True
except ImportError as e:
    ProjectIntelligenceService = None
    PROJECT_INTEL_SERVICE_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ProjectIntelligenceService: {e}.")

try:
    from core.project_summary_coordinator import ProjectSummaryCoordinator

    PROJECT_SUMMARY_COORDINATOR_AVAILABLE = True
except ImportError as e:
    ProjectSummaryCoordinator = None
    PROJECT_SUMMARY_COORDINATOR_AVAILABLE = False
    logging.error(f"ApplicationOrchestrator: Failed to import ProjectSummaryCoordinator: {e}.")

from services.session_service import SessionService
from services.upload_service import UploadService
from services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


class ApplicationOrchestrator:
    def __init__(self, session_service: SessionService, upload_service: UploadService):
        logger.info("ApplicationOrchestrator initializing...")
        self._session_service = session_service
        self._upload_service = upload_service
        self._vector_db_service = getattr(upload_service, '_vector_db_service', None)
        if not isinstance(self._vector_db_service, VectorDBService):
            self._vector_db_service = None
            logger.warning("ApplicationOrchestrator: VectorDBService instance not available from UploadService!")

        self.gemini_chat_default_adapter = GeminiAdapter()
        self.ollama_chat_adapter = OllamaAdapter()
        self.gpt_chat_adapter = GPTAdapter()
        self.gemini_planner_adapter = GeminiAdapter()
        self.ollama_generator_adapter: BackendInterface = self.ollama_chat_adapter

        self._all_backend_adapters_dict: Dict[str, BackendInterface] = {
            DEFAULT_CHAT_BACKEND_ID: self.gemini_chat_default_adapter,
            OLLAMA_CHAT_BACKEND_ID: self.ollama_chat_adapter,
            GPT_CHAT_BACKEND_ID: self.gpt_chat_adapter,
            PLANNER_BACKEND_ID: self.gemini_planner_adapter,
            GENERATOR_BACKEND_ID: self.ollama_generator_adapter,
        }

        self.project_context_manager = ProjectContextManager()

        self.backend_coordinator = BackendCoordinator(self._all_backend_adapters_dict)

        self.rag_handler: Optional[RagHandler] = None
        if self._upload_service and self._vector_db_service:
            self.rag_handler = RagHandler(self._upload_service, self._vector_db_service)
        else:
            logger.warning(
                "ApplicationOrchestrator: RagHandler cannot be instantiated (UploadService or VectorDBService missing).")

        self.modification_handler_instance: Optional[ModificationHandler] = None
        if MOD_HANDLER_AVAILABLE and ModificationHandler is not None:
            try:
                self.modification_handler_instance = ModificationHandler()
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
            except Exception as e:
                logger.critical(f"ApplicationOrchestrator: Failed to instantiate UserInputProcessor: {e}",
                                exc_info=True)
        else:
            logger.critical("ApplicationOrchestrator: Cannot instantiate UserInputProcessor, RagHandler missing.")

        self.modification_coordinator: Optional[ModificationCoordinator] = None
        if MOD_COORDINATOR_AVAILABLE and ModificationCoordinator is not None and \
                self.modification_handler_instance and self.backend_coordinator and \
                self.project_context_manager and self.rag_handler:
            try:
                self.modification_coordinator = ModificationCoordinator(
                    modification_handler=self.modification_handler_instance,
                    backend_coordinator=self.backend_coordinator,
                    project_context_manager=self.project_context_manager,
                    rag_handler=self.rag_handler
                )
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
        else:
            logger.critical(
                "ApplicationOrchestrator: SessionFlowManager could not be initialized due to missing dependencies.")

        self.project_intelligence_service: Optional[ProjectIntelligenceService] = None
        if PROJECT_INTEL_SERVICE_AVAILABLE and ProjectIntelligenceService is not None and self._vector_db_service:
            try:
                self.project_intelligence_service = ProjectIntelligenceService(
                    vector_db_service=self._vector_db_service)
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ProjectIntelligenceService: {e}",
                             exc_info=True)
        else:
            logger.warning(
                "ApplicationOrchestrator: ProjectIntelligenceService cannot be instantiated (VectorDBService or import failed).")

        self.project_summary_coordinator: Optional[ProjectSummaryCoordinator] = None
        if PROJECT_SUMMARY_COORDINATOR_AVAILABLE and ProjectSummaryCoordinator is not None and \
                self.project_intelligence_service and self.backend_coordinator and self.project_context_manager:
            try:
                self.project_summary_coordinator = ProjectSummaryCoordinator(
                    project_intelligence_service=self.project_intelligence_service,
                    backend_coordinator=self.backend_coordinator,
                    project_context_manager=self.project_context_manager
                )
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate ProjectSummaryCoordinator: {e}",
                             exc_info=True)
        else:
            logger.warning(
                "ApplicationOrchestrator: ProjectSummaryCoordinator cannot be instantiated (dependencies or import failed).")

        self.upload_coordinator: Optional[UploadCoordinator] = None
        if self._upload_service and self.project_context_manager:
            try:
                self.upload_coordinator = UploadCoordinator(
                    upload_service=self._upload_service,
                    project_context_manager=self.project_context_manager,
                    project_summary_coordinator=self.project_summary_coordinator
                )
            except Exception as e:
                logger.error(f"ApplicationOrchestrator: Failed to instantiate UploadCoordinator: {e}", exc_info=True)
        else:
            logger.error(
                "ApplicationOrchestrator: Cannot initialize UploadCoordinator (UploadService or ProjectContextManager missing).")

        self.user_input_handler: Optional[UserInputHandler] = None
        if self.user_input_processor_instance and self.project_context_manager:
            try:
                self.user_input_handler = UserInputHandler(
                    user_input_processor=self.user_input_processor_instance,
                    project_context_manager=self.project_context_manager,
                    modification_coordinator=self.modification_coordinator,
                    project_summary_coordinator=self.project_summary_coordinator
                )
            except Exception as e:
                logger.critical(f"ApplicationOrchestrator: Failed to initialize UserInputHandler: {e}", exc_info=True)
        else:
            logger.critical(
                "ApplicationOrchestrator: UserInputHandler cannot be initialized (UserInputProcessor or ProjectContextManager missing).")

        logger.info("ApplicationOrchestrator core components instantiation process complete.")

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