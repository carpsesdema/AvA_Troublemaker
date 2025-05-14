# core/project_summary_coordinator.py
import logging
import uuid
from typing import Optional, List, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

try:
    from services.project_intelligence_service import ProjectIntelligenceService
    from core.backend_coordinator import BackendCoordinator
    from core.project_context_manager import ProjectContextManager
    from core.models import ChatMessage, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
    # Assuming constants might be used for backend IDs if not hardcoded
    from utils import constants
except ImportError as e:
    logging.critical(f"ProjectSummaryCoordinator: Failed to import core components or services: {e}")
    # Define fallbacks for type hinting or basic functionality if imports fail
    ProjectIntelligenceService = type("ProjectIntelligenceService", (object,), {})  # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {})  # type: ignore
    ProjectContextManager = type("ProjectContextManager", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    USER_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "system", "error"  # type: ignore
    constants = type("constants", (object,), {  # type: ignore
        "DEFAULT_CHAT_BACKEND_ID": "gemini_chat_default",
        "GENERATOR_BACKEND_ID": "ollama_generator"
    })

logger = logging.getLogger(__name__)

# Define Backend IDs (can be moved to utils.constants later if not already there)
# These should align with what ApplicationOrchestrator and ChatManager expect.
DEFAULT_CHAT_BACKEND_ID = getattr(constants, "DEFAULT_CHAT_BACKEND_ID", "gemini_chat_default")
GENERATOR_BACKEND_ID = getattr(constants, "GENERATOR_BACKEND_ID",
                               "ollama_generator")  # E.g., CodeLlama for technical summary


class ProjectSummaryCoordinator(QObject):
    """
    Coordinates the generation of a project RAG summary, involving:
    1. Fetching a condensed overview from ProjectIntelligenceService.
    2. Requesting a technical summary from a Generator/Code LLM.
    3. Requesting a friendly presentation of the technical summary from the Chat LLM (Ava).
    """

    # Signals to ChatManager
    summary_generated = pyqtSignal(str, str)  # project_id, friendly_summary_text
    summary_generation_failed = pyqtSignal(str, str)  # project_id, error_message

    # Signal for direct status updates to the UI (via ChatManager if it relays)
    # Or ChatManager can interpret the specific generated/failed signals to show status.
    # Let's use specific signals for now and ChatManager can decide on status messages.
    # status_update = pyqtSignal(str, str, bool, int) # msg, color, temporary, duration_ms

    def __init__(self,
                 project_intelligence_service: ProjectIntelligenceService,
                 backend_coordinator: BackendCoordinator,
                 project_context_manager: ProjectContextManager,  # Not directly used yet, but good for future context
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([project_intelligence_service, backend_coordinator, project_context_manager]):
            err = "ProjectSummaryCoordinator requires valid ProjectIntelligenceService, BackendCoordinator, and ProjectContextManager."
            logger.critical(err)
            raise ValueError(err)

        self._project_intelligence_service = project_intelligence_service
        self._backend_coordinator = backend_coordinator
        self._project_context_manager = project_context_manager  # Store for potential future use

        self._is_active: bool = False
        self._current_project_id: Optional[str] = None
        self._current_request_id_tech_summary: Optional[str] = None
        self._current_request_id_friendly_summary: Optional[str] = None
        self._technical_summary_text: Optional[str] = None
        self._original_condensed_overview: Optional[str] = None

        self._connect_signals()
        logger.info("ProjectSummaryCoordinator initialized.")

    def _connect_signals(self):
        """Connect to signals from BackendCoordinator."""
        if self._backend_coordinator:
            self._backend_coordinator.response_completed.connect(self._handle_backend_response)
            self._backend_coordinator.response_error.connect(self._handle_backend_error)
        else:
            logger.error("ProjectSummaryCoordinator: BackendCoordinator is None, cannot connect signals.")

    def generate_project_summary(self, project_id: str):
        """
        Starts the process of generating a project summary.
        """
        if self._is_active:
            logger.warning(f"ProjectSummaryCoordinator is already active for project '{self._current_project_id}'. "
                           f"Ignoring new request for '{project_id}'.")
            # Optionally, emit a busy signal or specific error
            self.summary_generation_failed.emit(project_id, "Summary generation is already in progress.")
            return

        logger.info(f"ProjectSummaryCoordinator: Starting summary generation for project_id '{project_id}'.")
        self._is_active = True
        self._current_project_id = project_id
        # self.status_update.emit(f"Ava is preparing a project summary for '{project_id}'...", "#61afef", False, 0)

        try:
            condensed_overview = self._project_intelligence_service.get_condensed_rag_overview_for_summarization(
                project_id)
            if not condensed_overview or condensed_overview.startswith("[INFO:") or condensed_overview.startswith(
                    "[Error:"):
                error_msg = f"Could not get RAG overview for '{project_id}'. {condensed_overview}"
                logger.warning(error_msg)
                self.summary_generation_failed.emit(project_id,
                                                    condensed_overview or "Failed to retrieve project overview.")
                self._reset_state()
                return

            self._original_condensed_overview = condensed_overview
        except Exception as e:
            logger.exception(f"Error getting condensed RAG overview for '{project_id}':")
            self.summary_generation_failed.emit(project_id, f"Error retrieving project details: {e}")
            self._reset_state()
            return

        # --- Request Technical Summary ---
        self._current_request_id_tech_summary = f"psc_tech_summary_{project_id}_{uuid.uuid4().hex[:8]}"

        technical_prompt_text = (
            "You are a technical writer AI. Based on the following condensed project RAG overview, "
            "provide a concise technical summary highlighting the project's structure, key components, "
            "and purpose. Focus on technical aspects like main technologies hinted at, primary modules, "
            "and overall architecture if discernible. Output only the summary text itself, without any preamble or conversational filler.\n\n"
            "--- Condensed Project Overview ---\n"
            f"{self._original_condensed_overview}\n"
            "--- End of Overview ---"
        )

        history_for_tech_summary = [ChatMessage(role=USER_ROLE, parts=[technical_prompt_text])]
        tech_summary_options = {"temperature": 0.3}  # Factual summary
        tech_summary_metadata = {
            "purpose": "psc_technical_summary",
            "project_id_for_summary": project_id  # Ensure we know which project this is for
        }

        logger.info(
            f"Requesting technical summary (ReqID: {self._current_request_id_tech_summary}) from '{GENERATOR_BACKEND_ID}'.")
        self._backend_coordinator.request_response_stream(
            target_backend_id=GENERATOR_BACKEND_ID,
            request_id=self._current_request_id_tech_summary,
            history_to_send=history_for_tech_summary,
            is_modification_response_expected=True,  # Internal task, not direct chat
            options=tech_summary_options,
            request_metadata=tech_summary_metadata
        )
        # self.status_update.emit(f"Requesting technical summary from Code AI for '{project_id}'...", "#e5c07b", False, 0)

    @pyqtSlot(str, ChatMessage, dict)
    def _handle_backend_response(self, request_id: str, completed_message: ChatMessage,
                                 usage_stats_with_metadata: dict):
        if not self._is_active:
            logger.warning(f"PSC: Received backend response for ReqID '{request_id}' but not active. Ignoring.")
            return

        purpose = usage_stats_with_metadata.get("purpose")
        project_id_from_meta = usage_stats_with_metadata.get("project_id_for_summary")

        if project_id_from_meta != self._current_project_id:
            logger.warning(
                f"PSC: Received response for ReqID '{request_id}' (Purpose: {purpose}) intended for project '{project_id_from_meta}', "
                f"but current active project in PSC is '{self._current_project_id}'. Ignoring.")
            return

        if request_id == self._current_request_id_tech_summary and purpose == "psc_technical_summary":
            logger.info(f"Technical summary received for project '{self._current_project_id}' (ReqID: {request_id}).")
            self._technical_summary_text = completed_message.text.strip()
            self._current_request_id_tech_summary = None  # Clear this request ID

            if not self._technical_summary_text:
                logger.error(f"Technical summary for '{self._current_project_id}' was empty.")
                self.summary_generation_failed.emit(self._current_project_id,
                                                    "Technical summary generation returned empty.")
                self._reset_state()
                return

            # --- Request Friendly Summary from Ava ---
            self._current_request_id_friendly_summary = f"psc_friendly_summary_{self._current_project_id}_{uuid.uuid4().hex[:8]}"

            friendly_prompt_text = (
                f"Ava, my brilliant AI assistant! I have a technical summary of the current project ('{self._current_project_id}'). "
                "Could you please present this to the user in your wonderfully bubbly, enthusiastic, and helpful style? "
                "Make it easy to understand and engaging! Highlight the key takeaways for someone who might not be deeply technical but wants to know what the project is about.\n\n"
                "Here's the technical summary:\n"
                "--- TECHNICAL SUMMARY START ---\n"
                f"{self._technical_summary_text}\n"
                "--- TECHNICAL SUMMARY END ---\n\n"
                "Remember to be yourself! Start with a friendly greeting and make it shine! ✨"
            )

            history_for_friendly_summary = [ChatMessage(role=USER_ROLE, parts=[friendly_prompt_text])]
            friendly_summary_options = {"temperature": 0.7}  # Ava's default
            friendly_summary_metadata = {
                "purpose": "psc_friendly_summary",
                "project_id_for_summary": self._current_project_id
            }

            logger.info(
                f"Requesting friendly summary (ReqID: {self._current_request_id_friendly_summary}) from '{DEFAULT_CHAT_BACKEND_ID}'.")
            self._backend_coordinator.request_response_stream(
                target_backend_id=DEFAULT_CHAT_BACKEND_ID,
                request_id=self._current_request_id_friendly_summary,
                history_to_send=history_for_friendly_summary,
                is_modification_response_expected=True,  # Internal task
                options=friendly_summary_options,
                request_metadata=friendly_summary_metadata
            )
            # self.status_update.emit(f"Asking Ava to present the summary for '{self._current_project_id}'...", "#e5c07b", False, 0)

        elif request_id == self._current_request_id_friendly_summary and purpose == "psc_friendly_summary":
            logger.info(f"Friendly project summary received for '{self._current_project_id}' (ReqID: {request_id}).")
            friendly_summary = completed_message.text.strip()
            if not friendly_summary:
                logger.error(
                    f"Friendly summary for '{self._current_project_id}' was empty. Falling back to technical summary.")
                # Fallback: emit the technical summary if friendly one is empty
                fallback_summary = f"**Technical Project Summary for {self._current_project_id}:**\n\n{self._technical_summary_text}"
                self.summary_generated.emit(self._current_project_id, fallback_summary)
            else:
                self.summary_generated.emit(self._current_project_id, friendly_summary)

            # self.status_update.emit(f"Project summary for '{self._current_project_id}' is ready!", "#98c379", True, 4000)
            self._reset_state()

        else:
            logger.warning(
                f"PSC: Received unexpected backend response for ReqID '{request_id}', Purpose: '{purpose}'. Ignoring.")

    @pyqtSlot(str, str)
    def _handle_backend_error(self, request_id: str, error_message_str: str):
        if not self._is_active:
            logger.warning(f"PSC: Received backend error for ReqID '{request_id}' but not active. Ignoring.")
            return

        error_source = ""
        if request_id == self._current_request_id_tech_summary:
            error_source = "technical summary generation"
            logger.error(
                f"Error during {error_source} for project '{self._current_project_id}' (ReqID: {request_id}): {error_message_str}")
        elif request_id == self._current_request_id_friendly_summary:
            error_source = "friendly summary presentation"
            logger.error(
                f"Error during {error_source} for project '{self._current_project_id}' (ReqID: {request_id}): {error_message_str}")
        else:
            # Error for a request_id not currently tracked by this coordinator for an active summary
            logger.warning(
                f"PSC: Received backend error for an unrecognized or completed ReqID '{request_id}'. Error: {error_message_str}. Ignoring.")
            return

        failure_message = f"Error during {error_source}: {error_message_str}"
        self.summary_generation_failed.emit(self._current_project_id or "unknown_project", failure_message)
        # self.status_update.emit(f"Failed to generate project summary: {error_message_str}", "#e06c75", True, 7000)
        self._reset_state()

    def _reset_state(self):
        """Resets the coordinator to its idle state."""
        logger.debug("ProjectSummaryCoordinator resetting state.")
        self._is_active = False
        self._current_project_id = None
        self._current_request_id_tech_summary = None
        self._current_request_id_friendly_summary = None
        self._technical_summary_text = None
        self._original_condensed_overview = None