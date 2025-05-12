# services/code_summary_service.py
import logging
import uuid # <--- ADDED IMPORT
from typing import Optional, Dict, List

# Assuming ChatMessage and BackendCoordinator are accessible for type hinting
# Adjust paths if necessary based on your project structure
try:
    from core.models import ChatMessage, USER_ROLE
    from core.backend_coordinator import BackendCoordinator
except ImportError as e:
    logging.critical(f"CodeSummaryService: Failed to import core components: {e}")
    # Define fallback types for type hinting if imports fail
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    USER_ROLE = "user" # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {}) # type: ignore


logger = logging.getLogger(__name__)

# --- PROMPT TEMPLATE FOR CODE SUMMARY ---
# This is the same template previously in ChatManager
PLANNER_PROMPT_TEMPLATE_FOR_SUMMARY = (
    "You are AvA, a bubbly, enthusiastic, and incredibly helpful AI assistant!\n"
    "Your Coder AI buddy has just finished crafting some Python code for the file '{target_filename}' based on a set of instructions.\n\n"
    "Here are the instructions the Coder followed:\n"
    "---\n"
    "{coder_instructions}\n"
    "---\n\n"
    "And here's the awesome code the Coder produced for '{target_filename}':\n"
    + "```python\n"
    "{generated_code}\n"
    + "```\n"
    "---\n\n"
    "Your mission is to provide a concise, upbeat summary of the key changes or features implemented in '{target_filename}'.\n"
    "Please highlight what was achieved in relation to the original instructions.\n"
    "Keep it brief, super friendly, and in your signature Ava style! For example: \"Woohoo! Our Coder just worked its magic on '{target_filename}', and here's the scoop: ...\"\n"
)
# --- END PROMPT TEMPLATE ---

# Backend ID for the planner/summarizer AI (should match ChatManager/Orchestrator)
PLANNER_BACKEND_ID = "gemini_planner"

class CodeSummaryService:
    """
    A service dedicated to requesting code summaries from an AI backend.
    """

    def __init__(self):
        logger.info("CodeSummaryService initialized.")
        # This service is stateless for now, no complex __init__ needed.

    def request_code_summary(self,
                             backend_coordinator: BackendCoordinator,
                             target_filename: str,
                             coder_instructions: str,
                             generated_code: str) -> bool:
        """
        Requests a code summary from the planner AI via the BackendCoordinator.

        Args:
            backend_coordinator: The BackendCoordinator instance to use for the LLM call.
            target_filename: The name of the file that was generated/modified.
            coder_instructions: The instructions given to the AI that produced the code.
            generated_code: The actual code that was generated.

        Returns:
            True if the request was successfully dispatched, False otherwise.
        """
        logger.info(f"CodeSummaryService: Requesting summary for '{target_filename}'.")

        if not isinstance(backend_coordinator, BackendCoordinator):
            logger.error("CodeSummaryService: Invalid BackendCoordinator instance provided.")
            return False

        if not backend_coordinator.is_backend_configured(PLANNER_BACKEND_ID):
            err_msg = f"Planner AI ('{PLANNER_BACKEND_ID}') is not configured. Cannot generate code summary for '{target_filename}'."
            logger.error(err_msg)
            return False

        try:
            summary_prompt = PLANNER_PROMPT_TEMPLATE_FOR_SUMMARY.format(
                target_filename=target_filename,
                coder_instructions=coder_instructions,
                generated_code=generated_code
            )
        except KeyError as e_fmt:
            logger.error(f"Error formatting summary prompt for '{target_filename}': Missing key {e_fmt}", exc_info=True)
            return False
        except Exception as e_fmt_general:
            logger.exception(f"Unexpected error formatting summary prompt for '{target_filename}': {e_fmt_general}")
            return False

        history_for_summary: List[ChatMessage] = [ChatMessage(role=USER_ROLE, parts=[summary_prompt])]
        planner_options: Dict[str, float] = {"temperature": 0.6}

        # --- MODIFICATION: Generate and include request_id ---
        summary_request_id = f"summary_{target_filename.replace('/', '_')}_{str(uuid.uuid4())[:8]}"
        # --- END MODIFICATION ---

        summary_request_metadata: Dict[str, str] = {
            "purpose": "code_summary",
            "original_target_filename": target_filename,
            # Optionally include the summary_request_id in metadata if needed for tracking by ChatManager
            # "summary_request_id": summary_request_id
        }

        logger.debug(f"Dispatching summary request for '{target_filename}' (request_id: {summary_request_id}) to BackendCoordinator.")
        try:
            # --- MODIFICATION: Update call to include request_id ---
            backend_coordinator.request_response_stream(
                target_backend_id=PLANNER_BACKEND_ID,
                request_id=summary_request_id, # <-- PASSING THE NEW request_id
                history_to_send=history_for_summary,
                is_modification_response_expected=True,
                options=planner_options,
                request_metadata=summary_request_metadata
            )
            # --- END MODIFICATION ---
            logger.info(f"Summary request for '{target_filename}' dispatched successfully.")
            return True
        except Exception as e_dispatch:
            logger.exception(f"Error dispatching summary request via BackendCoordinator for '{target_filename}': {e_dispatch}")
            return False