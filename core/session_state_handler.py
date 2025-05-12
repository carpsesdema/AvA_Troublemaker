# core/session_state_handler.py
# UPDATED - Adapts to ProjectContextManager's state dictionary format.
#         - load_last_session_state now expects/returns project_context_data dict.
#         - save_current_state now accepts/passes project_context_data dict.

import logging
from typing import List, Optional, Tuple, Dict, Any # Added Any

# Assuming services are in the parent directory 'services' relative to 'core'
try:
    from services.session_service import SessionService
    from core.models import ChatMessage # Keep for type hinting within SessionService if it still uses it directly
except ImportError as e:
    logging.critical(f"SessionStateHandler: Failed to import services/models: {e}")
    SessionService = type("SessionService", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})

logger = logging.getLogger(__name__)

class SessionStateHandler:
    """
    Handles loading/saving the current session state via SessionService.
    Now works with a consolidated project_context_data dictionary.
    """

    def __init__(self, session_service: SessionService):
        if not isinstance(session_service, SessionService):
            raise TypeError("SessionStateHandler requires a valid SessionService instance.")
        self._session_service = session_service
        logger.info("SessionStateHandler initialized.")

    def load_last_session_state(self) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Loads state from the last session file.
        SessionService is expected to return a project_context_data dictionary.

        Returns:
            A tuple: (model_name, personality_prompt, project_context_data_dict)
            project_context_data_dict is None if loading fails or no session exists.
        """
        try:
            # <<< CHANGE: Expect project_context_data from SessionService >>>
            # SessionService.get_last_session() will now return:
            # (model_name, personality, project_context_data_dict)
            model, personality, project_context_data = self._session_service.get_last_session()

            if project_context_data is not None:
                logger.info(f"SessionStateHandler loaded state. Model: {model}, Pers: {'Set' if personality else 'None'}, ProjectContextData: {'Present' if project_context_data else 'None'}")
            else:
                logger.info(f"SessionStateHandler: No last session data found or error during load. Model: {model}, Pers: {'Set' if personality else 'None'}")
            return model, personality, project_context_data
        except Exception as e:
            logger.exception("Error loading last session state via SessionStateHandler:")
            # Return default empty state on error
            return None, None, None

    def save_current_state(self,
                           model_name: Optional[str],
                           personality: Optional[str],
                           project_context_data: Dict[str, Any]): # <<< CHANGE: Accept project_context_data dict >>>
        """
        Saves the current state to the last session file.
        Passes the project_context_data dictionary (from ProjectContextManager.save_state())
        to the SessionService.
        """
        try:
            # <<< CHANGE: Pass project_context_data to SessionService >>>
            logger.debug(f"SessionStateHandler saving state. Model: {model_name}, Pers: {'Set' if personality else 'None'}, ProjectContextData: {'Provided' if project_context_data else 'Not provided'}")
            self._session_service.save_last_session(
                model_name=model_name,
                personality=personality,
                project_context_data=project_context_data # Pass the actual dict
            )
            logger.debug("Saved current state via SessionStateHandler.")
        except Exception as e:
            logger.exception("Error saving current session state via SessionStateHandler:")
            # Decide if error should be propagated