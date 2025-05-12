# core/models.py
# UPDATED FILE - Modified ChatMessage.parts to support images
# UPDATED FILE - Added id and loading_state to ChatMessage for new loading indicator

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
import uuid # Added for generating unique message IDs

# Import the new MessageLoadingState enum
# Assuming message_enums.py will be in the same 'core' directory
try:
    from .message_enums import MessageLoadingState
except ImportError: # Fallback for environments where relative import might not work during dev
    try:
        from message_enums import MessageLoadingState # type: ignore
    except ImportError:
        # Define a dummy enum if the import fails, to allow the rest of the code to run
        # This should be replaced by the actual enum definition in message_enums.py
        from enum import Enum, auto
        class MessageLoadingState(Enum):
            IDLE = auto()
            LOADING = auto()
            COMPLETED = auto()
            ERROR = auto()
        print("WARNING: core.models - Could not import MessageLoadingState enum. Using a dummy definition.")


# Define standard role constants
USER_ROLE = "user"
MODEL_ROLE = "model"
SYSTEM_ROLE = "system" # For internal prompts or info
ERROR_ROLE = "error"   # For displaying errors in UI

@dataclass
class ChatMessage:
    """
    Represents a single message in the conversation.
    Can contain text parts and/or image parts.
    Includes an ID and loading state for AI messages.
    """
    role: str # e.g., USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    # Parts can be strings (text) or dictionaries (images)
    parts: List[Union[str, Dict[str, Any]]]
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    # Optional: Add metadata if needed (e.g., message_id, source_file)
    metadata: Optional[dict] = None

    # --- NEW FIELDS for loading indicator and message tracking ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    loading_state: MessageLoadingState = MessageLoadingState.IDLE
    # --- END NEW FIELDS ---

    # Helper to get combined text content easily
    @property
    def text(self) -> str:
        """Returns the combined text from all string parts."""
        return "".join(part for part in self.parts if isinstance(part, str)).strip()

    # Helper to check if the message contains images
    @property
    def has_images(self) -> bool:
        """Checks if any part is an image dictionary."""
        return any(isinstance(part, dict) and part.get("type") == "image" for part in self.parts)

    # Helper to get image parts
    @property
    def image_parts(self) -> List[Dict[str, Any]]:
        """Returns a list of all image dictionary parts."""
        return [part for part in self.parts if isinstance(part, dict) and part.get("type") == "image"]