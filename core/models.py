# SynChat/core/models.py
# UPDATED FILE - Modified ChatMessage.parts to support images

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Dict, Any # Added Union, Dict, Any

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
    """
    role: str # e.g., USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    # Parts can be strings (text) or dictionaries (images)
    parts: List[Union[str, Dict[str, Any]]]
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    # Optional: Add metadata if needed (e.g., message_id, source_file)
    metadata: Optional[dict] = None

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