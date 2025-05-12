# core/message_enums.py
from enum import Enum, auto

class MessageLoadingState(Enum):
    """
    Represents the loading state of an AI-generated message.
    """
    IDLE = auto()      # Default state, or message is fully processed and no specific indicator needed.
                       # For AI messages, this means the message is finalized and not actively being generated.
    LOADING = auto()   # AI is actively generating this message or about to start.
    COMPLETED = auto() # AI finished generating this message successfully. The static "done" icon will be shown.
    ERROR = auto()     # An error occurred while generating this message (optional, for specific error indication).
                       # Could show a different static icon if desired.