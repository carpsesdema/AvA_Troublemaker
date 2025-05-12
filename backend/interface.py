# SynChat/backend/interface.py
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple  # Dict, Any already here

from core.models import ChatMessage

class BackendInterface(ABC):
    """Abstract Base Class defining the interface for AI backend communication."""

    @abstractmethod
    def configure(self, api_key: Optional[str], model_name: str, system_prompt: Optional[str] = None) -> bool:
        """
        Configures the backend adapter with necessary credentials and settings.
        """
        pass

    # --- MODIFIED SIGNATURE ---
    @abstractmethod
    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Gets a streaming response from the AI backend based on the provided history.

        Args:
            history: A list of ChatMessage objects representing the conversation history.
            options: An optional dictionary for backend-specific parameters like temperature.

        Yields:
            str: Chunks of the generated response text as they become available.

        Raises:
            Exception: If an error occurs during the API call or streaming.
        """
        # This is abstract, so implementations must define it.
        # The type hint ensures it's treated as an async generator.
        if False: # This code is never executed, it's just for type hinting correctness
             yield ''
        pass
    # --- END MODIFIED SIGNATURE ---

    @abstractmethod
    def get_last_error(self) -> Optional[str]:
        """
        Returns the last error message encountered by the adapter, if any.
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Checks if the backend adapter is currently configured and ready.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Returns a list of model names available through this backend.
        """
        pass

    # --- NEW (from previous adapter updates, already present in your combined.txt) ---
    # This was already added to your combined.txt for the adapters, but good to ensure it's here.
    @abstractmethod
    def get_last_token_usage(self) -> Optional[Tuple[int, int]]: # Added Tuple
        """
        Returns the token usage (prompt_tokens, completion_tokens) from the last call.
        Returns None if not available or not applicable.
        """
        pass
    # --- END NEW ---