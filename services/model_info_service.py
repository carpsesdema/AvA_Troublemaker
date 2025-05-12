# services/model_info_service.py
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ModelInfoService:
    """
    Provides information about various LLM models, starting with max token context.
    """

    def __init__(self):
        logger.info("ModelInfoService initialized.")
        # This service can be stateless for now, or load model data from a config file in the future.

    def get_max_tokens(self, model_name_str: Optional[str]) -> int:
        """
        Returns the approximate maximum context tokens for a given model name.
        This logic was previously in ChatManager._get_model_max_tokens().
        """
        if not model_name_str:
            return 0

        model_name_lower = model_name_str.lower()

        # Gemini models
        if "gemini-1.5-pro" in model_name_lower or "gemini-2.5-pro" in model_name_lower:
            return 1_048_576  # 1M tokens (often 2M available via API, but 1M is a common default window)
        if "gemini-1.5-flash" in model_name_lower:
            return 1_048_576  # 1M tokens
        if "gemini-1.0-pro" in model_name_lower or "gemini-pro" in model_name_lower: # Standard gemini-pro
            return 32_768  # 30720 input, 2048 output = 32k total
        # Specific preview versions that might have smaller contexts
        if "gemini-2.5-pro-preview-05-06" in model_name_lower: # Example, check actual limits
            return 8192 # Often previews start with smaller context windows
        if "gemini-2.5-flash-preview-04-17" in model_name_lower: # Example
            return 8192

        # Ollama models (these are typical, actual may vary by specific quant/variant)
        if "codellama:34b" in model_name_lower: return 16_384
        if "codellama:13b" in model_name_lower: return 16_384
        if "codellama:7b" in model_name_lower: return 16_384 # Can be 4k or 16k depending on variant
        if "codellama" in model_name_lower: return 4096 # Default for smaller CodeLlamas or if unsure

        if "llama3:70b" in model_name_lower: return 8192
        if "llama3:8b" in model_name_lower: return 8192
        if "llama3" in model_name_lower: return 8192 # Default for Llama3

        if "llava" in model_name_lower: return 4096 # Often smaller for multimodal

        if "mistral" in model_name_lower: return 8192 # Common for 7B Mistral (can be 32k with sliding window)
        if "mixtral" in model_name_lower: return 32_768 # Mixtral 8x7B

        # Default if model is not recognized
        logger.warning(f"Max token context for model '{model_name_str}' unknown in ModelInfoService. Defaulting to 0.")
        return 0

    # --- Potential future methods ---
    # def get_model_capabilities(self, model_name: str) -> List[str]:
    #     # e.g., ["text", "image_input", "tool_use"]
    #     pass
    #
    # def get_tokens_per_minute_limit(self, model_name: str) -> Optional[int]:
    #     pass