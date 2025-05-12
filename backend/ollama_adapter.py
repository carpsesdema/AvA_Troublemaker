# SynChat/backend/ollama_adapter.py
import logging
import asyncio
import base64
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple # Added Tuple
import time

try:
    import ollama
    API_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None # type: ignore
    API_LIBRARY_AVAILABLE = False
    logging.warning("OllamaAdapter: 'ollama' library not found. Please install it: pip install ollama")

from .interface import BackendInterface
from core.models import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE

logger = logging.getLogger(__name__)

_SENTINEL = object()

def _run_ollama_stream_sync(client, model_name, messages, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: # Added options
    all_chunks = []
    try:
        logger.debug(f"[Thread {time.time():.2f}] Calling ollama.chat (sync within thread) with options: {options}...")
        stream = client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options=options # <-- PASSING OPTIONS (for temperature)
        )
        logger.debug(f"[Thread {time.time():.2f}] Got stream iterator.")
        for chunk in stream:
            all_chunks.append(chunk)
            if chunk.get('done', False):
                if chunk.get('error'):
                    logger.error(f"[Thread {time.time():.2f}] Error in stream chunk: {chunk['error']}")
                else:
                    logger.debug(f"[Thread {time.time():.2f}] Stream done flag received.")
                break
        logger.debug(f"[Thread {time.time():.2f}] Finished iterating stream. Collected {len(all_chunks)} chunks.")
    except Exception as e:
        logger.exception(f"[Thread {time.time():.2f}] Exception during synchronous Ollama stream processing:")
        all_chunks.append({"error": f"Thread Error: {type(e).__name__} - {e}"})
    return all_chunks


class OllamaAdapter(BackendInterface):
    """Implementation of the BackendInterface for local models via Ollama."""

    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llava:latest" # Default if none specified in config

    def __init__(self):
        super().__init__()
        self._sync_client: Optional[ollama.Client] = None
        self._model_name: str = self.DEFAULT_MODEL
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._ollama_host: str = self.DEFAULT_OLLAMA_HOST
        self._last_prompt_tokens: Optional[int] = None      # <-- NEW: For token count
        self._last_completion_tokens: Optional[int] = None # <-- NEW: For token count
        logger.info("OllamaAdapter initialized.")

    def configure(self, api_key: Optional[str], model_name: Optional[str], system_prompt: Optional[str] = None) -> bool:
        logger.info(f"OllamaAdapter: Configuring. Host: {self._ollama_host}, Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._sync_client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None # Reset on reconfigure
        self._last_completion_tokens = None # Reset on reconfigure


        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library ('ollama') not installed."
            logger.error(self._last_error)
            return False

        self._model_name = model_name if model_name else self.DEFAULT_MODEL
        self._system_prompt = system_prompt.strip() if isinstance(system_prompt, str) else None

        try:
            self._sync_client = ollama.Client(host=self._ollama_host)
            try:
                # Attempt a light operation to confirm connection
                self._sync_client.list() # This also helps ensure the service is responsive
                logger.info(f"  Successfully connected to Ollama at {self._ollama_host}.")
            except Exception as conn_err:
                self._last_error = f"Failed to connect to Ollama at {self._ollama_host}: {conn_err}"
                logger.error(self._last_error)
                return False

            self._is_configured = True
            logger.info(f"  OllamaAdapter configured successfully for model '{self._model_name}' at {self._ollama_host}.")
            return True

        except Exception as e:
            self._last_error = f"Unexpected error configuring Ollama client: {type(e).__name__} - {e}"
            logger.exception(f"OllamaAdapter Config Error:")
            return False

    def is_configured(self) -> bool:
        return self._is_configured and self._sync_client is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    # --- MODIFIED get_response_stream to accept options (for temperature) ---
    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        logger.info(f"OllamaAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None # Reset before new request
        self._last_completion_tokens = None # Reset before new request

        if not self.is_configured():
            self._last_error = "Adapter is not configured."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages = self._format_history_for_api(history)
        if not messages:
            self._last_error = "Cannot send request: No valid messages in history for the API format."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.info(f"  Sending {len(messages)} messages to model '{self._model_name}'.")
        if logger.isEnabledFor(logging.DEBUG):
             for i, msg in enumerate(messages):
                 content_preview = str(msg.get('content', ''))[:100] + ('...' if len(str(msg.get('content', ''))) > 100 else '')
                 images_preview = f", Images: {len(msg.get('images', []))}" if 'images' in msg else ""
                 logger.debug(f"    Message {i}: Role={msg['role']}, Content='{content_preview}'{images_preview}")

        # --- Prepare options for Ollama (temperature) ---
        ollama_api_options = {}
        if options and "temperature" in options and isinstance(options["temperature"], (float, int)):
            temp_val = float(options["temperature"])
            ollama_api_options["temperature"] = temp_val
            logger.info(f"  Applying temperature from options: {temp_val} to Ollama request.")
        # Add other options here if needed (e.g., top_p, top_k)
        # --- End options preparation ---

        try:
           logger.debug("Calling asyncio.to_thread to run Ollama stream...")
           all_chunks = await asyncio.to_thread(
               _run_ollama_stream_sync,
               self._sync_client,
               self._model_name,
               messages,
               ollama_api_options # <-- PASSING OPTIONS
           )
           logger.debug(f"asyncio.to_thread completed. Received {len(all_chunks)} chunks.")

           # --- TOKEN COUNT EXTRACTION ---
           # Ollama typically includes eval_count and prompt_eval_count in the *final* chunk.
           if all_chunks:
               final_chunk = all_chunks[-1]
               if final_chunk.get('done', False) and not final_chunk.get('error'):
                   self._last_prompt_tokens = final_chunk.get('prompt_eval_count')
                   self._last_completion_tokens = final_chunk.get('eval_count')
                   logger.info(f"  Ollama Token Usage: Prompt={self._last_prompt_tokens}, Completion={self._last_completion_tokens}")
               elif final_chunk.get('error'):
                    logger.warning(f"  Ollama final chunk reported an error: {final_chunk.get('error')}. Token counts might be unavailable.")
               else:
                    logger.warning("  Ollama final chunk 'done' flag not true or missing. Token counts may be unavailable.")
           # --- END TOKEN COUNT EXTRACTION ---


           for chunk in all_chunks:
               if chunk.get("error"):
                   self._last_error = chunk["error"]
                   logger.error(f"Error received from Ollama thread: {self._last_error}")
                   # Don't raise immediately, let it yield any content first if applicable,
                   # then BackendCoordinator will handle the error if stream ends abruptly.
                   # If the error implies no further content, this loop might break or yield nothing.
                   yield f"[SYSTEM ERROR: {self._last_error}]" # Yield error to UI
                   break # Stop processing more chunks if there's a definitive error

               content_part = chunk.get('message', {}).get('content', '')
               if content_part:
                   yield content_part

               if chunk.get('done', False):
                   logger.info("Ollama stream finished flag received in collected chunks.")
                   break # Stop if 'done' is true, even if not the last physical chunk

        except ollama.ResponseError as e: # type: ignore
            self._last_error = f"Ollama API Response Error: {e.status_code} - {e.error}"
            logger.error(self._last_error)
            raise RuntimeError(self._last_error) from e
        except Exception as e:
            self._last_error = f"Unexpected error during Ollama stream processing: {type(e).__name__} - {e}"
            logger.exception("OllamaAdapter stream failed:")
            raise RuntimeError(self._last_error) from e

    def get_available_models(self) -> List[str]:
        # ... (this method remains the same) ...
        if not self.is_configured() or not self._sync_client:
            logger.warning("OllamaAdapter is not configured, cannot list models.")
            return []
        model_names = []
        try:
            logger.debug("Calling self._sync_client.list() to fetch models.")
            models_response = self._sync_client.list()
            logger.debug(f"Raw response from ollama.Client().list(): {models_response}")
            if models_response and 'models' in models_response and isinstance(models_response['models'], list):
                models_data = models_response['models']
                for i, m_data_item in enumerate(models_data):
                    # ollama client.list() returns a list of dicts, not model objects directly anymore.
                    if isinstance(m_data_item, dict) and 'name' in m_data_item:
                        model_names.append(m_data_item['name'])
                    else:
                        logger.warning(f"Item {i} in models list is an unexpected format: {m_data_item}")
                logger.info(f"Successfully listed {len(model_names)} models from Ollama.")
                return model_names
            else:
                logger.warning("Ollama list() returned unexpected format or empty list.")
                return []
        except Exception as e:
            logger.error(f"Error listing models from Ollama: {e}")
            self._last_error = f"Failed to list Ollama models: {type(e).__name__} - {e}"
            return []

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        # ... (this method remains the same) ...
        ollama_messages = []
        skipped_count = 0
        if self._system_prompt:
             ollama_messages.append({"role": "system", "content": self._system_prompt})
        for msg in history:
            if msg.role == USER_ROLE: role = 'user'
            elif msg.role == MODEL_ROLE: role = 'assistant'
            elif msg.role in [SYSTEM_ROLE, ERROR_ROLE] and msg.metadata and msg.metadata.get("is_internal"):
                 skipped_count += 1; continue
            else:
                logger.warning(f"Skipping message with unhandled role '{msg.role}' for Ollama API format."); skipped_count += 1; continue
            content = msg.text; images_base64 = []
            if msg.has_images:
                for img_part in msg.image_parts:
                    img_data = img_part.get("data")
                    if isinstance(img_data, str):
                         try: base64.b64decode(img_data); images_base64.append(img_data)
                         except Exception: logger.warning(f"Skipping invalid base64 data in message part for role {role}.")
                    else: logger.warning(f"Skipping non-string image data part for role {role}.")
            ollama_msg: Dict[str, Any] = {"role": role}
            if content.strip(): ollama_msg["content"] = content
            if images_base64: ollama_msg["images"] = images_base64
            if "content" in ollama_msg or "images" in ollama_msg: ollama_messages.append(ollama_msg)
            else: skipped_count += 1; logger.warning(f"Skipping message with no valid text or image parts for role {role}.")
        if skipped_count > 0:
            logger.debug(f"Skipped {skipped_count} messages (non-user/model, internal, or empty) when formatting for Ollama API.")
        return ollama_messages

    # --- NEW: Method to get last token usage ---
    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        """
        Returns the token usage from the last successful call.
        Returns: (prompt_tokens, completion_tokens) or None if not available.
        """
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None
    # --- END NEW METHOD ---