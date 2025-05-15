# SynChat/backend/ollama_adapter.py
import logging
import asyncio
import base64
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple
import time

try:
    import ollama

    # If Pydantic is used by ollama lib, Model might be a Pydantic model
    # For type hinting, we can try importing it or use a general approach
    try:
        # Attempt to import the specific type if the library exposes it cleanly.
        # This is often named Model or similar within the library's types module.
        from ollama._types import Model as OllamaModelType  # Example path, adjust if known
    except ImportError:
        # Fallback if the specific type is not easily importable or its path is unknown.
        # Using 'Any' or 'object' is a safe bet for type hinting in such cases.
        # Or, if you know it's dict-like, 'dict' could be used.
        OllamaModelType = Any  # type: ignore
    API_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
    OllamaModelType = Any  # type: ignore
    API_LIBRARY_AVAILABLE = False
    logging.warning("OllamaAdapter: 'ollama' library not found. Please install it: pip install ollama")

from .interface import BackendInterface
from core.models import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _run_ollama_stream_sync(client, model_name, messages, options: Optional[Dict[str, Any]] = None) -> List[
    Dict[str, Any]]:
    all_chunks = []
    try:
        logger.debug(f"[Thread {time.time():.2f}] Calling ollama.chat (sync within thread) with options: {options}...")
        stream = client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options=options
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
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llava:latest"  # This default might be more for a general purpose Ollama model

    def __init__(self):
        super().__init__()
        self._sync_client: Optional[ollama.Client] = None
        self._model_name: str = self.DEFAULT_MODEL
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._ollama_host: str = self.DEFAULT_OLLAMA_HOST
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("OllamaAdapter initialized.")

    def configure(self, api_key: Optional[str], model_name: Optional[str], system_prompt: Optional[str] = None) -> bool:
        logger.info(
            f"OllamaAdapter: Configuring. Host: {self._ollama_host}, Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._sync_client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library ('ollama') not installed."
            logger.error(self._last_error)
            return False

        self._model_name = model_name if model_name else self.DEFAULT_MODEL
        self._system_prompt = system_prompt.strip() if isinstance(system_prompt, str) else None

        try:
            self._sync_client = ollama.Client(host=self._ollama_host)
            # Test connection by listing models. This also pre-warms the client.
            try:
                self._sync_client.list()  # This call can throw if server is down
                logger.info(f"  Successfully connected to Ollama at {self._ollama_host}.")
            except Exception as conn_err:
                self._last_error = f"Failed to connect to Ollama at {self._ollama_host}: {conn_err}"
                logger.error(self._last_error)
                self._sync_client = None  # Nullify client on connection error
                return False  # Configuration fails if cannot connect

            self._is_configured = True
            logger.info(
                f"  OllamaAdapter configured successfully for model '{self._model_name}' at {self._ollama_host}.")
            return True
        except Exception as e:
            self._last_error = f"Unexpected error configuring Ollama client: {type(e).__name__} - {e}"
            logger.exception(f"OllamaAdapter Config Error:")
            self._sync_client = None  # Ensure client is null on any configuration error
            return False

    def is_configured(self) -> bool:
        return self._is_configured and self._sync_client is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
            AsyncGenerator[str, None]:
        logger.info(
            f"OllamaAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured():  # Relies on self._sync_client being not None
            self._last_error = "Adapter is not configured."
            logger.error(self._last_error);
            raise RuntimeError(self._last_error)

        messages = self._format_history_for_api(history)
        if not messages:
            self._last_error = "Cannot send request: No valid messages in history for the API format."
            logger.error(self._last_error);
            raise ValueError(self._last_error)

        logger.info(f"  Sending {len(messages)} messages to model '{self._model_name}'.")

        ollama_api_options = {}
        if options and "temperature" in options and isinstance(options["temperature"], (float, int)):
            temp_val = float(options["temperature"])
            ollama_api_options["temperature"] = temp_val  # Ollama client handles actual valid range
            logger.info(f"  Applying temperature from options: {temp_val} to Ollama request.")

        try:
            logger.debug("Calling asyncio.to_thread to run Ollama stream...")
            # Pass the confirmed non-None self._sync_client
            all_chunks = await asyncio.to_thread(
                _run_ollama_stream_sync,
                self._sync_client,  # type: ignore # is_configured() checks this
                self._model_name,
                messages,
                ollama_api_options
            )
            logger.debug(f"asyncio.to_thread completed. Received {len(all_chunks)} chunks.")

            # Process token counts from the final chunk if available
            if all_chunks:
                final_chunk = all_chunks[-1]
                if final_chunk.get('done', False) and not final_chunk.get('error'):
                    self._last_prompt_tokens = final_chunk.get('prompt_eval_count')
                    self._last_completion_tokens = final_chunk.get('eval_count')
                    logger.info(
                        f"  Ollama Token Usage: Prompt={self._last_prompt_tokens}, Completion={self._last_completion_tokens}")
                elif final_chunk.get('error'):
                    logger.warning(
                        f"  Ollama final chunk reported an error: {final_chunk.get('error')}. Token counts might be unavailable.")
                else:
                    logger.warning(
                        "  Ollama final chunk 'done' flag not true or missing. Token counts may be unavailable.")

            # Yield content from chunks
            for chunk in all_chunks:
                if chunk.get("error"):
                    self._last_error = chunk["error"]
                    logger.error(f"Error received from Ollama thread: {self._last_error}")
                    yield f"[SYSTEM ERROR: {self._last_error}]";
                    break  # Stop yielding on error
                content_part = chunk.get('message', {}).get('content', '')
                if content_part: yield content_part
                if chunk.get('done', False):
                    logger.info("Ollama stream finished flag received in collected chunks.");
                    break  # Stop if done flag is in an intermediate chunk (should be last)
        except ollama.ResponseError as e:  # Specific error from ollama client
            self._last_error = f"Ollama API Response Error: {e.status_code} - {e.error}"  # type: ignore
            logger.error(self._last_error);
            raise RuntimeError(self._last_error) from e
        except Exception as e:  # General errors
            self._last_error = f"Unexpected error during Ollama stream processing: {type(e).__name__} - {e}"
            logger.exception("OllamaAdapter stream failed:");
            raise RuntimeError(self._last_error) from e

    def get_available_models(self) -> List[str]:
        if not self.is_configured() or not self._sync_client:  # self._sync_client check is redundant due to is_configured
            logger.warning("OllamaAdapter is not configured, cannot list models.")
            return []

        model_names = []
        try:
            logger.debug("Calling self._sync_client.list() to fetch models.")
            models_response_dict = self._sync_client.list()  # type: ignore # self._sync_client is checked by is_configured
            logger.debug(f"Raw response dict from ollama.Client().list(): {models_response_dict}")

            if models_response_dict and 'models' in models_response_dict and \
                    isinstance(models_response_dict['models'], list):

                models_obj_or_dict_list = models_response_dict['models']

                for i, item in enumerate(models_obj_or_dict_list):
                    model_name_to_add = None
                    # --- MODIFICATION START ---
                    # Check if the item is an instance of the ollama Model type (or similar structure)
                    # and try to access its 'model' attribute which contains the model name string.
                    if hasattr(item, 'model') and isinstance(getattr(item, 'model'), str):
                        model_name_to_add = getattr(item, 'model')
                        logger.debug(
                            f"  Extracted model name '{model_name_to_add}' from item attribute 'model' (Type: {type(item)})")
                    # Fallback for plain dictionary (less likely with current ollama client but good for robustness)
                    elif isinstance(item, dict) and 'name' in item and isinstance(item['name'], str):
                        model_name_to_add = item['name']
                        logger.debug(f"  Extracted model name '{model_name_to_add}' from dict key 'name'")
                    elif isinstance(item, dict) and 'model' in item and isinstance(item['model'],
                                                                                   str):  # Added for dicts with 'model' key
                        model_name_to_add = item['model']
                        logger.debug(f"  Extracted model name '{model_name_to_add}' from dict key 'model'")
                    # --- MODIFICATION END ---

                    if model_name_to_add:
                        model_names.append(model_name_to_add)
                    else:
                        # This log will now only appear if the item is truly unexpected
                        # or if it's a type that doesn't have 'model' or 'name' as expected.
                        logger.warning(
                            f"Item {i} in models list is an unexpected format or type, or 'model'/'name' attribute missing/invalid: {item} (Type: {type(item)})")

                logger.info(f"Successfully listed {len(model_names)} models from Ollama.")
            else:
                logger.warning(
                    f"Ollama list() returned unexpected format or empty 'models' list: {models_response_dict}")

            return model_names

        except Exception as e:
            logger.error(f"Error listing models from Ollama: {e}", exc_info=True)  # Added exc_info
            self._last_error = f"Failed to list Ollama models: {type(e).__name__} - {e}"
            return []

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        ollama_messages = []
        skipped_count = 0

        # Add system prompt first if it exists
        if self._system_prompt:
            ollama_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:
            role: Optional[str] = None
            if msg.role == USER_ROLE:
                role = 'user'
            elif msg.role == MODEL_ROLE:
                role = 'assistant'
            # Allow SYSTEM_ROLE messages from history ONLY if no adapter-level system prompt is set,
            # to avoid conflicting system messages. Typically, the adapter's system prompt takes precedence.
            elif msg.role == SYSTEM_ROLE and not self._system_prompt:
                role = 'system'
            elif msg.role in [SYSTEM_ROLE, ERROR_ROLE] and msg.metadata and msg.metadata.get("is_internal"):
                # Skip internal system/error messages that are for UI display only
                skipped_count += 1;
                continue
            else:
                logger.warning(f"Skipping message with unhandled role '{msg.role}' for Ollama API format.");
                skipped_count += 1;
                continue

            # Ensure content processing
            content_text = msg.text  # msg.text is already a property that joins string parts

            images_base64: List[str] = []
            if msg.has_images:
                for img_part in msg.image_parts:  # image_parts is a property returning List[Dict]
                    img_data = img_part.get("data")
                    if isinstance(img_data, str):
                        try:
                            # Quick check if it's valid base64, though Ollama lib might do more checks.
                            # This is a basic validation.
                            base64.b64decode(img_data, validate=True)
                            images_base64.append(img_data)
                        except Exception:
                            logger.warning(f"Skipping invalid base64 data in message part for role {role}.")
                    else:
                        logger.warning(f"Skipping non-string image data part for role {role}.")

            ollama_msg: Dict[str, Any] = {"role": role}
            # Only add content key if there's actual text
            if content_text.strip():  # Use strip() to avoid sending empty strings as content
                ollama_msg["content"] = content_text

            # Only add images key if there are valid images
            if images_base64:
                ollama_msg["images"] = images_base64

            # Add message to list only if it has content or images
            if "content" in ollama_msg or "images" in ollama_msg:
                ollama_messages.append(ollama_msg)
            elif role == "system" and "content" not in ollama_msg and not images_base64:
                # Allow system messages with no content if that's intended (e.g. only role)
                # but typically a system message has content.
                logger.debug(f"Formatting system message for Ollama with no text content or images (Role: {role}).")
                ollama_messages.append(ollama_msg)  # This might be an empty content system message.
            else:
                skipped_count += 1;
                logger.warning(
                    f"Skipping message with no valid text or image parts for role {role}.")

        if skipped_count > 0:
            logger.debug(
                f"Skipped {skipped_count} messages (e.g. non-user/model, internal, or empty) when formatting for Ollama API.")
        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None