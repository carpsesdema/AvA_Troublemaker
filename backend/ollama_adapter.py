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
        from ollama_types import Model as OllamaModelType  # Assuming ollama might expose its types
    except ImportError:
        OllamaModelType = dict  # Fallback if Model type is not easily importable
    API_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None
    OllamaModelType = dict
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
    DEFAULT_MODEL = "llava:latest"

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
            try:
                self._sync_client.list()
                logger.info(f"  Successfully connected to Ollama at {self._ollama_host}.")
            except Exception as conn_err:
                self._last_error = f"Failed to connect to Ollama at {self._ollama_host}: {conn_err}"
                logger.error(self._last_error)
                return False

            self._is_configured = True
            logger.info(
                f"  OllamaAdapter configured successfully for model '{self._model_name}' at {self._ollama_host}.")
            return True
        except Exception as e:
            self._last_error = f"Unexpected error configuring Ollama client: {type(e).__name__} - {e}"
            logger.exception(f"OllamaAdapter Config Error:")
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

        if not self.is_configured():
            self._last_error = "Adapter is not configured."
            logger.error(self._last_error);
            raise RuntimeError(self._last_error)

        messages = self._format_history_for_api(history)
        if not messages:
            self._last_error = "Cannot send request: No valid messages in history for the API format."
            logger.error(self._last_error);
            raise ValueError(self._last_error)

        logger.info(f"  Sending {len(messages)} messages to model '{self._model_name}'.")
        # ... (debug logging for messages remains same) ...

        ollama_api_options = {}
        if options and "temperature" in options and isinstance(options["temperature"], (float, int)):
            temp_val = float(options["temperature"])
            ollama_api_options["temperature"] = temp_val
            logger.info(f"  Applying temperature from options: {temp_val} to Ollama request.")

        try:
            logger.debug("Calling asyncio.to_thread to run Ollama stream...")
            all_chunks = await asyncio.to_thread(
                _run_ollama_stream_sync,
                self._sync_client,
                self._model_name,
                messages,
                ollama_api_options
            )
            logger.debug(f"asyncio.to_thread completed. Received {len(all_chunks)} chunks.")

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

            for chunk in all_chunks:
                if chunk.get("error"):
                    self._last_error = chunk["error"]
                    logger.error(f"Error received from Ollama thread: {self._last_error}")
                    yield f"[SYSTEM ERROR: {self._last_error}]";
                    break
                content_part = chunk.get('message', {}).get('content', '')
                if content_part: yield content_part
                if chunk.get('done', False):
                    logger.info("Ollama stream finished flag received in collected chunks.");
                    break
        except ollama.ResponseError as e:
            self._last_error = f"Ollama API Response Error: {e.status_code} - {e.error}"  # type: ignore
            logger.error(self._last_error);
            raise RuntimeError(self._last_error) from e
        except Exception as e:
            self._last_error = f"Unexpected error during Ollama stream processing: {type(e).__name__} - {e}"
            logger.exception("OllamaAdapter stream failed:");
            raise RuntimeError(self._last_error) from e

    def get_available_models(self) -> List[str]:
        if not self.is_configured() or not self._sync_client:
            logger.warning("OllamaAdapter is not configured, cannot list models.")
            return []

        model_names = []
        try:
            logger.debug("Calling self._sync_client.list() to fetch models.")
            # ollama library's client.list() returns a dictionary like:
            # {'models': [ModelObject1, ModelObject2, ...]}
            # where ModelObject can be a Pydantic model instance or a TypedDict,
            # both of which have a 'name' attribute/key.
            models_response_dict = self._sync_client.list()
            logger.debug(f"Raw response dict from ollama.Client().list(): {models_response_dict}")

            if models_response_dict and 'models' in models_response_dict and isinstance(models_response_dict['models'],
                                                                                        list):
                models_obj_or_dict_list = models_response_dict['models']

                for i, item in enumerate(models_obj_or_dict_list):
                    model_name_to_add = None
                    if hasattr(item, 'name') and isinstance(getattr(item, 'name'), str):
                        # Handles Pydantic model instance or other objects with .name
                        model_name_to_add = getattr(item, 'name')
                    elif isinstance(item, dict) and 'name' in item and isinstance(item['name'], str):
                        # Handles plain dictionary
                        model_name_to_add = item['name']

                    if model_name_to_add:
                        model_names.append(model_name_to_add)
                    else:
                        # This log will now only appear if the item is truly unexpected
                        logger.warning(
                            f"Item {i} in models list is an unexpected format or type: {item} (Type: {type(item)})")

                logger.info(f"Successfully listed {len(model_names)} models from Ollama.")
            else:
                logger.warning(
                    f"Ollama list() returned unexpected format or empty 'models' list: {models_response_dict}")

            return model_names

        except Exception as e:
            logger.error(f"Error listing models from Ollama: {e}")
            self._last_error = f"Failed to list Ollama models: {type(e).__name__} - {e}"
            return []

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        ollama_messages = []
        skipped_count = 0
        if self._system_prompt:
            ollama_messages.append({"role": "system", "content": self._system_prompt})
        for msg in history:
            if msg.role == USER_ROLE:
                role = 'user'
            elif msg.role == MODEL_ROLE:
                role = 'assistant'
            elif msg.role in [SYSTEM_ROLE, ERROR_ROLE] and msg.metadata and msg.metadata.get("is_internal"):
                skipped_count += 1;
                continue
            else:
                logger.warning(f"Skipping message with unhandled role '{msg.role}' for Ollama API format.");
                skipped_count += 1;
                continue

            content = msg.text;
            images_base64 = []
            if msg.has_images:
                for img_part in msg.image_parts:
                    img_data = img_part.get("data")
                    if isinstance(img_data, str):
                        try:
                            base64.b64decode(img_data); images_base64.append(img_data)
                        except Exception:
                            logger.warning(f"Skipping invalid base64 data in message part for role {role}.")
                    else:
                        logger.warning(f"Skipping non-string image data part for role {role}.")

            ollama_msg: Dict[str, Any] = {"role": role}
            if content.strip(): ollama_msg["content"] = content
            if images_base64: ollama_msg["images"] = images_base64

            if "content" in ollama_msg or "images" in ollama_msg:
                ollama_messages.append(ollama_msg)
            else:
                skipped_count += 1; logger.warning(
                    f"Skipping message with no valid text or image parts for role {role}.")

        if skipped_count > 0:
            logger.debug(
                f"Skipped {skipped_count} messages (non-user/model, internal, or empty) when formatting for Ollama API.")
        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None