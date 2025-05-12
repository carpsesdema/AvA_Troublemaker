# SynChat/backend/gemini_adapter.py
import logging
import asyncio
import os
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple  # Added Tuple

# Import interface and model
from .interface import BackendInterface
from core.models import ChatMessage, MODEL_ROLE, USER_ROLE

# Attempt import for type hinting and error checking
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory, GenerationConfig  # type: ignore
    from google.api_core.exceptions import GoogleAPIError, ClientError, PermissionDenied, ResourceExhausted, \
        InvalidArgument  # type: ignore

    API_LIBRARY_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    HarmCategory = type("HarmCategory", (object,), {})  # type: ignore
    HarmBlockThreshold = type("HarmBlockThreshold", (object,), {})  # type: ignore
    GenerationConfig = type("GenerationConfig", (object,), {})  # type: ignore
    GoogleAPIError = type("GoogleAPIError", (Exception,), {})  # type: ignore
    ClientError = type("ClientError", (GoogleAPIError,), {})  # type: ignore
    PermissionDenied = type("PermissionDenied", (ClientError,), {})  # type: ignore
    ResourceExhausted = type("ResourceExhausted", (ClientError,), {})  # type: ignore
    InvalidArgument = type("InvalidArgument", (ClientError,), {})  # type: ignore
    API_LIBRARY_AVAILABLE = False
    logging.warning("GeminiAdapter: google-generativeai library not found.")

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _next_or_sentinel(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL
    except Exception as e:
        logger.error(f"Error during 'next' call in thread: {e}")
        raise


class GeminiAdapter(BackendInterface):
    """Implementation of the BackendInterface for Google Gemini models."""

    def __init__(self):
        self._model: Optional[genai.GenerativeModel] = None  # type: ignore
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._last_prompt_tokens: Optional[int] = None  # <-- NEW: For token count
        self._last_completion_tokens: Optional[int] = None  # <-- NEW: For token count
        logger.info("GeminiAdapter initialized.")

    def configure(self, api_key: Optional[str], model_name: str, system_prompt: Optional[str] = None) -> bool:
        logger.info(
            f"GeminiAdapter: Configuring. Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._model = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None  # Reset on reconfigure
        self._last_completion_tokens = None  # Reset on reconfigure

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            logger.error(self._last_error)
            return False

        if not api_key or not api_key.strip():
            logger.warning(
                "GeminiAdapter: API key not provided directly to configure method. Library will attempt to use environment variables if genai.configure hasn't been called yet with a key.")

        if not model_name:
            self._last_error = "Model name is required for configuration."
            logger.error(self._last_error)
            return False

        try:
            if api_key and api_key.strip():
                logger.info(f"  Configuring genai with API Key starting: {api_key[:5]}...")
                genai.configure(api_key=api_key)
            else:
                logger.info(
                    "  genai.configure(api_key=...) skipped in this instance; assuming key is in environment or configured elsewhere.")

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            effective_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                   str) and system_prompt.strip() else None

            logger.info(
                f"  Instantiating GenerativeModel: '{model_name}'. System Instruction: {'Present' if effective_prompt else 'None'}")
            self._model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_settings,
                system_instruction=effective_prompt
                # generation_config can be set here too if temperature is fixed per config
            )
            self._model_name = model_name
            self._system_prompt = effective_prompt
            self._is_configured = True
            logger.info(f"  GeminiAdapter configured successfully for model '{model_name}'.")
            return True

        except ValueError as ve:
            self._last_error = f"Configuration Error (ValueError): {ve}. This might be due to an invalid model name."
            logger.error(f"GeminiAdapter Config Error (ValueError): {ve}")
        except InvalidArgument as iae:
            self._last_error = f"Configuration Error (InvalidArgument): {iae}. This might be due to an invalid API key or model name."
            logger.error(f"GeminiAdapter Config Error (InvalidArgument): {iae}")
        except PermissionDenied as pde:
            self._last_error = f"Configuration Error (PermissionDenied): {pde}. Check your API key and permissions."
            logger.error(f"GeminiAdapter Config Error (PermissionDenied): {pde}")
        except Exception as e:
            self._last_error = f"Unexpected error configuring Gemini model '{model_name}': {type(e).__name__} - {e}"
            logger.exception(f"GeminiAdapter Config Error:")

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    # --- MODIFIED get_response_stream to accept options (for temperature) ---
    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
    AsyncGenerator[str, None]:
        logger.info(f"GeminiAdapter: Generating stream. History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None  # Reset before new request
        self._last_completion_tokens = None  # Reset before new request

        if not self.is_configured() or not self._model:
            self._last_error = "Adapter is not configured or model object is missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        gemini_history = self._format_history_for_api(history)
        if not gemini_history:
            self._last_error = "Cannot send request: No valid messages in history for the API format."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.info(f"  Sending {len(gemini_history)} entries to model '{self._model_name}'.")

        # --- Prepare GenerationConfig for temperature ---
        generation_config_dict = {}
        if options and "temperature" in options and isinstance(options["temperature"], (float, int)):
            temp_val = float(options["temperature"])
            # Gemini Pro models typically have a temperature range of 0.0 to 1.0 or 2.0
            # We assume the value is already validated/clamped by ChatManager if needed.
            generation_config_dict["temperature"] = temp_val
            logger.info(f"  Applying temperature from options: {temp_val}")

        effective_generation_config = GenerationConfig(**generation_config_dict) if generation_config_dict else None
        # --- End GenerationConfig preparation ---

        try:
            if not hasattr(self._model, 'generate_content'):
                self._last_error = "Internal Error: Configured model object lacks 'generate_content'."
                logger.error(self._last_error);
                raise AttributeError(self._last_error)

            logger.debug("  Making initial blocking API call in thread...")
            # Pass generation_config to the API call
            response_object = await asyncio.to_thread(
                self._model.generate_content,
                gemini_history,
                stream=True,
                generation_config=effective_generation_config  # <-- PASSING TEMP CONFIG
            )
            logger.debug("  Initial API call returned response object.")

            sync_iterator = iter(response_object)

            async def _internal_chunk_generator() -> AsyncGenerator[str, None]:
                logger.debug("    Starting async chunk yielding loop...")
                chunk_count = 0
                try:
                    while True:
                        chunk_or_sentinel = await asyncio.to_thread(_next_or_sentinel, sync_iterator)
                        if chunk_or_sentinel is _SENTINEL:
                            logger.info("    Stream finished normally (Sentinel received from thread).")
                            break

                        chunk = chunk_or_sentinel
                        chunk_count += 1
                        error_in_chunk = None

                        prompt_feedback = getattr(chunk, 'prompt_feedback', None)
                        if prompt_feedback:
                            block_reason = getattr(prompt_feedback, 'block_reason', None)
                            if block_reason:
                                error_in_chunk = f"Content blocked by API safety filters: {block_reason}."
                                logger.warning(f"API Blocked in stream: {error_in_chunk}")
                                self._last_error = error_in_chunk
                                yield f"[SYSTEM ERROR: {error_in_chunk}]"
                                break

                        if error_in_chunk: continue

                        text_to_yield_from_chunk_parts = []
                        if hasattr(chunk, 'candidates'):
                            for candidate in chunk.candidates:
                                if (hasattr(candidate, 'content') and
                                        candidate.content and
                                        hasattr(candidate.content, 'parts') and
                                        candidate.content.parts):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            text_to_yield_from_chunk_parts.append(part.text)
                        if text_to_yield_from_chunk_parts:
                            full_chunk_text = "".join(text_to_yield_from_chunk_parts)
                            if full_chunk_text:
                                yield full_chunk_text
                except Exception as e_yield:
                    self._last_error = f"Error during stream processing/yielding chunk: {type(e_yield).__name__} - {e_yield}"
                    logger.exception("    Error during async yield loop:")
                    raise RuntimeError(self._last_error) from e_yield
                finally:
                    logger.info(f"    Async chunk yielding loop finished after {chunk_count} chunks.")

            async for text_chunk in _internal_chunk_generator():
                yield text_chunk

            # --- TOKEN COUNT EXTRACTION ---
            # After the stream is fully consumed, response_object might have usage_metadata
            # This might only work if the stream is fully iterated and not broken early.
            if hasattr(response_object, 'usage_metadata') and response_object.usage_metadata:
                usage = response_object.usage_metadata
                self._last_prompt_tokens = getattr(usage, 'prompt_token_count', None)
                self._last_completion_tokens = getattr(usage, 'candidates_token_count',
                                                       None)  # Sum of tokens from all candidates
                logger.info(
                    f"  Gemini Token Usage: Prompt={self._last_prompt_tokens}, Completion={self._last_completion_tokens}")
            else:
                # Sometimes, for streaming, the usage metadata might be on the *last chunk* or not available until the full response aggregates.
                # This part might need adjustment based on how google-generativeai handles streaming usage reporting.
                # Let's log if it's not directly on the response_object.
                logger.warning(
                    "  Gemini usage_metadata not found directly on response_object after stream. Token counts may be unavailable.")


        except InvalidArgument as e:
            self._last_error = f"API Error (Invalid Argument): {e}. This might be an issue with the request format or API key.";
            logger.error(self._last_error);
            raise
        except PermissionDenied as e:
            self._last_error = f"API Error (Permission Denied/Safety Block): {e}. Check API key, model permissions, or content safety.";
            logger.error(self._last_error);
            raise
        # ... (other exception handling remains the same) ...
        except Exception as e:
            if not self._last_error or isinstance(e, (AttributeError, ValueError)):
                self._last_error = f"Unexpected error preparing/executing Gemini stream: {type(e).__name__} - {e}"
            logger.exception("GeminiAdapter stream preparation/execution failed (outer catch):")
            raise RuntimeError(self._last_error) from e

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        # ... (this method remains the same) ...
        gemini_history = []
        skipped_count = 0
        for msg in history:
            if msg.role == USER_ROLE:
                role = 'user'
            elif msg.role == MODEL_ROLE:
                role = 'model'
            else:
                skipped_count += 1;
                continue
            parts_list = []
            if msg.parts:
                for part_item in msg.parts:
                    if isinstance(part_item, str):
                        stripped_part = part_item.strip()
                        if stripped_part:
                            parts_list.append(stripped_part)
            if not parts_list:
                skipped_count += 1;
                continue
            gemini_history.append({"role": role, "parts": parts_list})
        if skipped_count > 0:
            logger.debug(
                f"Skipped {skipped_count} messages (non-user/model or empty text parts) when formatting for Gemini API.")
        return gemini_history

    def get_available_models(self) -> List[str]:
        # ... (this method remains the same) ...
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            logger.error(self._last_error)
            return []
        logger.info("GeminiAdapter: Attempting to dynamically fetch available models from genai.list_models()...")
        fetched_models: List[str] = []
        try:
            if not self._is_configured and not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
                try:
                    if genai.get_model("gemini-pro"):  # type: ignore
                        pass
                except Exception:
                    logger.warning(
                        "GeminiAdapter: API key seems unavailable for listing models. `genai.configure` likely not called or env var missing/invalid.")
            for model_info in genai.list_models():  # type: ignore
                if 'generateContent' in model_info.supported_generation_methods:
                    if ("gemini" in model_info.name or "models/gemini" in model_info.name) and \
                            not any(unwanted in model_info.name for unwanted in ["embedding", "aqa", "retriever"]):
                        fetched_models.append(model_info.name)
            if fetched_models:
                logger.info(f"Dynamically fetched {len(fetched_models)} suitable Gemini models: {fetched_models}")
            else:
                logger.warning("Dynamic fetch from genai.list_models() returned no models matching criteria.")
        except PermissionDenied as pde:
            self._last_error = f"API Permission Denied while listing models: {pde}. Check API key permissions."
            logger.error(self._last_error);
            return []
        except InvalidArgument as iae:
            self._last_error = f"API Invalid Argument while listing models: {iae}. Check API key format or service endpoint."
            logger.error(self._last_error);
            return []
        except GoogleAPIError as api_err:
            self._last_error = f"Google API Error while listing models: {type(api_err).__name__} - {api_err}"
            logger.error(self._last_error);
            return []
        except Exception as e:
            self._last_error = f"Unexpected error dynamically fetching models from genai.list_models(): {type(e).__name__} - {e}"
            logger.exception("GeminiAdapter model listing failed:");
            return []
        final_model_list = list(set(fetched_models))
        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            logger.warning(
                f"Configured model '{self._model_name}' not in dynamically fetched list. Adding it as it was configured successfully.")
            final_model_list.insert(0, self._model_name)
            final_model_list = list(set(final_model_list))
        final_model_list.sort()
        return final_model_list

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