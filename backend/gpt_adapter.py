# === backend/gpt_adapter.py ===
import logging
import asyncio
import os
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

# Import interface and model
from .interface import BackendInterface
from core.models import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE  # Added SYSTEM_ROLE

# Attempt import for type hinting and error checking for the OpenAI library
try:
    import openai
    from openai import APIError, AuthenticationError, RateLimitError, NotFoundError  # Specific errors

    OPENAI_API_LIBRARY_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    # Define dummy types for specific errors if openai is not available
    APIError = type("APIError", (Exception,), {})  # type: ignore
    AuthenticationError = type("AuthenticationError", (APIError,), {})  # type: ignore
    RateLimitError = type("RateLimitError", (APIError,), {})  # type: ignore
    NotFoundError = type("NotFoundError", (APIError,), {})  # type: ignore
    OPENAI_API_LIBRARY_AVAILABLE = False
    logging.warning("GPTAdapter: 'openai' library not found. Please install it: pip install openai")

logger = logging.getLogger(__name__)

_SENTINEL = object()  # For the async generator pattern


def _next_or_sentinel_gpt(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL
    except Exception as e:
        logger.error(f"GPTAdapter: Error during 'next' call in thread: {e}")
        raise  # Re-raise to be caught by the calling async task


class GPTAdapter(BackendInterface):
    """Implementation of the BackendInterface for OpenAI GPT models."""

    def __init__(self):
        self._client: Optional[openai.OpenAI] = None  # Using synchronous client
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("GPTAdapter initialized.")

    def configure(self, api_key: Optional[str], model_name: str, system_prompt: Optional[str] = None) -> bool:
        logger.info(f"GPTAdapter: Configuring. Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not OPENAI_API_LIBRARY_AVAILABLE:
            self._last_error = "OpenAI API library ('openai') not installed."
            logger.error(self._last_error)
            return False

        effective_api_key = api_key
        if not effective_api_key or not effective_api_key.strip():
            effective_api_key = os.getenv("OPENAI_API_KEY")
            if not effective_api_key or not effective_api_key.strip():
                self._last_error = "OpenAI API key not provided directly and not found in OPENAI_API_KEY environment variable."
                logger.error(self._last_error)
                return False
            logger.info("GPTAdapter: Using API key from OPENAI_API_KEY environment variable.")

        if not model_name:
            self._last_error = "Model name is required for GPT configuration."
            logger.error(self._last_error)
            return False

        try:
            self._client = openai.OpenAI(api_key=effective_api_key)
            # Optional: Test client connectivity here if desired, e.g., self._client.models.list()
            # For now, defer error to first actual use or get_available_models.

            self._model_name = model_name
            self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                      str) and system_prompt.strip() else None
            self._is_configured = True
            logger.info(f"GPTAdapter configured successfully for model '{model_name}'.")
            return True

        except AuthenticationError as e:
            self._last_error = f"OpenAI Authentication Error: {e}. Check your API key."
            logger.error(self._last_error)
        except RateLimitError as e:
            self._last_error = f"OpenAI Rate Limit Error: {e}."
            logger.error(self._last_error)
        except NotFoundError as e:  # Could happen if model_name is invalid during a test call
            self._last_error = f"OpenAI Not Found Error (e.g., invalid model name during test): {e}."
            logger.error(self._last_error)
        except APIError as e:  # Generic API error
            self._last_error = f"OpenAI API Error: {e}."
            logger.error(self._last_error)
        except Exception as e:
            self._last_error = f"Unexpected error configuring OpenAI model '{model_name}': {type(e).__name__} - {e}"
            logger.exception("GPTAdapter Config Error:")

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
    AsyncGenerator[str, None]:
        logger.info(
            f"GPTAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._client:  # self._client should be OpenAI instance here
            self._last_error = "GPTAdapter is not configured or client object is missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)
        if not messages_for_api and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages in history for API and no system prompt."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.info(f"  Sending {len(messages_for_api)} message parts to model '{self._model_name}'.")

        api_params: Dict[str, Any] = {
            "model": self._model_name,  # type: ignore # self._model_name is checked in configure
            "messages": messages_for_api,
            "stream": True
        }
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                api_params["temperature"] = float(options["temperature"])
                logger.info(f"  Applying temperature: {api_params['temperature']}")
            if "max_tokens" in options and isinstance(options["max_tokens"], int):
                api_params["max_tokens"] = options["max_tokens"]
                logger.info(f"  Applying max_tokens: {api_params['max_tokens']}")
            # Add other OpenAI specific params if needed: top_p, presence_penalty, frequency_penalty

        try:
            # Using asyncio.to_thread for the synchronous OpenAI client's stream method
            sync_iterator = await asyncio.to_thread(
                self._client.chat.completions.create,  # type: ignore
                **api_params
            )
            logger.debug("  Initial API call returned response stream iterator.")

            async def _internal_chunk_generator() -> AsyncGenerator[str, None]:
                # full_completion_text_for_tokens = "" # Not strictly needed if SDK gives usage
                try:
                    while True:
                        chunk_or_sentinel = await asyncio.to_thread(_next_or_sentinel_gpt, sync_iterator)
                        if chunk_or_sentinel is _SENTINEL:
                            logger.info("    Stream finished normally (Sentinel received from thread).")
                            break

                        chunk = chunk_or_sentinel  # chunk is openai.types.chat.ChatCompletionChunk
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta
                        finish_reason = chunk.choices[0].finish_reason

                        if delta and delta.content:
                            yield delta.content
                            # full_completion_text_for_tokens += delta.content

                        if chunk.usage:
                            self._last_prompt_tokens = chunk.usage.prompt_tokens
                            self._last_completion_tokens = chunk.usage.completion_tokens
                            logger.info(
                                f"  GPT Token Usage (from stream chunk): Prompt={self._last_prompt_tokens}, Completion={self._last_completion_tokens}")

                        if finish_reason:
                            logger.info(f"    Stream finished. Finish reason: {finish_reason}")
                            # If it's a 'length' finish_reason, it means max_tokens was hit.
                            break  # Break from while True loop

                except Exception as e_yield:
                    self._last_error = f"Error during GPT stream processing/yielding chunk: {type(e_yield).__name__} - {e_yield}"
                    logger.exception("    Error during async yield loop:")
                    # No re-raise here, allow finally to run if needed, error will be propagated from outer try
                finally:
                    logger.info("    Async chunk yielding loop finished.")
                    # Check if tokens were found; if not, they remain None.
                    if self._last_prompt_tokens is None or self._last_completion_tokens is None:
                        logger.warning("    GPTAdapter: Token usage not found in stream. Counts may be unavailable.")

            async for text_chunk in _internal_chunk_generator():
                yield text_chunk

        except AuthenticationError as e:
            self._last_error = f"OpenAI API Authentication Error: {e}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e
        except RateLimitError as e:
            self._last_error = f"OpenAI API Rate Limit Error: {e}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e
        except NotFoundError as e:  # E.g. model not found
            self._last_error = f"OpenAI API Not Found Error (e.g., model '{self._model_name}' invalid): {e}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e
        except APIError as e:
            self._last_error = f"OpenAI API Error: {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e
        except Exception as e:
            if not self._last_error:
                self._last_error = f"Unexpected error executing OpenAI stream: {type(e).__name__} - {e}"
            logger.exception("GPTAdapter stream execution failed (outer catch):")
            raise RuntimeError(self._last_error) from e

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        openai_messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            openai_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:
            role_for_api: Optional[str] = None
            if msg.role == USER_ROLE:
                role_for_api = "user"
            elif msg.role == MODEL_ROLE:
                role_for_api = "assistant"
            elif msg.role == SYSTEM_ROLE:  # Allow passthrough of system messages from history
                # Be cautious: multiple system messages are often not well-supported or might override initial one.
                # For OpenAI, the first system message is typically the primary one.
                # If one is already added from self._system_prompt, adding more from history might be an issue.
                # For now, let's assume if self._system_prompt is set, it's the main one.
                # If history contains system messages, and self._system_prompt was NOT set, then they can be added.
                if not self._system_prompt:  # Only add if no class-level system prompt
                    role_for_api = "system"
                else:
                    logger.debug(
                        f"GPTAdapter: Skipping history system message due to existing adapter system prompt. Text: {msg.text[:50]}...")
                    continue
            else:
                logger.warning(f"GPTAdapter: Skipping message with unhandled role '{msg.role}'.")
                continue

            # For now, assuming ChatMessage.text is the primary content.
            # And ChatMessage.parts can contain image data if we extend for multimodal.
            text_content = msg.text

            # Check for image parts (basic structure, needs refinement for actual multimodal)
            # OpenAI expects content to be an array of parts for multimodal.
            # Example: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            message_content_parts: List[Dict[str, Any]] = []
            has_text_part = False
            if text_content and text_content.strip():
                message_content_parts.append({"type": "text", "text": text_content})
                has_text_part = True

            if msg.has_images:
                for img_part_dict in msg.image_parts:  # image_parts from ChatMessage model
                    # Expecting img_part_dict to be like:
                    # {"type": "image", "mime_type": "image/jpeg", "data": "base64_string"}
                    if img_part_dict.get("type") == "image" and \
                            img_part_dict.get("mime_type") and \
                            img_part_dict.get("data"):

                        # Format for OpenAI vision API:
                        # "data:image/jpeg;base64,{base64_image}"
                        image_url_data = f"data:{img_part_dict['mime_type']};base64,{img_part_dict['data']}"
                        message_content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": image_url_data}
                        })
                    else:
                        logger.warning(f"Skipping malformed image part in message ID {msg.id}")

            final_content_for_api: Any
            if len(message_content_parts) > 1:  # Multimodal (text + image, or multiple images)
                final_content_for_api = message_content_parts
            elif has_text_part:  # Only text
                final_content_for_api = text_content
            elif message_content_parts:  # Only image(s), no text part added initially
                final_content_for_api = message_content_parts
            else:  # No valid text or image content after processing
                if role_for_api in ["user", "assistant"]:
                    logger.warning(
                        f"GPTAdapter: Skipping {role_for_api} message (ID: {msg.id}) with no valid text or image content after formatting.")
                    continue
                else:  # e.g. system message might be just role sometimes
                    final_content_for_api = ""

            openai_messages.append({"role": role_for_api, "content": final_content_for_api})
        return openai_messages

    def get_available_models(self) -> List[str]:
        self._last_error = None
        if not OPENAI_API_LIBRARY_AVAILABLE:
            self._last_error = "OpenAI API library ('openai') not installed."
            logger.error(self._last_error);
            return []
        if not self.is_configured() or not self._client:
            self._last_error = "GPTAdapter not configured (or API key missing). Cannot list models."
            logger.warning(self._last_error);
            return []

        fetched_models: List[str] = []
        try:
            logger.info("GPTAdapter: Dynamically fetching available models from OpenAI API...")
            model_list_response = self._client.models.list()

            # GPT-4 and GPT-3.5 Turbo series are primary chat models.
            # Other models like DALL-E, Whisper, Embeddings, Moderations are not for chat.
            # Fine-tuned models might have custom names but often derive from base models.
            chat_model_prefixes = ("gpt-4", "gpt-3.5-turbo")
            # Exclude models that are clearly not for chat/text generation based on common naming
            excluded_suffixes_or_types = ("embedding", "vision", "image", "audio", "edit", "instruct", "search",
                                          "similarity", "code-interpreter", "plugins")

            for model_obj in model_list_response.data:
                model_id = model_obj.id.lower()
                is_chat_candidate = any(model_id.startswith(prefix) for prefix in chat_model_prefixes)

                if is_chat_candidate:
                    is_excluded = any(excluded_type in model_id for excluded_type in excluded_suffixes_or_types if
                                      excluded_type not in chat_model_prefixes)  # Avoid self-exclusion
                    # Special case: gpt-4-vision-preview is for vision, not general chat here.
                    # However, gpt-4-turbo with vision is still primarily a text model.
                    # This needs careful tuning. For now, let's assume pure text generation focus.
                    if "vision" in model_id and "turbo" not in model_id:  # e.g. gpt-4-vision-preview
                        is_excluded = True

                    if not is_excluded:
                        fetched_models.append(model_obj.id)  # Use original casing

            if fetched_models:
                logger.info(f"Dynamically fetched {len(fetched_models)} potential GPT chat models: {fetched_models}")
            else:
                logger.warning("Dynamic fetch from OpenAI API returned no models matching primary chat criteria.")

        except AuthenticationError as e:
            self._last_error = f"OpenAI API Authentication Error while listing models: {e}"
            logger.error(self._last_error, exc_info=True);
            return []
        except RateLimitError as e:
            self._last_error = f"OpenAI API Rate Limit Error while listing models: {e}"
            logger.error(self._last_error, exc_info=True);
            return []
        except APIError as e:  # Generic API error
            self._last_error = f"OpenAI API Error while listing models: {e}"
            logger.error(self._last_error, exc_info=True);
            return []
        except Exception as e:
            self._last_error = f"Unexpected error dynamically fetching models from OpenAI: {type(e).__name__} - {e}"
            logger.exception("GPTAdapter model listing failed:");
            return []

        # Add the currently configured model if it's not in the list (e.g. fine-tuned, or preview not listed yet)
        if self._model_name and self._is_configured and self._model_name not in fetched_models:
            logger.warning(
                f"Configured GPT model '{self._model_name}' not in dynamically fetched list. Adding it as it was configured.")
            fetched_models.insert(0, self._model_name)

        final_model_list = sorted(list(set(fetched_models)))
        return final_model_list

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None