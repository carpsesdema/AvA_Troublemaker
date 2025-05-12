# core/backend_coordinator.py
# FULL FILE - Includes changes to _internal_get_response_stream for metadata preservation.

import logging
import asyncio
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

# Assuming BackendInterface and ChatMessage are correctly importable
try:
    from backend.interface import BackendInterface
    from .models import ChatMessage, MODEL_ROLE
except ImportError:
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MODEL_ROLE = "model"  # type: ignore
    logging.error("BackendCoordinator: Could not import BackendInterface or ChatMessage from expected locations.")

logger = logging.getLogger(__name__)


class BackendCoordinator(QObject):
    stream_started = pyqtSignal(str)
    stream_chunk_received = pyqtSignal(str, str)
    response_completed = pyqtSignal(str, ChatMessage, dict)
    response_error = pyqtSignal(str, str)
    busy_state_changed = pyqtSignal(bool)
    configuration_changed = pyqtSignal(str, str, bool, list)

    def __init__(self, backend_adapters: Dict[str, BackendInterface], parent: Optional[QObject] = None):
        super().__init__(parent)
        if not backend_adapters:
            raise ValueError("BackendCoordinator requires a non-empty dictionary of BackendInterface instances.")
        self._backend_adapters: Dict[str, BackendInterface] = backend_adapters
        self._current_model_names: Dict[str, Optional[str]] = {}
        self._current_system_prompts: Dict[str, Optional[str]] = {}
        self._is_configured_map: Dict[str, bool] = {bid: False for bid in backend_adapters}
        self._available_models_map: Dict[str, List[str]] = {bid: [] for bid in backend_adapters}
        self._last_errors_map: Dict[str, Optional[str]] = {bid: None for bid in backend_adapters}
        self._active_backend_tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._overall_is_busy: bool = False
        logger.info(
            f"BackendCoordinator initialized with {len(self._backend_adapters)} adapters: {list(self._backend_adapters.keys())}")

    def _update_overall_busy_state(self):
        new_busy_state = any(task and not task.done() for task in self._active_backend_tasks.values())
        if self._overall_is_busy != new_busy_state:
            self._overall_is_busy = new_busy_state
            logger.debug(f"BC: Overall busy state changed to {self._overall_is_busy} by _update_overall_busy_state.")
            self.busy_state_changed.emit(self._overall_is_busy)

    def configure_backend(self, backend_id: str, api_key: Optional[str], model_name: str,
                          system_prompt: Optional[str] = None) -> bool:
        logger.info(
            f"BackendCoordinator: Configuring backend '{backend_id}'. Model: {model_name}, System Prompt: {'Set' if system_prompt else 'None'}")
        adapter = self._backend_adapters.get(backend_id)
        if not adapter:
            logger.error(f"Configuration failed: No adapter found for backend_id '{backend_id}'.")
            self._is_configured_map[backend_id] = False;
            self._last_errors_map[backend_id] = f"Adapter not found for {backend_id}"
            self.configuration_changed.emit(backend_id, model_name, False, []);
            return False
        self._current_model_names[backend_id] = model_name;
        self._current_system_prompts[backend_id] = system_prompt
        is_configured = adapter.configure(api_key=api_key, model_name=model_name, system_prompt=system_prompt)
        self._is_configured_map[backend_id] = is_configured
        self._last_errors_map[backend_id] = adapter.get_last_error() if not is_configured else None
        available_models_for_backend: List[str] = []
        if is_configured:
            logger.info(
                f"Adapter for '{backend_id}' configured successfully for model '{model_name}'. Fetching available models...")
            try:
                available_models_for_backend = adapter.get_available_models()
                self._available_models_map[backend_id] = available_models_for_backend
                logger.info(f"Fetched {len(available_models_for_backend)} available models for '{backend_id}'.")
                if model_name not in available_models_for_backend and available_models_for_backend:
                    if model_name and any(
                        m.startswith(model_name.split('/')[0]) for m in available_models_for_backend if
                        isinstance(m, str)):
                        pass
                    else:
                        logger.warning(
                            f"Configured model '{model_name}' for '{backend_id}' not in its available list: {available_models_for_backend}")
            except Exception as e:
                logger.exception(f"Error fetching available models for backend '{backend_id}':");
                self._available_models_map[backend_id] = []
        else:
            logger.error(
                f"Adapter configuration FAILED for backend '{backend_id}', model '{model_name}': {self._last_errors_map[backend_id]}")
            self._available_models_map[backend_id] = []
        self.configuration_changed.emit(backend_id, model_name, is_configured, available_models_for_backend[:]);
        return is_configured

    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        adapter = self._backend_adapters.get(backend_id)
        if adapter and self._is_configured_map.get(backend_id, False):
            try:
                return adapter.get_available_models()
            except Exception as e:
                logger.exception(f"Error fetching available models directly from adapter '{backend_id}':")
        logger.warning(f"Cannot get available models for '{backend_id}': Adapter not configured or not found.");
        return []

    def request_response_stream(self,
                                target_backend_id: str,
                                request_id: str,
                                history_to_send: List[ChatMessage],
                                is_modification_response_expected: bool,
                                options: Optional[Dict[str, Any]] = None,
                                request_metadata: Optional[Dict[str, Any]] = None):
        if not request_id:
            logger.error(
                f"BC: Cannot request response: 'request_id' is missing or empty for backend '{target_backend_id}'.")
            return

        task_key = (target_backend_id, request_id)
        active_task_for_request = self._active_backend_tasks.get(task_key)

        if active_task_for_request and not active_task_for_request.done():
            logger.warning(
                f"BC: Backend '{target_backend_id}' is already busy with task for request_id '{request_id}'. Task: {active_task_for_request}. Ignoring new request.")
            self.response_error.emit(request_id, "Backend busy with this request. Please wait.")
            return

        adapter = self._backend_adapters.get(target_backend_id)
        if not adapter:
            logger.error(f"BC: Cannot request response: No adapter for backend_id '{target_backend_id}'.")
            self.response_error.emit(request_id, f"Backend adapter '{target_backend_id}' not found.");
            return
        if not self._is_configured_map.get(target_backend_id, False):
            logger.error(f"BC: Cannot request response: Backend '{target_backend_id}' is not configured.")
            self.response_error.emit(request_id, f"Backend '{target_backend_id}' not configured.");
            return

        logger.info(
            f"BC: Creating task for backend '{target_backend_id}', request_id '{request_id}'. Mod expected: {is_modification_response_expected}. Options: {options}, Meta: {request_metadata}")

        task = asyncio.create_task(
            self._internal_get_response_stream(
                backend_id=target_backend_id,
                request_id=request_id,
                adapter=adapter,
                history=history_to_send,
                is_modification_response_expected=is_modification_response_expected,
                options=options,
                request_metadata=request_metadata  # Pass our custom metadata here
            )
        )
        self._active_backend_tasks[task_key] = task
        self._update_overall_busy_state()
        logger.info(
            f"BC: Task for '{target_backend_id}' (req_id: {request_id}) added. Task: {task}. Active tasks count: {len(self._active_backend_tasks)}")

    async def _internal_get_response_stream(self,
                                            backend_id: str,
                                            request_id: str,
                                            adapter: BackendInterface,
                                            history: List[ChatMessage],
                                            is_modification_response_expected: bool,
                                            options: Optional[Dict[str, Any]] = None,
                                            request_metadata: Optional[
                                                Dict[str, Any]] = None):  # This is our custom metadata
        logger.info(
            f"Task _internal_get_response_stream for backend '{backend_id}', req_id '{request_id}' ({asyncio.current_task()}) started. Custom Meta: {request_metadata}")
        response_buffer = ""

        # Initialize usage_stats_dict WITH our custom request_metadata first, if it exists
        usage_stats_dict: Dict[str, Any] = {}
        if request_metadata:
            usage_stats_dict.update(request_metadata)  # Start with our metadata

        try:
            if not hasattr(adapter, 'get_response_stream'):
                raise AttributeError(f"Backend adapter '{backend_id}' missing get_response_stream method.")

            self.stream_started.emit(request_id)
            logger.debug(f"BC: Emitted stream_started for request_id '{request_id}'.")

            stream_iterator = adapter.get_response_stream(history, options)

            async for chunk in stream_iterator:
                if not is_modification_response_expected:
                    self.stream_chunk_received.emit(request_id, chunk)
                response_buffer += chunk

            logger.info(f"Backend stream iterator for '{backend_id}' (req_id: '{request_id}') finished normally.")
            final_response_text = response_buffer.strip()

            # Get token usage from adapter and add to usage_stats_dict if not already present
            if hasattr(adapter, 'get_last_token_usage'):
                token_usage = adapter.get_last_token_usage()
                if token_usage:
                    if "prompt_tokens" not in usage_stats_dict:  # Avoid overwriting if passed in request_metadata
                        usage_stats_dict["prompt_tokens"] = token_usage[0]
                    if "completion_tokens" not in usage_stats_dict:  # Avoid overwriting
                        usage_stats_dict["completion_tokens"] = token_usage[1]

            # The 'purpose' and 'backend_id_for_mc' should now be preserved in usage_stats_dict
            # as they were added from request_metadata at the beginning.

            if final_response_text:
                completed_message = ChatMessage(role=MODEL_ROLE, parts=[final_response_text])
                if completed_message.metadata is None: completed_message.metadata = {}
                completed_message.metadata["request_id"] = request_id
                self.response_completed.emit(request_id, completed_message, usage_stats_dict)
            elif is_modification_response_expected or self.stream_chunk_received.receivers(request_id, ""):
                empty_msg_text = "[AI returned an empty response]"
                if is_modification_response_expected: empty_msg_text = "[System: AI returned an empty response during modification step.]"

                # Ensure usage_stats_dict has the request_id for the metadata of the empty message
                # This is actually redundant if request_metadata always contains request_id, but safe.
                # The important metadata like 'purpose' should already be in usage_stats_dict.
                if "request_id" not in usage_stats_dict: usage_stats_dict["request_id"] = request_id

                empty_msg = ChatMessage(role=MODEL_ROLE, parts=[empty_msg_text], metadata=usage_stats_dict.copy())
                self.response_completed.emit(request_id, empty_msg, usage_stats_dict)
            else:
                backend_error = adapter.get_last_error()
                err_msg_text = backend_error if backend_error else f"AI for '{backend_id}' failed or returned no response for req_id '{request_id}'"
                self.response_error.emit(request_id, err_msg_text)

        except asyncio.CancelledError:
            logger.info(f"BC task for '{backend_id}' (req_id: '{request_id}') cancelled.")
            self.response_error.emit(request_id, "[AI response cancelled by user]")
        except Exception as e:
            logger.exception(f"Error in _internal_get_response_stream for '{backend_id}' (req_id: '{request_id}'):")
            error_msg = adapter.get_last_error() or f"Backend Task Error ({backend_id}, req_id: {request_id}): {type(e).__name__}"
            self.response_error.emit(request_id, error_msg)
        finally:
            task_key = (backend_id, request_id)
            logger.info(
                f"BC Task Finally: Task for '{backend_id}' (req_id: '{request_id}', task: {asyncio.current_task()}) finishing...")
            task_instance_in_dict = self._active_backend_tasks.get(task_key)
            if task_instance_in_dict is asyncio.current_task():
                del self._active_backend_tasks[task_key]
                logger.debug(f"BC Task Finally: Removed task for {task_key} from active tasks.")
            elif task_instance_in_dict:
                logger.warning(
                    f"BC Task Finally: Task instance for {task_key} in dict ({task_instance_in_dict}) does NOT match current task ({asyncio.current_task()}). Not deleting from dict.")
            else:
                logger.warning(
                    f"BC Task Finally: Task key {task_key} not found in _active_backend_tasks during finally block.")
            self._update_overall_busy_state()

    def cancel_current_task(self, backend_id: Optional[str] = None, request_id: Optional[str] = None):
        if backend_id and request_id:
            task_key = (backend_id, request_id)
            task = self._active_backend_tasks.get(task_key)
            if task and not task.done():
                logger.info(f"BC: Cancelling task {task} for backend '{backend_id}', request_id '{request_id}'...");
                task.cancel()
            else:
                logger.debug(
                    f"BC: No active task for backend '{backend_id}', request_id '{request_id}' to cancel (Task: {task}).")
        elif backend_id:
            logger.info(f"BC: Cancelling ALL active tasks for backend '{backend_id}'...")
            tasks_to_cancel_for_backend = {
                key: task for key, task in self._active_backend_tasks.items() if key[0] == backend_id
            }
            for key, task_to_cancel in tasks_to_cancel_for_backend.items():
                if task_to_cancel and not task_to_cancel.done():
                    logger.info(f"  Cancelling task {task_to_cancel} for key {key}...");
                    task_to_cancel.cancel()
        else:
            logger.info("BC: Cancelling ALL active backend tasks...")
            for key, task_to_cancel in list(self._active_backend_tasks.items()):
                if task_to_cancel and not task_to_cancel.done():
                    logger.info(f"  Cancelling task {task_to_cancel} for key {key}...");
                    task_to_cancel.cancel()
            logger.debug("Requested cancellation for all active tasks.")

    def is_backend_configured(self, backend_id: str) -> bool:
        return self._is_configured_map.get(backend_id, False)

    def get_last_error_for_backend(self, backend_id: str) -> Optional[str]:
        adapter = self._backend_adapters.get(backend_id)
        direct_error = adapter.get_last_error() if adapter else None
        return direct_error or self._last_errors_map.get(backend_id)

    def is_processing_request(self) -> bool:
        return self._overall_is_busy

    def get_current_configured_model(self, backend_id: str) -> Optional[str]:
        return self._current_model_names.get(backend_id)

    def get_current_system_prompt(self, backend_id: str) -> Optional[str]:
        return self._current_system_prompts.get(backend_id)