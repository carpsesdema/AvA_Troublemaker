# core/backend_coordinator.py
import logging
import asyncio
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from backend.interface import BackendInterface
from .models import ChatMessage, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE

logger = logging.getLogger(__name__)


class BackendCoordinator(QObject):
    stream_chunk_received = pyqtSignal(str, str) # backend_id, chunk
    # usage_stats dictionary might now contain original request_metadata
    response_completed = pyqtSignal(str, ChatMessage, dict) # backend_id, completed_message, usage_stats_with_metadata
    response_error = pyqtSignal(str, str) # backend_id, error_message
    busy_state_changed = pyqtSignal(bool) # overall_is_busy
    configuration_changed = pyqtSignal(str, str, bool, list) # backend_id, model_name, is_configured, available_models

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
        self._active_backend_tasks: Dict[str, asyncio.Task] = {}
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
            self._is_configured_map[backend_id] = False; self._last_errors_map[backend_id] = f"Adapter not found for {backend_id}"
            self.configuration_changed.emit(backend_id, model_name, False, []); return False
        self._current_model_names[backend_id] = model_name; self._current_system_prompts[backend_id] = system_prompt
        is_configured = adapter.configure(api_key=api_key, model_name=model_name, system_prompt=system_prompt)
        self._is_configured_map[backend_id] = is_configured
        self._last_errors_map[backend_id] = adapter.get_last_error() if not is_configured else None
        available_models_for_backend: List[str] = []
        if is_configured:
            logger.info(f"Adapter for '{backend_id}' configured successfully for model '{model_name}'. Fetching available models...")
            try:
                available_models_for_backend = adapter.get_available_models()
                self._available_models_map[backend_id] = available_models_for_backend
                logger.info(f"Fetched {len(available_models_for_backend)} available models for '{backend_id}'.")
                if model_name not in available_models_for_backend and available_models_for_backend:
                    if model_name and any(m.startswith(model_name.split('/')[0]) for m in available_models_for_backend if isinstance(m, str)): pass
                    else: logger.warning(f"Configured model '{model_name}' for '{backend_id}' not in its available list: {available_models_for_backend}")
            except Exception as e: logger.exception(f"Error fetching available models for backend '{backend_id}':"); self._available_models_map[backend_id] = []
        else:
            logger.error(f"Adapter configuration FAILED for backend '{backend_id}', model '{model_name}': {self._last_errors_map[backend_id]}")
            self._available_models_map[backend_id] = []
        self.configuration_changed.emit(backend_id, model_name, is_configured, available_models_for_backend[:]); return is_configured


    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        adapter = self._backend_adapters.get(backend_id)
        if adapter and self._is_configured_map.get(backend_id, False):
            try: return adapter.get_available_models()
            except Exception as e: logger.exception(f"Error fetching available models directly from adapter '{backend_id}':")
        logger.warning(f"Cannot get available models for '{backend_id}': Adapter not configured or not found."); return []

    # --- MODIFIED: Added request_metadata parameter ---
    def request_response_stream(self, target_backend_id: str, history_to_send: List[ChatMessage],
                                is_modification_response_expected: bool,
                                options: Optional[Dict[str, Any]] = None,
                                request_metadata: Optional[Dict[str, Any]] = None): # <-- NEW PARAMETER
        active_task_for_backend = self._active_backend_tasks.get(target_backend_id)
        if active_task_for_backend and not active_task_for_backend.done():
            logger.warning(
                f"BC: Backend '{target_backend_id}' is already busy with task {active_task_for_backend}. Ignoring new request.")
            self.response_error.emit(target_backend_id, "Backend busy. Please wait.")
            return

        adapter = self._backend_adapters.get(target_backend_id)
        if not adapter:
            logger.error(f"BC: Cannot request response: No adapter for backend_id '{target_backend_id}'.")
            self.response_error.emit(target_backend_id, f"Backend adapter '{target_backend_id}' not found."); return
        if not self._is_configured_map.get(target_backend_id, False):
            logger.error(f"BC: Cannot request response: Backend '{target_backend_id}' is not configured.")
            self.response_error.emit(target_backend_id, f"Backend '{target_backend_id}' not configured."); return

        logger.info(
            f"BC: Creating task for backend '{target_backend_id}'. Mod expected: {is_modification_response_expected}. Options: {options}, Metadata: {request_metadata}")
        task = asyncio.create_task(
            self._internal_get_response_stream(target_backend_id, adapter, history_to_send,
                                               is_modification_response_expected, options, request_metadata) # <-- PASS request_metadata
        )
        logger.info(
            f"BC: Adding task for '{target_backend_id}' to _active_backend_tasks. Current tasks before add: {list(self._active_backend_tasks.keys())}")
        self._active_backend_tasks[target_backend_id] = task
        self._update_overall_busy_state()
        logger.info(
            f"BC: Task for '{target_backend_id}' ({task}) added. _active_backend_tasks now: {list(self._active_backend_tasks.keys())}")

    # --- MODIFIED: Added request_metadata parameter & processing ---
    async def _internal_get_response_stream(self, backend_id: str, adapter: BackendInterface,
                                            history: List[ChatMessage], is_modification_response_expected: bool,
                                            options: Optional[Dict[str, Any]] = None,
                                            request_metadata: Optional[Dict[str, Any]] = None): # <-- NEW PARAMETER
        logger.info(
            f"Task _internal_get_response_stream for backend '{backend_id}' ({asyncio.current_task()}) started. History: {len(history)}, Options: {options}, Metadata: {request_metadata}")
        response_buffer = ""
        streaming_started_for_ui = False
        usage_stats_dict: Dict[str, Any] = {} # For token counts and now also request_metadata

        try:
            if not hasattr(adapter, 'get_response_stream'):
                raise AttributeError(f"Backend adapter '{backend_id}' missing get_response_stream method.")
            if history: logger.debug(f"Calling backend '{backend_id}' stream with prompt: '{history[-1].text[:100]}...'")
            else: logger.warning(f"Calling backend '{backend_id}' stream with empty history!")

            stream_iterator = adapter.get_response_stream(history, options) # Pass options

            async for chunk in stream_iterator:
                if not is_modification_response_expected:
                    if not streaming_started_for_ui: streaming_started_for_ui = True
                    self.stream_chunk_received.emit(backend_id, chunk)
                response_buffer += chunk

            logger.info(f"Backend stream iterator for '{backend_id}' finished normally.")
            final_response_text = response_buffer.strip()

            # Fetch token usage
            if hasattr(adapter, 'get_last_token_usage'):
                token_usage: Optional[Tuple[int, int]] = adapter.get_last_token_usage()
                if token_usage:
                    usage_stats_dict["prompt_tokens"] = token_usage[0]
                    usage_stats_dict["completion_tokens"] = token_usage[1]
                    logger.info(f"BC: Fetched token usage for '{backend_id}': Prompt={token_usage[0]}, Comp={token_usage[1]}")
                else:
                    logger.warning(f"BC: Adapter '{backend_id}' get_last_token_usage returned None.")
            else:
                logger.warning(f"BC: Adapter '{backend_id}' does not have get_last_token_usage method.")

            # --- NEW: Merge request_metadata into usage_stats_dict ---
            if request_metadata:
                logger.debug(f"BC: Merging request_metadata {request_metadata} into usage_stats for backend '{backend_id}'.")
                usage_stats_dict.update(request_metadata) # This adds/overwrites keys in usage_stats_dict
            # --- END NEW ---

            if final_response_text:
                completed_message = ChatMessage(role=MODEL_ROLE, parts=[final_response_text])
                self.response_completed.emit(backend_id, completed_message, usage_stats_dict) # Emit with augmented usage_stats
                logger.info(f"Emitted response_completed for '{backend_id}' with final message and stats: {usage_stats_dict}.")
            elif is_modification_response_expected or streaming_started_for_ui:
                logger.warning(f"Stream for '{backend_id}' finished, but final buffer was empty.")
                if is_modification_response_expected:
                    self.response_error.emit(backend_id, "[System: AI returned an empty response during modification step.]")
                else:
                    empty_msg = ChatMessage(role=MODEL_ROLE, parts=["[AI returned an empty response]"])
                    self.response_completed.emit(backend_id, empty_msg, usage_stats_dict) # Emit with augmented usage_stats
            else:
                backend_error = adapter.get_last_error()
                err_msg_text = backend_error if backend_error else f"AI for '{backend_id}' failed before streaming or returned no response"
                logger.warning(f"Backend stream issue for '{backend_id}': {err_msg_text}")
                self.response_error.emit(backend_id, err_msg_text)

        except asyncio.CancelledError:
            logger.info(f"BackendCoordinator task for '{backend_id}' ({asyncio.current_task()}) cancelled.")
            self.response_error.emit(backend_id, "[AI response cancelled by user]")
        except Exception as e:
            logger.exception(
                f"Error during BackendCoordinator's _internal_get_response_stream task for '{backend_id}' ({asyncio.current_task()}):")
            error_msg = adapter.get_last_error() or f"Backend Task Error ({backend_id}): {type(e).__name__}"
            # If the error has metadata (e.g., from a summary request failure), pass it along
            if request_metadata:
                 # We'll emit the textual error, and ChatManager can look at the 'purpose' if it's a summary error
                 # For now, the response_error signal doesn't carry the metadata directly.
                 # The error_msg itself should be descriptive enough for ChatManager to act if needed.
                 pass
            self.response_error.emit(backend_id, error_msg)
        finally:
            logger.info(f"BC Task Finally: Task for '{backend_id}' ({asyncio.current_task()}) finishing...")
            task_instance_in_dict = self._active_backend_tasks.get(backend_id)
            if task_instance_in_dict is asyncio.current_task():
                del self._active_backend_tasks[backend_id]
            elif task_instance_in_dict:
                logger.warning(f"BC Task Finally: Task instance for '{backend_id}' in dict ({task_instance_in_dict}) does NOT match current task ({asyncio.current_task()}). Not deleting.")
            else:
                logger.warning(f"BC Task Finally: Backend ID '{backend_id}' not found in _active_backend_tasks.")
            self._update_overall_busy_state()


    def cancel_current_task(self, backend_id: Optional[str] = None):
        if backend_id:
            task = self._active_backend_tasks.get(backend_id)
            if task and not task.done(): logger.info(f"BC: Cancelling task {task} for backend '{backend_id}'..."); task.cancel()
            else: logger.debug(f"BC: No active task for backend '{backend_id}' to cancel (Task: {task}).")
        else:
            logger.info("BC: Cancelling ALL active backend tasks...")
            for bid, task_to_cancel in list(self._active_backend_tasks.items()): # Iterate over a copy
                if task_to_cancel and not task_to_cancel.done():
                    logger.info(f"  Cancelling task {task_to_cancel} for backend '{bid}'..."); task_to_cancel.cancel()
            logger.debug("Requested cancellation for all active tasks.")


    def is_backend_configured(self, backend_id: str) -> bool:
        return self._is_configured_map.get(backend_id, False)

    def get_last_error_for_backend(self, backend_id: str) -> Optional[str]:
        adapter = self._backend_adapters.get(backend_id)
        if adapter:
            direct_error = adapter.get_last_error() # type: ignore
        else:
            direct_error = None # Ensure direct_error is defined
        if direct_error: return direct_error
        return self._last_errors_map.get(backend_id)

    def is_processing_request(self) -> bool:
        return self._overall_is_busy

    def get_current_configured_model(self, backend_id: str) -> Optional[str]:
        return self._current_model_names.get(backend_id)

    def get_current_system_prompt(self, backend_id: str) -> Optional[str]:
        return self._current_system_prompts.get(backend_id)