# core/user_input_processor.py
import logging
import os
from typing import List, Optional, Dict, Any, Tuple, Set, Union, NamedTuple

from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from .rag_handler import RagHandler
from .modification_handler import ModificationHandler  # Keep for other uses if any
from utils import constants

logger = logging.getLogger(__name__)


class ProcessResult(NamedTuple):
    action_type: str
    prompt_or_history: Union[str, List[ChatMessage]]
    original_query: Optional[str] = None
    original_context: Optional[str] = None
    original_focus_prefix: Optional[str] = None


class UserInputProcessor:
    _NEXT_COMMANDS = {"next", "ok", "okay", "continue", "yes", "proceed", "go", "next file"}
    _MODIFICATION_KEYWORDS = {"refactor", "change", "update", "modify", "apply", "implement", "add", "fix", "remove",
                              "create"}

    def __init__(self, rag_handler: Optional[RagHandler], modification_handler: Optional[ModificationHandler]):
        self._rag_handler = rag_handler
        self._modification_handler = modification_handler  # Still needed for preparing instructions later
        logger.info("UserInputProcessor initialized.")
        if not self._rag_handler: logger.warning("UserInputProcessor: RagHandler not provided.")
        if not self._modification_handler: logger.warning("UserInputProcessor: ModificationHandler not provided.")

    def process(self,
                user_query_text: str,
                image_data: List[Dict[str, Any]],
                is_modification_active: bool,
                current_project_id: Optional[str],
                focus_paths: Optional[List[str]],
                rag_available: bool,
                rag_initialized: bool) -> ProcessResult:
        logger.debug(
            f"UserInputProcessor processing. Mod Active: {is_modification_active}, Query: '{user_query_text[:50]}...'")

        if is_modification_active and self._modification_handler:
            logger.debug("Processing input during active modification sequence.")
            is_next_command = user_query_text.lower() in self._NEXT_COMMANDS
            if is_next_command:
                # For "next", UIP doesn't prepare an LLM prompt itself.
                # It signals ChatManager (via UIH) to advance the ModificationCoordinator.
                # The prompt_or_history can be empty or a marker.
                return ProcessResult(action_type="NEXT_MODIFICATION",
                                     prompt_or_history=user_query_text)  # Send command as payload
            else:
                # For refinement, UIP also doesn't prepare the LLM prompt directly.
                # It signals ChatManager to pass feedback to ModificationCoordinator.
                return ProcessResult(action_type="REFINE_MODIFICATION",
                                     prompt_or_history=user_query_text)  # Send feedback as payload

        else:  # Normal Chat or Start Modification
            logger.debug("Processing input for normal chat or potential modification start.")
            is_potential_modification_request = False
            if self._modification_handler:  # Ensure handler is available for mod logic
                query_lower = user_query_text.lower()
                has_mod_keyword = any(kw in query_lower for kw in self._MODIFICATION_KEYWORDS)
                has_code_like_chars = any(
                    c in user_query_text for c in ['/', '\\', '.py', '()', '{}', 'class ', 'def '])
                mentions_file = "file" in query_lower or ".py" in query_lower

                if has_mod_keyword and (focus_paths or has_code_like_chars or mentions_file):
                    is_potential_modification_request = True
                if len(user_query_text.split()) < 4 and not focus_paths and not has_code_like_chars:
                    is_potential_modification_request = False
                logger.debug(
                    f"Potential modification check: Keywords={has_mod_keyword}, Focus={bool(focus_paths)}, CodeChars={has_code_like_chars}, FileMention={mentions_file} -> Potential={is_potential_modification_request}")

            rag_context_str, focus_prefix = self._get_rag_and_focus(
                query=user_query_text,
                is_modification=is_potential_modification_request,
                current_project_id=current_project_id,
                focus_paths=focus_paths,
                rag_available=rag_available,
                rag_initialized=rag_initialized
            )

            if is_potential_modification_request and (
                    mentions_file or has_code_like_chars) and not rag_context_str and not focus_paths:
                logger.info(
                    "Potential modification request mentions files/code but no RAG/Focus context. Treating as normal chat.")
                is_potential_modification_request = False

            if is_potential_modification_request and self._modification_handler:
                logger.info("Potential modification request confirmed by UIP. Emitting intent to start sequence.")
                # ---- MODIFIED SECTION ----
                # UIP no longer calls modification_handler.start_modification.
                # It just prepares the data for UserInputHandler to signal ChatManager.
                return ProcessResult(
                    action_type="START_MODIFICATION",  # Signal intent to start
                    prompt_or_history=[],  # No immediate LLM call from UIP for this action
                    original_query=user_query_text,
                    original_context=rag_context_str,
                    original_focus_prefix=focus_prefix
                )
                # ---- END MODIFIED SECTION ----
            else:  # Normal Chat Message Flow
                logger.info("Processing as normal chat message.")
                final_text_for_llm = self._prepare_normal_chat_prompt(user_query_text, rag_context_str, focus_prefix)
                final_parts = [final_text_for_llm] + (image_data or [])
                message_for_backend = ChatMessage(role=USER_ROLE, parts=final_parts)
                # History for normal chat will be the single new user message, plus RAG context.
                # ChatManager will combine this with actual project history before sending to backend.
                return ProcessResult(action_type="NORMAL_CHAT", prompt_or_history=[message_for_backend])

    def _get_rag_and_focus(self, query: str, is_modification: bool, current_project_id: Optional[str],
                           focus_paths: Optional[List[str]], rag_available: bool, rag_initialized: bool) -> Tuple[
        str, str]:
        rag_context_str = ""
        focus_prefix = ""
        if self._rag_handler:
            should_rag = self._rag_handler.should_perform_rag(query, rag_available, rag_initialized) or is_modification
            logger.debug(
                f"RAG Check: should_rag={should_rag} (rag_available={rag_available}, rag_initialized={rag_initialized}, is_modification={is_modification})")
            if should_rag:
                query_entities = self._rag_handler.extract_code_entities(query)
                rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                    query=query,
                    query_entities=query_entities,
                    project_id=current_project_id,
                    focus_paths=focus_paths,
                    is_modification_request=is_modification
                )
            else:
                logger.debug("RAG not performed for this query.")
        else:
            logger.warning("RagHandler not available, cannot perform RAG.")

        if focus_paths:
            num_focused = len(focus_paths)
            focus_prefix = "[Focus on:\n"
            focus_prefix += "\n".join([f"- `{os.path.basename(p)}`" for p in focus_paths])
            focus_prefix += "\n]\n\n"
            logger.debug(f"Generated focus prefix for {num_focused} paths.")
        return rag_context_str, focus_prefix

    def _prepare_normal_chat_prompt(self, user_query: str, rag_context: str, focus_prefix: str) -> str:
        if rag_context or focus_prefix:
            prompt_template = (
                "{focus}"
                "User Query: {query}\n\n"
                "{context_section}"
                "Based on the above query and context (if provided), please respond."
            )
            context_section = f"Relevant Context:\n{rag_context}\n\n" if rag_context else ""
            final_prompt = prompt_template.format(focus=focus_prefix, query=user_query, context_section=context_section)
            logger.debug(f"Prepared normal chat prompt with context/focus. Length: {len(final_prompt)}")
            return final_prompt
        else:
            logger.debug("Prepared normal chat prompt without context/focus.")
            return user_query

    # Modification related prompt preparations remain as they are primarily for constructing
    # user-facing prompts that the ModificationCoordinator will then process,
    # rather than direct LLM calls from UIP.
    def _prepare_modification_next(self) -> ProcessResult:
        logger.info("UIP: Preparing for 'next' modification step (signal to MC).")
        # This action type will be handled by UIH -> ChatManager -> ModificationCoordinator
        return ProcessResult(action_type="NEXT_MODIFICATION", prompt_or_history="")

    def _prepare_modification_feedback(self, user_feedback_text: str,
                                       image_data: List[Dict[str, Any]]) -> ProcessResult:
        logger.info("UIP: Preparing for 'refine' modification step (signal to MC).")
        # This action type will be handled by UIH -> ChatManager -> ModificationCoordinator
        # The user_feedback_text is the payload here.
        # Images are not directly used by MC for refinement prompts yet, but kept for completeness.
        return ProcessResult(action_type="REFINE_MODIFICATION", prompt_or_history=user_feedback_text)