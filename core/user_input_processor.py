# core/user_input_processor.py
# UPDATED - Enhanced logic to better identify multi-file modification requests.

import logging
import os
from typing import List, Optional, Dict, Any, Tuple, Set, Union, NamedTuple

from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from .rag_handler import RagHandler
from .modification_handler import ModificationHandler
from utils import constants

logger = logging.getLogger(__name__)


class ProcessResult(NamedTuple):
    action_type: str
    prompt_or_history: Union[str, List[ChatMessage]] # For NORMAL_CHAT, this is the augmented message. For others, it's specific payload.
    original_query: Optional[str] = None
    original_context: Optional[str] = None
    original_focus_prefix: Optional[str] = None


class UserInputProcessor:
    _NEXT_COMMANDS = {"next", "ok", "okay", "continue", "yes", "proceed", "go", "next file"}

    # --- MODIFICATION START: More nuanced keyword lists ---
    _STRONG_MODIFICATION_KEYWORDS = { # Keywords strongly indicating a multi-file task
        "refactor", "restructure", "reorganize my project", "overhaul",
        "implement feature across", "integrate throughout", "update all instances of"
    }
    _GENERAL_MODIFICATION_KEYWORDS = { # Keywords that *could* be part of a modification
        "change", "update", "modify", "apply", "implement", "add", "fix", "remove",
        "create", "generate files for"
    }
    # --- MODIFICATION END ---

    def __init__(self, rag_handler: Optional[RagHandler], modification_handler: Optional[ModificationHandler]):
        self._rag_handler = rag_handler
        self._modification_handler = modification_handler # Still needed for preparing instructions later
        logger.info("UserInputProcessor initialized.")
        if not self._rag_handler: logger.warning("UserInputProcessor: RagHandler not provided.")
        if not self._modification_handler: logger.warning("UserInputProcessor: ModificationHandler not provided.")

    def process(self,
                user_query_text: str,
                image_data: List[Dict[str, Any]], # Images are generally not for modification flow
                is_modification_active: bool,
                current_project_id: Optional[str],
                focus_paths: Optional[List[str]],
                rag_available: bool,
                rag_initialized: bool) -> ProcessResult:
        logger.debug(
            f"UserInputProcessor processing. Mod Active: {is_modification_active}, Query: '{user_query_text[:50]}...'")

        # --- Path 1: A modification sequence is ALREADY active ---
        if is_modification_active and self._modification_handler:
            logger.debug("Processing input during active modification sequence.")
            is_next_command = user_query_text.lower() in self._NEXT_COMMANDS
            if is_next_command:
                return ProcessResult(action_type="NEXT_MODIFICATION",
                                     prompt_or_history=user_query_text) # Send command as payload
            else:
                # Any other input is considered feedback/refinement for the current step
                return ProcessResult(action_type="REFINE_MODIFICATION",
                                     prompt_or_history=user_query_text) # Send feedback as payload

        # --- Path 2: No modification sequence is active; determine if this input should START one ---
        else:
            logger.debug("Processing input for normal chat or potential modification start.")
            query_lower = user_query_text.lower()
            is_potential_modification_request = False

            if self._modification_handler: # Ensure handler is available for mod logic
                # Check for strong keywords first
                if any(kw in query_lower for kw in self._STRONG_MODIFICATION_KEYWORDS):
                    is_potential_modification_request = True
                    logger.debug(f"Strong modification keyword found. Potential mod: {is_potential_modification_request}")
                else:
                    # Check for general keywords combined with contextual clues
                    has_general_mod_keyword = any(kw in query_lower for kw in self._GENERAL_MODIFICATION_KEYWORDS)
                    has_code_like_chars = any(
                        c in user_query_text for c in ['/', '\\', '.py', '()', '{}', 'class ', 'def ']
                    )
                    mentions_file_or_project = "file" in query_lower or ".py" in query_lower or "project" in query_lower or "module" in query_lower

                    if has_general_mod_keyword and (focus_paths or has_code_like_chars or mentions_file_or_project):
                        is_potential_modification_request = True
                    logger.debug(
                        f"General mod check: Keywords={has_general_mod_keyword}, Focus={bool(focus_paths)}, "
                        f"CodeChars={has_code_like_chars}, FileMention={mentions_file_or_project} -> Potential={is_potential_modification_request}"
                    )

                # --- MODIFICATION START: Adjust downgrade logic ---
                # Avoid downgrading too easily if strong intent is present or context is rich
                if is_potential_modification_request:
                    # If it's a very short query without strong keywords AND no focus/RAG, it might be ambiguous
                    is_short_and_ambiguous = len(user_query_text.split()) < 5 and \
                                             not any(kw in query_lower for kw in self._STRONG_MODIFICATION_KEYWORDS) and \
                                             not focus_paths # RAG context will be fetched next, so don't use it for this check yet

                    if is_short_and_ambiguous:
                        # Let's get RAG context first before deciding to downgrade.
                        # If RAG context is found, it might clarify that it's a normal chat question about the code.
                        # If no RAG context, and still short/ambiguous, then maybe downgrade.
                        # For now, we'll proceed assuming it *could* be a modification if keywords matched.
                        # The ModificationCoordinator's planner can also decide if no files are needed.
                        pass # Don't downgrade yet, let MC's planner decide if no files.
                        logger.info(f"Query '{user_query_text[:30]}' is short & lacks strong cues, but proceeding with mod check.")
                # --- MODIFICATION END ---


            # --- If it's a potential modification, prepare for that flow ---
            if is_potential_modification_request and self._modification_handler:
                logger.info("Potential modification request identified. Preparing for START_MODIFICATION.")
                # Get RAG/Focus specifically for the modification task
                rag_context_str_for_mod, focus_prefix_for_mod = self._get_rag_and_focus(
                    query=user_query_text,
                    is_modification=True, # Signal that this RAG is for a modification task
                    current_project_id=current_project_id,
                    focus_paths=focus_paths,
                    rag_available=rag_available,
                    rag_initialized=rag_initialized
                )
                return ProcessResult(
                    action_type="START_MODIFICATION",
                    prompt_or_history=[], # No immediate LLM call from UIP for this action
                    original_query=user_query_text,
                    original_context=rag_context_str_for_mod,
                    original_focus_prefix=focus_prefix_for_mod
                )

            # --- Otherwise, it's a normal chat message ---
            else:
                logger.info("Processing as normal chat message.")
                # Get RAG/Focus for a general chat query
                rag_context_str_for_chat, focus_prefix_for_chat = self._get_rag_and_focus(
                    query=user_query_text,
                    is_modification=False, # Signal that this RAG is for normal chat
                    current_project_id=current_project_id,
                    focus_paths=focus_paths,
                    rag_available=rag_available,
                    rag_initialized=rag_initialized
                )
                final_text_for_llm = self._prepare_normal_chat_prompt(user_query_text, rag_context_str_for_chat, focus_prefix_for_chat)
                final_parts = [final_text_for_llm] + (image_data or []) # image_data is usually empty for code chat
                message_for_backend = ChatMessage(role=USER_ROLE, parts=final_parts)
                return ProcessResult(action_type="NORMAL_CHAT", prompt_or_history=[message_for_backend])


    def _get_rag_and_focus(self, query: str, is_modification: bool, current_project_id: Optional[str],
                           focus_paths: Optional[List[str]], rag_available: bool, rag_initialized: bool) -> Tuple[
        str, str]:
        rag_context_str = ""
        focus_prefix = ""

        if not self._rag_handler:
            logger.warning("RagHandler not available, cannot perform RAG or generate focus prefix.")
            return rag_context_str, focus_prefix

        # Determine if RAG should be performed
        # For modifications, we almost always want to try RAG.
        # For normal chat, use the standard should_perform_rag check.
        should_rag = False
        if is_modification:
            should_rag = rag_available and rag_initialized # If RAG is usable, use it for modifications
            logger.debug(f"RAG Check (for modification): should_rag={should_rag} (rag_available={rag_available}, rag_initialized={rag_initialized})")
        else: # Normal chat
            should_rag = self._rag_handler.should_perform_rag(query, rag_available, rag_initialized)
            logger.debug(
                f"RAG Check (for normal chat): should_rag={should_rag} (rag_available={rag_available}, rag_initialized={rag_initialized})")

        if should_rag:
            query_entities = self._rag_handler.extract_code_entities(query)
            rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                query=query,
                query_entities=query_entities,
                project_id=current_project_id,
                focus_paths=focus_paths,
                is_modification_request=is_modification # Pass the flag here
            )
        else:
            logger.debug("RAG not performed for this query based on checks.")

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

    # Modification related prompt preparations are not used directly by UIP to make LLM calls anymore.
    # They are now implicitly handled by the action_types "NEXT_MODIFICATION" and "REFINE_MODIFICATION"
    # which signal the ModificationCoordinator to use its own logic (and ModificationHandler)
    # to prepare prompts for the specialized LLMs.

    # def _prepare_modification_next(self) -> ProcessResult: ... (No longer needed here)
    # def _prepare_modification_feedback(self, user_feedback_text: str,
    #                                    image_data: List[Dict[str, Any]]) -> ProcessResult: ... (No longer needed here)