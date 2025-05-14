# core/user_input_processor.py
import logging
import os
from typing import List, Optional, Dict, Any, Tuple, Set, Union, NamedTuple, TYPE_CHECKING

# --- Corrected Imports for core components ---
from core.models import ChatMessage, USER_ROLE  # Keep USER_ROLE if needed for other actions

# Forward references for type hinting to avoid circular dependencies if RagHandler/ModHandler import UIP
if TYPE_CHECKING:
    from core.rag_handler import RagHandler
    from core.modification_handler import ModificationHandler
# --- End Corrected Imports ---

from utils import constants  # Assuming constants is in utils at the project root

logger = logging.getLogger(__name__)


class ProcessResult(NamedTuple):
    action_type: str
    prompt_or_history: Union[str, List[ChatMessage], str]  # Payload: augmented msg, command string, or project_id
    original_query: Optional[str] = None
    original_context: Optional[str] = None
    original_focus_prefix: Optional[str] = None
    # Added for UserInputHandler to get the original message ID if UIP creates one
    # (though for summary, UIP doesn't create a ChatMessage for the user's command)
    # original_query_message_id: Optional[str] = None


class UserInputProcessor:
    _NEXT_COMMANDS = {"next", "ok", "okay", "continue", "yes", "proceed", "go", "next file"}
    _STRONG_MODIFICATION_KEYWORDS = {
        "refactor", "restructure", "reorganize my project", "overhaul",
        "implement feature across", "integrate throughout", "update all instances of"
    }
    _GENERAL_MODIFICATION_KEYWORDS = {
        "change", "update", "modify", "apply", "implement", "add", "fix", "remove",
        "create", "generate files for"
    }
    _PROJECT_SUMMARY_KEYWORDS = {
        "summarize project", "project summary", "summarise project", "project overview",
        "overview of project", "tell me about this project", "what's in this project",
        "explain this project", "summarize rag", "rag summary", "summarise rag",
        "ava summarize", "ava project summary", "ava project overview"
    }

    def __init__(self, rag_handler: Optional['RagHandler'], modification_handler: Optional['ModificationHandler']):
        self._rag_handler = rag_handler
        self._modification_handler = modification_handler
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
        query_lower = user_query_text.lower().strip()

        # Path 0: Check for Project Summary Request
        if any(keyword in query_lower for keyword in self._PROJECT_SUMMARY_KEYWORDS):
            is_direct_summary_command = False
            for kw in self._PROJECT_SUMMARY_KEYWORDS:
                if query_lower == kw or \
                        query_lower.startswith(kw + " ") or \
                        query_lower.startswith(kw + ",") or \
                        query_lower.startswith(kw + "."):
                    is_direct_summary_command = True
                    break

            if is_direct_summary_command:
                logger.info(
                    f"Project summary request detected for project: '{current_project_id}'. Query: '{user_query_text}'")
                return ProcessResult(
                    action_type="REQUEST_PROJECT_SUMMARY",
                    prompt_or_history=current_project_id or "",
                    original_query=user_query_text
                )

        # Path 1: A modification sequence is ALREADY active
        if is_modification_active and self._modification_handler:
            logger.debug("Processing input during active modification sequence.")
            is_next_command = user_query_text.lower() in self._NEXT_COMMANDS
            if is_next_command:
                return ProcessResult(action_type="NEXT_MODIFICATION",
                                     prompt_or_history=user_query_text)
            else:
                return ProcessResult(action_type="REFINE_MODIFICATION",
                                     prompt_or_history=user_query_text)

        # Path 2: No modification sequence is active; determine if this input should START one
        else:
            logger.debug("Processing input for normal chat or potential modification start.")
            is_potential_modification_request = False
            if self._modification_handler:  # Check if mod handler is available
                if any(kw in query_lower for kw in self._STRONG_MODIFICATION_KEYWORDS):
                    is_potential_modification_request = True
                else:
                    has_general_mod_keyword = any(kw in query_lower for kw in self._GENERAL_MODIFICATION_KEYWORDS)
                    has_code_like_chars = any(
                        c in user_query_text for c in ['/', '\\', '.py', '()', '{}', 'class ', 'def '])
                    mentions_file_or_project = "file" in query_lower or ".py" in query_lower or "project" in query_lower or "module" in query_lower
                    if has_general_mod_keyword and (focus_paths or has_code_like_chars or mentions_file_or_project):
                        is_potential_modification_request = True

                if is_potential_modification_request:
                    # Avoid downgrading too easily if strong intent or rich context
                    is_short_and_ambiguous = len(user_query_text.split()) < 5 and \
                                             not any(kw in query_lower for kw in self._STRONG_MODIFICATION_KEYWORDS) and \
                                             not focus_paths
                    if is_short_and_ambiguous:
                        pass
                        logger.info(
                            f"Query '{user_query_text[:30]}' is short & lacks strong cues for mod, but proceeding with mod check.")

            if is_potential_modification_request and self._modification_handler:
                logger.info("Potential modification request identified. Preparing for START_MODIFICATION.")
                rag_context_str_for_mod, focus_prefix_for_mod = self._get_rag_and_focus(
                    query=user_query_text, is_modification=True,
                    current_project_id=current_project_id, focus_paths=focus_paths,
                    rag_available=rag_available, rag_initialized=rag_initialized
                )
                return ProcessResult(
                    action_type="START_MODIFICATION", prompt_or_history=[],
                    original_query=user_query_text, original_context=rag_context_str_for_mod,
                    original_focus_prefix=focus_prefix_for_mod
                )
            # Otherwise, it's a normal chat message
            else:
                logger.info("Processing as normal chat message.")
                rag_context_str_for_chat, focus_prefix_for_chat = self._get_rag_and_focus(
                    query=user_query_text, is_modification=False,
                    current_project_id=current_project_id, focus_paths=focus_paths,
                    rag_available=rag_available, rag_initialized=rag_initialized
                )
                final_text_for_llm = self._prepare_normal_chat_prompt(user_query_text, rag_context_str_for_chat,
                                                                      focus_prefix_for_chat)
                final_parts = [final_text_for_llm] + (image_data or [])
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

        should_rag = False
        if is_modification:
            should_rag = rag_available and rag_initialized
        else:
            should_rag = self._rag_handler.should_perform_rag(query, rag_available, rag_initialized)

        if should_rag:
            query_entities = self._rag_handler.extract_code_entities(query)
            rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                query=query, query_entities=query_entities, project_id=current_project_id,
                focus_paths=focus_paths, is_modification_request=is_modification
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
            return final_prompt
        else:
            return user_query