# core/modification_handler.py
import logging
import re
from typing import List, Tuple, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class ModificationHandler(QObject):
    code_file_ready = pyqtSignal(str, str)
    status_message_ready = pyqtSignal(str)
    modification_parsing_error = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._last_emitted_filename: Optional[str] = None
        self._is_active: bool = False
        logger.info("ModificationHandler initialized (for CodeLlama interaction).")

    def activate_sequence(self):
        logger.info("ModificationHandler activated for a sequence.")
        self.cancel_modification()  # Reset state if previously active
        self._is_active = True

    def cancel_modification(self):
        if self._is_active:
            logger.info("ModificationHandler resetting/cancelling state.")
        self._last_emitted_filename = None
        self._is_active = False

    def is_active(self) -> bool:
        return self._is_active

    def prepare_standard_codellama_instruction(
            self,
            target_filename: str,
            original_user_query: str,
            full_plan: List[str],
            original_file_content: Optional[str] = None  # <-- NEW PARAMETER
    ) -> str:
        if not self._is_active:
            logger.warning("MH: prepare_standard_instruction called when not active.")
            return "[ERROR: Handler not active]"

        logger.debug(
            f"MH: Preparing standard CodeLlama instruction for: {target_filename}. Original content provided: {original_file_content is not None}")
        planned_files_str = ", ".join(f"'{f}'" for f in full_plan)
        if not planned_files_str:
            planned_files_str = "[No other files in plan or plan not specified]"

        if original_file_content:
            task_description = f"Your current task is to UPDATE the existing file: `{target_filename}`."
            content_section = (
                f"Here is the current content of `{target_filename}`. You MUST analyze this existing code carefully:\n"
                f"```python\n"  # Assuming python, adjust if language can vary and is known
                f"{original_file_content}\n"
                f"```\n\n"
                f"Please provide the ENTIRE, UPDATED source code for `{target_filename}` incorporating the necessary changes based on the user's request. "
                f"**It is CRUCIAL that you preserve all existing code that is NOT directly affected by the user's request.** "
                f"Do not omit any unchanged parts of the original file. Your output must be the complete file."
            )
            example_format_guidance = (
                f"Example of expected output format (remember to include ALL code, changed and unchanged):\n"
                f"```python {target_filename}\n"
                f"# Full UPDATED code for {target_filename} goes here...\n"
                f"# ... including preserved unchanged original parts ...\n"
                f"# ... and the new/modified parts ...\n"
                f"```"
            )
        else:
            task_description = f"Your current task is to GENERATE the complete and correct code for a NEW file: `{target_filename}`."
            content_section = f"Please provide the ENTIRE source code for the new file `{target_filename}` based on the user's request."
            example_format_guidance = (
                f"Example of expected output format:\n"
                f"```python {target_filename}\n"
                f"# Full NEW code for {target_filename} goes here...\n"
                f"```"
            )

        instruction = (
            f"You are an expert Python coding assistant. {task_description}\n"
            f"This modification is part of a larger plan based on the user's request: \"{original_user_query}\"\n"
            f"The overall plan involves changes to the following file(s): {planned_files_str}.\n\n"
            f"{content_section}\n\n"
            f"Ensure your output is ONLY the code itself, formatted correctly.\n"
            f"**CRITICAL OUTPUT FORMAT:**\n"
            f"Your response MUST contain ONLY a single standard Markdown fenced code block. This block MUST be labeled with the exact filename `{target_filename}` immediately after the opening ``` backticks.\n"
            f"{example_format_guidance}\n"
            f"DO NOT include ANY other text, explanations, summaries, greetings, apologies, acknowledgements, or any other conversational fluff whatsoever, neither before nor after the single required code block. "
            f"Your entire response must be ONLY the code block for `{target_filename}`."
        )
        return instruction

    def prepare_codellama_refinement_instruction(
            self,
            target_filename: str,
            user_feedback: str,
            previous_llm_instruction: str
            # Consider adding original_file_content here too if refinement needs it directly
    ) -> str:
        if not self._is_active:
            logger.warning("MH: prepare_refinement_instruction called when not active.")
            return "[ERROR: Handler not active]"
        logger.debug(f"MH: Preparing CodeLlama refinement instruction for: {target_filename}")
        # This instruction is for Gemini to *refine* a prompt for CodeLlama.
        # The `previous_llm_instruction` (which is the `standard_instruction`)
        # will already contain the original file content if it was an update.
        instruction = (
            f"You are an expert Python coding assistant. We are refining the file: `{target_filename}`.\n"
            f"The user has provided the following feedback or additional request: \"{user_feedback}\"\n\n"
            f"The previous instruction prepared for the code generation LLM (which includes the original file content if applicable, and the user's main goal) was:\n"
            f"--- PREVIOUS INSTRUCTION START ---\n{previous_llm_instruction}\n--- PREVIOUS INSTRUCTION END ---\n\n"
            f"Please provide the NEW, COMPLETE, and CORRECTED source code for `{target_filename}`, incorporating the user's feedback and ensuring all necessary functionality. "
            f"If updating, remember the importance of preserving unchanged parts of the original code.\n"
            f"**CRITICAL OUTPUT FORMAT (Same as before):**\n"
            f"Your response MUST contain ONLY a single standard Markdown fenced code block, labeled with the exact filename `{target_filename}`.\n"
            f"Example:\n"
            f"```python {target_filename}\n"
            f"# Full REVISED code for {target_filename} incorporating feedback...\n"
            f"```\n"
            f"DO NOT include ANY other text, explanations, summaries, or conversational elements. "
            f"ONLY the revised code block for `{target_filename}`."
        )
        return instruction

    def process_llm_code_generation_response(self, response_text: str, expected_filename: str) -> bool:
        if not self._is_active:
            logger.warning(
                f"MH: process_llm_code_generation_response called for '{expected_filename}' when not active.")
            return False

        logger.info(f"MH: Processing Generator AI response, expecting file: '{expected_filename}'")
        parsed_file_tuple = self._parse_first_code_block_lenient(response_text, expected_filename)

        if not parsed_file_tuple:
            err_msg = f"[System: Generator AI response format error. Expected a single Markdown code block labeled with filename '{expected_filename}'. Response did not contain a recognizable code block or the label was incorrect. Response preview:\n{response_text[:300]}...]"
            logger.error(err_msg)
            self.modification_parsing_error.emit(err_msg)
            return False

        parsed_filename, content = parsed_file_tuple
        self._last_emitted_filename = parsed_filename

        self.code_file_ready.emit(parsed_filename, content)
        self.status_message_ready.emit(
            f"[System: Code for '{parsed_filename}' received from Generator AI. Review and provide feedback or type 'next'.]")
        logger.info(f"MH: Successfully parsed and emitted code for '{parsed_filename}'.")
        return True

    def _parse_first_code_block_lenient(self, text_to_parse: str, expected_filename: str) -> Optional[Tuple[str, str]]:
        escaped_expected_filename = re.escape(expected_filename)
        specific_pattern = rf"```(?:[a-zA-Z0-9_\-\.]*)?\s*{escaped_expected_filename}\s*\n(.*?)\n```"
        match = re.search(specific_pattern, text_to_parse, re.DOTALL | re.IGNORECASE)

        if match:
            filepath = expected_filename
            content = match.group(1)
        else:
            logger.debug(f"MH_Lenient: Specific pattern failed for '{expected_filename}'. Trying generic python block.")
            generic_pattern = r"```python\s*\n(.*?)\n```"
            match = re.search(generic_pattern, text_to_parse, re.DOTALL | re.IGNORECASE)
            if match:
                filepath = expected_filename
                content = match.group(1)
                logger.warning(
                    f"MH_Lenient: Matched generic 'python' block for '{expected_filename}'. Assuming content is correct.")
            else:
                logger.debug(f"MH_Lenient: Generic python block failed. Trying any code block.")
                any_code_block_pattern = r"```(?:[a-zA-Z0-9_\-\.+]+)?\s*\n(.*?)\n```"
                match = re.search(any_code_block_pattern, text_to_parse, re.DOTALL | re.IGNORECASE)
                if match:
                    filepath = expected_filename
                    content = match.group(1)
                    logger.warning(
                        f"MH_Lenient: Matched ANY code block for '{expected_filename}'. Assuming content is correct.")
                else:
                    logger.warning("MH_Lenient: No code block found even with lenient parsing.")
                    return None

        if content.startswith('\n'): content = content[1:]
        if content.endswith('\n'): content = content[:-1]

        end_of_block = match.end()
        remaining_text_after_block = text_to_parse[end_of_block:].strip()
        if remaining_text_after_block:
            extra_text_warning = f"[System Warning: Generator AI included extra text after the required code block, which was ignored. Extra text: '{remaining_text_after_block[:100]}...']"
            logger.warning(extra_text_warning)
            self.status_message_ready.emit(extra_text_warning)

        return filepath, content

    def get_last_emitted_filename(self) -> Optional[str]:
        return self._last_emitted_filename