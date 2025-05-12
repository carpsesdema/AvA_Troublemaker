# core/modification_handler.py
import logging
import re
import ast  # For literal_eval (though likely not needed here anymore for plan parsing)
from typing import List, Tuple, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class ModificationHandler(QObject):
    """
    Handles CodeLlama-specific instruction preparation and output parsing
    for the multi-file code modification workflow.
    Works in conjunction with ModificationCoordinator.
    """
    code_file_ready = pyqtSignal(str, str)  # Emits (filename, content)
    status_message_ready = pyqtSignal(str)  # Emits informational message
    modification_parsing_error = pyqtSignal(str)  # Emits on LLM code generation response parsing failure

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._last_emitted_filename: Optional[str] = None
        self._is_active: bool = False
        logger.info("ModificationHandler initialized (for CodeLlama interaction).")

    def activate_sequence(self):
        logger.info("ModificationHandler activated for a sequence.")
        self.cancel_modification()
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
            full_plan: List[str]
    ) -> str:
        if not self._is_active:
            logger.warning("MH: prepare_standard_instruction called when not active.")
            return "[ERROR: Handler not active]"
        logger.debug(f"MH: Preparing standard CodeLlama instruction for: {target_filename}")
        planned_files_str = ", ".join(f"'{f}'" for f in full_plan)
        if not planned_files_str: planned_files_str = "[No other files in plan or plan not specified]"
        instruction = (
            f"You are an expert Python coding assistant. Your current task is to generate the complete and correct code for the file: `{target_filename}`.\n"
            f"This is part of a larger modification based on the user's request: \"{original_user_query}\"\n"
            f"The overall plan involves changes to the following file(s): {planned_files_str}.\n\n"
            f"Please provide the ENTIRE, UPDATED source code for `{target_filename}`. Ensure your output is ONLY the code itself.\n"
            f"**CRITICAL OUTPUT FORMAT:**\n"
            f"Your response MUST contain ONLY a single standard Markdown fenced code block. This block MUST be labeled with the exact filename `{target_filename}` immediately after the opening ``` backticks.\n"
            f"Example:\n"
            f"```python {target_filename}\n"
            f"# Full updated code for {target_filename} goes here...\n"
            f"# ... all necessary imports, classes, functions, etc. ...\n"
            f"print('Hello from {target_filename}')\n"
            f"```\n"
            f"DO NOT include ANY other text, explanations, summaries, greetings, apologies, acknowledgements, or any other conversational fluff before or after the code block. "
            f"ONLY the code block for `{target_filename}`."
        )
        return instruction

    def prepare_codellama_refinement_instruction(
            self,
            target_filename: str,
            user_feedback: str,
            previous_llm_instruction: str
    ) -> str:
        if not self._is_active:
            logger.warning("MH: prepare_refinement_instruction called when not active.")
            return "[ERROR: Handler not active]"
        logger.debug(f"MH: Preparing CodeLlama refinement instruction for: {target_filename}")
        instruction = (
            f"You are an expert Python coding assistant. We are refining the file: `{target_filename}`.\n"
            f"The user has provided the following feedback or additional request: \"{user_feedback}\"\n\n"
            f"The previous instruction given to generate the code for `{target_filename}` was:\n"
            f"--- PREVIOUS INSTRUCTION START ---\n{previous_llm_instruction}\n--- PREVIOUS INSTRUCTION END ---\n\n"
            f"Please provide the NEW, COMPLETE, and CORRECTED source code for `{target_filename}`, incorporating the user's feedback and ensuring all necessary functionality.\n"
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
        response_text_stripped = response_text  # Keep original structure for parsing

        # Call the modified parser
        parsed_file_tuple = self._parse_first_code_block_lenient(response_text_stripped, expected_filename)

        if not parsed_file_tuple:
            err_msg = f"[System: Generator AI response format error. Expected a single Markdown code block labeled with filename '{expected_filename}'. Response did not contain a recognizable code block or the label was incorrect. Response preview:\n{response_text_stripped[:300]}...]"
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
        """
        Leniently parses text for the first Markdown fenced code block that is
        explicitly labeled with the expected filename or a generic Python label.
        It will skip any leading text.
        """
        # Try to find the specifically labeled block first
        # Regex: ``` optionally language, then optional whitespace, then the EXACT expected_filename, then newline, content, newline, ```
        # Escape the expected_filename for regex since it can contain dots, slashes etc.
        escaped_expected_filename = re.escape(expected_filename)
        specific_pattern = rf"```(?:[a-zA-Z0-9_\-\.]*)?\s*{escaped_expected_filename}\s*\n(.*?)\n```"

        logger.debug(f"MH_Lenient: Trying specific pattern: {specific_pattern}")
        match = re.search(specific_pattern, text_to_parse, re.DOTALL | re.IGNORECASE)

        if match:
            filepath = expected_filename  # We matched it directly
            content = match.group(1)
            logger.debug(f"MH_Lenient: Matched specific labeled block for '{filepath}'.")
        else:
            # Fallback: Look for any Python code block and hope it's the right one
            # This is less precise but handles cases where the LLM might omit or alter the label slightly
            logger.debug(f"MH_Lenient: Specific pattern failed. Trying generic python block.")
            generic_pattern = r"```python\s*\n(.*?)\n```"  # Simpler, just looks for ```python
            match = re.search(generic_pattern, text_to_parse, re.DOTALL | re.IGNORECASE)
            if match:
                filepath = expected_filename  # Assume it's for the expected file if a generic python block is found
                content = match.group(1)
                logger.warning(
                    f"MH_Lenient: Matched generic 'python' block for '{expected_filename}'. Assuming content is correct.")
            else:
                # Fallback 2: Look for any code block at all if even generic python fails
                logger.debug(f"MH_Lenient: Generic python block failed. Trying any code block.")
                any_code_block_pattern = r"```(?:[a-zA-Z0-9_\-\.+]+)?\s*\n(.*?)\n```"
                match = re.search(any_code_block_pattern, text_to_parse, re.DOTALL | re.IGNORECASE)
                if match:
                    filepath = expected_filename  # Still assume it's for the expected file
                    content = match.group(1)
                    logger.warning(
                        f"MH_Lenient: Matched ANY code block for '{expected_filename}'. Assuming content is correct.")
                else:
                    logger.warning("MH_Lenient: No code block found even with lenient parsing.")
                    return None

        # Clean content (remove single leading/trailing newline if present)
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]

        # Check for extra text after the block (informational)
        end_of_block = match.end()
        remaining_text_after_block = text_to_parse[end_of_block:].strip()
        if remaining_text_after_block:
            extra_text_warning = f"[System Warning: Generator AI included extra text after the required code block, which was ignored. Extra text: '{remaining_text_after_block[:100]}...']"
            logger.warning(extra_text_warning)
            self.status_message_ready.emit(extra_text_warning)

        return filepath, content

    def get_last_emitted_filename(self) -> Optional[str]:
        return self._last_emitted_filename