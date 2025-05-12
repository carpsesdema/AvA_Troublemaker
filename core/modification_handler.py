# core/modification_handler.py
import logging
import re
import ast # For literal_eval (though likely not needed here anymore for plan parsing)
from typing import List, Tuple, Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# NOTE: The MODIFICATION_PROMPT_TEMPLATE (for initial planning) is now GONE from here.
# ModificationCoordinator handles the initial planning prompt to Gemini.

class ModificationHandler(QObject):
    """
    Handles CodeLlama-specific instruction preparation and output parsing
    for the multi-file code modification workflow.
    Works in conjunction with ModificationCoordinator.
    """
    # Signals to ModificationCoordinator
    code_file_ready = pyqtSignal(str, str) # Emits (filename, content)
    status_message_ready = pyqtSignal(str) # Emits informational message
    modification_parsing_error = pyqtSignal(str) # Emits on LLM code generation response parsing failure

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._last_emitted_filename: Optional[str] = None
        self._is_active: bool = False # Is the handler part of an active sequence?
        logger.info("ModificationHandler initialized (for CodeLlama interaction).")

    def activate_sequence(self):
        """Activates the handler for a new modification sequence."""
        logger.info("ModificationHandler activated for a sequence.")
        self.cancel_modification() # Ensure clean state
        self._is_active = True

    def cancel_modification(self):
        """Resets the handler's state, typically called by ModificationCoordinator."""
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
        full_plan: List[str] # List of all files in the modification plan
    ) -> str:
        """
        Prepares a standard instruction for CodeLlama to generate/update a single file.
        """
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
        previous_llm_instruction: str # The instruction that led to the code being refined
    ) -> str:
        """
        Prepares an instruction for CodeLlama to refine an existing file based on feedback.
        """
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
        """
        Processes the LLM's response which should contain generated code for a file.
        Emits code_file_ready on success, or modification_parsing_error on failure.
        Returns True if parsing and emission were successful, False otherwise.
        """
        if not self._is_active:
            logger.warning(f"MH: process_llm_code_generation_response called for '{expected_filename}' when not active.")
            # self.modification_parsing_error.emit("[System Error: Handler was not active during code processing.]") # Optional
            return False

        logger.info(f"MH: Processing CodeLlama response, expecting file: '{expected_filename}'")
        response_text_stripped = response_text.strip()

        parsed_file_tuple = self._parse_first_code_block(response_text_stripped)

        if not parsed_file_tuple:
            err_msg = f"[System: Generator AI response format error. Expected a single Markdown code block labeled with filename '{expected_filename}'. Response did not contain a recognizable code block. Response preview:\n{response_text_stripped[:300]}...]"
            logger.error(err_msg)
            self.modification_parsing_error.emit(err_msg)
            return False

        parsed_filename, content = parsed_file_tuple

        # Validate filename (optional: be strict or lenient)
        if parsed_filename != expected_filename:
            warn_msg = f"[System Warning: Generator AI provided code for file '{parsed_filename}', but expected '{expected_filename}'. Using AI's provided filename for this step.]"
            logger.warning(warn_msg)
            self.status_message_ready.emit(warn_msg)
            # Decide if this is an error or if we proceed with parsed_filename
            # For now, let's proceed but the ModificationCoordinator might also check this.
            # self._last_emitted_filename = parsed_filename # Update to what was actually parsed
        # else:
        # self._last_emitted_filename = expected_filename

        self._last_emitted_filename = parsed_filename # Always store the filename that was actually parsed from the block

        self.code_file_ready.emit(parsed_filename, content)
        self.status_message_ready.emit(f"[System: Code for '{parsed_filename}' received from Generator AI. Review and provide feedback or type 'next'.]")
        logger.info(f"MH: Successfully parsed and emitted code for '{parsed_filename}'.")
        return True


    def _parse_first_code_block(self, text_to_parse: str) -> Optional[Tuple[str, str]]:
        """
        Parses text for the *first* fenced code block labeled with a filename.
        This is a critical parser for the expected LLM output format.
        """
        # Regex: ``` optionally followed by language, then whitespace, then the path, then newline, content, newline, ```
        # Path can contain letters, numbers, /, ., _, -
        # Content is captured non-greedily until the final ```
        # Made language part optional and more flexible: (?:[a-zA-Z0-9_\-\.]*)?
        # Made whitespace after language optional: \s*
        # Filename capture group made more robust: ([\w\./\\_-]+)
        pattern = r"```(?:[a-zA-Z0-9_\-\.]*)?\s*([\w\./\\_-]+)\s*\n(.*?)\n```"
        # Example match: ```python src/my_app/utils.py\n# code\n```
        # Example match: ``` src/my_app/utils.py\n# code\n```
        # Example match: ```utils.py\n# code\n```

        try:
            match = re.search(pattern, text_to_parse, re.DOTALL | re.IGNORECASE)
            if match:
                filepath = match.group(1).strip().replace("\\", "/") # Normalize path separators
                content = match.group(2) # Get content, DO NOT STRIP YET - allow leading/trailing newlines within block

                # Remove a single leading newline IF PRESENT (often added by LLMs)
                if content.startswith('\n'):
                    content = content[1:]
                # Remove a single trailing newline IF PRESENT
                if content.endswith('\n'):
                    content = content[:-1]

                if filepath: # Content can be empty (e.g. creating an empty __init__.py)
                    logger.debug(f"MH: Parsed code block. File: '{filepath}', Content Length: {len(content)}")
                    # Check for extra text after the block (strict adherence)
                    end_of_block = match.end()
                    remaining_text_after_block = text_to_parse[end_of_block:].strip()
                    if remaining_text_after_block:
                         # This is a format violation by the LLM
                         extra_text_warning = f"[System Warning: Generator AI included extra text after the required code block, which was ignored. This may indicate an issue with the AI's response adherence. Extra text: '{remaining_text_after_block[:100]}...']"
                         logger.warning(extra_text_warning)
                         self.status_message_ready.emit(extra_text_warning) # Inform user via MC
                    return filepath, content
                else:
                    logger.warning(f"MH: Found code block, but filename ('{filepath}') was empty.")
            else:
                logger.warning("MH: No valid labeled code block found in text section using regex.")
        except Exception as e:
            logger.error(f"MH: Error parsing first code block with regex: {e}", exc_info=True)
        return None


    def get_last_emitted_filename(self) -> Optional[str]:
        """Returns the filename that was last emitted via code_file_ready."""
        return self._last_emitted_filename