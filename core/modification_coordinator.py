import logging
import ast
import re
import os
import asyncio
import uuid

from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

from .modification_handler import ModificationHandler
from .backend_coordinator import BackendCoordinator
from .project_context_manager import ProjectContextManager
from .rag_handler import RagHandler
from .models import ChatMessage, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
from utils import constants

logger = logging.getLogger(__name__)

PLANNER_BACKEND_ID = getattr(constants, "PLANNER_BACKEND_ID", "gemini_planner")
GENERATOR_BACKEND_ID = getattr(constants, "GENERATOR_BACKEND_ID", "ollama_generator")
RAG_SNIPPET_COUNT_FOR_CODER = 2
RAG_MAX_SNIPPET_LENGTH = 500

MAX_CONCURRENT_GENERATORS = 3


class ModPhase:
    IDLE = "IDLE"
    AWAITING_PLAN_AND_ALL_CODER_INSTRUCTIONS = "AWAITING_PLAN_AND_ALL_CODER_INSTRUCTIONS"
    GENERATING_CODE_CONCURRENTLY = "GENERATING_CODE_CONCURRENTLY"
    ALL_FILES_GENERATED_AWAITING_USER_ACTION = "ALL_FILES_GENERATED_AWAITING_USER_ACTION"


class ModificationCoordinator(QObject):
    request_llm_call = pyqtSignal(str, list)
    file_ready_for_display = pyqtSignal(str, str)
    modification_sequence_complete = pyqtSignal(str, str)
    modification_error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    MAX_LINES_BEFORE_SPLIT = 400

    def __init__(self,
                 modification_handler: ModificationHandler,
                 backend_coordinator: BackendCoordinator,
                 project_context_manager: ProjectContextManager,
                 rag_handler: RagHandler,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not all([modification_handler, backend_coordinator, project_context_manager, rag_handler]):
            logger.critical("MC Init: Missing critical dependencies (incl. RagHandler).")
            raise ValueError(
                "ModificationCoordinator requires valid handler, backend_coord, project_manager, and rag_handler.")

        self._handler = modification_handler
        self._backend_coordinator = backend_coordinator
        self._project_context_manager = project_context_manager
        self._rag_handler = rag_handler

        self._is_active: bool = False
        self._is_awaiting_llm: bool = False
        self._current_phase: str = ModPhase.IDLE
        self._original_query: Optional[str] = None
        self._original_query_at_start: Optional[str] = None
        self._original_context_from_rag: Optional[str] = None
        self._original_focus_prefix: Optional[str] = None

        self._full_planner_output_text: Optional[str] = None
        self._coder_instructions_map: Dict[str, str] = {}
        self._generated_file_data: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        self._planned_files_list: List[str] = []

        self._active_code_generation_tasks: Dict[str, asyncio.Task] = {}

        self._connect_handler_signals()
        logger.info("ModificationCoordinator initialized with RagHandler.")

    def _connect_handler_signals(self):
        if self._handler:
            try:
                self._handler.modification_parsing_error.connect(self._handle_mh_parsing_error)
            except Exception as e:
                logger.exception(f"Error connecting ModificationHandler signals in MC: {e}")

    def _reset_state(self):
        self._is_active = False
        self._is_awaiting_llm = False
        self._current_phase = ModPhase.IDLE
        self._original_query = None
        self._original_query_at_start = None
        self._original_context_from_rag = None
        self._original_focus_prefix = None

        self._full_planner_output_text = None
        self._coder_instructions_map = {}
        self._generated_file_data = {}
        self._planned_files_list = []

        for task_key, task in list(self._active_code_generation_tasks.items()):
            if task and not task.done():
                task.cancel()
        self._active_code_generation_tasks = {}

        if self._handler:
            self._handler.cancel_modification()

    def is_active(self) -> bool:
        return self._is_active

    def is_awaiting_llm_response(self) -> bool:
        return self._is_active and self._is_awaiting_llm

    def start_sequence(self, query: str, context_from_rag: str, focus_prefix: str):
        if self._is_active:
            self._reset_state()

        self._is_active = True
        self._original_query = query
        self._original_query_at_start = query
        self._original_context_from_rag = context_from_rag
        self._original_focus_prefix = focus_prefix
        self._is_awaiting_llm = False

        self._request_initial_plan_and_all_coder_instructions()

    def _request_initial_plan_and_all_coder_instructions(self):
        prompt_text_parts = [
            "You are an expert AI system planner and lead developer. Your task is to plan a multi-file application "
            "based on the user's request and provide **highly detailed, self-contained instructions for a specialized "
            "Coder AI for EACH file** that needs to be created or modified.\n"
            "For each file, also consider if providing 1-2 brief, relevant code examples from the RAG context would significantly help the Coder AI produce higher quality code (e.g., for complex logic, specific class structures, or idiomatic patterns). If so, include a placeholder like `[RAG_EXAMPLES_REQUESTED_FOR_THIS_FILE]` within your natural language plan for that file. The orchestrating system will attempt to fetch these examples.\n\n"
        ]

        if self._current_phase == ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION and self._original_query != self._original_query_at_start:
            prompt_text_parts.append(
                f"This is a REFINEMENT of a previous overall plan based on user feedback after all files were generated. "
                f"The original goal was: \"{self._original_query_at_start}\".\n"
                f"The user's latest feedback on the generated files (potentially affecting multiple files) is: \"{self._original_query}\".\n"
                f"Re-evaluate the entire plan and all Coder AI instructions. Identify ALL files that need to be changed (added, modified, or if a previously modified file now needs no changes or different changes) based on this feedback.\n\n"
            )
        else:
            prompt_text_parts.append(
                f"Implement the following user request:\nUSER REQUEST: \"{self._original_query_at_start}\"\n\n")

        prompt_text_parts.extend([
            f"ASSOCIATED PROJECT CONTEXT (from RAG, if any):\n{self._original_context_from_rag or 'N/A'}\n\n",
            f"PROJECT ROOT FOCUS (base path for relative file paths, if provided by user):\n{self._original_focus_prefix or 'N/A'}\n\n",

            "Your response MUST include THE FOLLOWING TWO PARTS, clearly demarcated:\n\n",

            "PART 1: FILES_TO_MODIFY\n"
            "This line MUST start EXACTLY with 'FILES_TO_MODIFY: ' followed by a Python-style list of relative file paths "
            "(e.g., FILES_TO_MODIFY: ['src/main.py', 'app/utils/helpers.py', 'new_module/service.py']). "
            "Use forward slashes for paths. If no files need changes, use FILES_TO_MODIFY: []. "
            "This line should appear first or very early in your response.\n\n",

            "PART 2: PER_FILE_CODER_INSTRUCTIONS\n"
            "For EACH file listed in FILES_TO_MODIFY, you MUST provide a dedicated section. Each section must contain:\n"
            "   a. A clear START marker: `--- CODER_INSTRUCTIONS_START: path/to/filename.ext ---` (replace with the actual relative path from the FILES_TO_MODIFY list).\n"
            "   b. A DETAILED NATURAL LANGUAGE PLAN for that specific file. This plan should include:\n"
            "      - The file's primary purpose and responsibility within the project.\n"
            "      - Specific classes, functions, methods, and logic to be implemented in THIS FILE.\n"
            "      - Key data structures or algorithms to be used in THIS FILE.\n"
            "      - Necessary imports FOR THIS FILE (suggest them clearly, e.g., 'from .models import User' or 'import os').\n"
            "      - State clearly if the file is NEW or an UPDATE to an existing one. (The orchestrating system will provide existing code to the Coder AI if it's an update based on your indication here).\n"
            "      - Explicitly state any interactions with other planned files IF those interactions define requirements for THIS file's content (e.g., 'This file should define a class MyClass that will be imported and used by other_file.py').\n"
            "      - If helpful, include the placeholder `[RAG_EXAMPLES_REQUESTED_FOR_THIS_FILE]` if you believe RAG examples are beneficial for this specific file's generation.\n"
            "   c. A CRITICAL OUTPUT FORMAT REMINDER to the Coder AI for this specific file:\n"
            "      \"IMPORTANT CODER OUTPUT FORMAT: Your response for this file (`path/to/filename.ext`) MUST be ONLY ONE single standard Markdown fenced code block, starting with ```python path/to/filename.ext\\n and ending with ```. ABSOLUTELY NO other text, explanations, or conversational fluff anywhere in your response.\"\n"
            "   d. A clear END marker: `--- CODER_INSTRUCTIONS_END: path/to/filename.ext ---`\n\n",

            "**Overall Production Quality Standards (Apply to ALL generated code by the Coder AI):**\n"
            "All generated Python code must adhere to PEP 8 (max line length 99 characters), include type hints for all function arguments, return values, and critical variables (PEP 484), "
            "have comprehensive docstrings for all modules, classes, functions, and methods (PEP 257), and use inline comments for complex or non-obvious logic. "
            "Ensure code is modular, functional, and correctly implements the specified requirements. Organize imports as per PEP 8.\n\n",

            "Generate the complete response now, including both parts, ensuring all paths use forward slashes."
        ])
        prompt_text = "".join(prompt_text_parts)

        self._full_planner_output_text = None
        self._coder_instructions_map = {}

        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[prompt_text])]
        self._is_awaiting_llm = True
        self._current_phase = ModPhase.AWAITING_PLAN_AND_ALL_CODER_INSTRUCTIONS
        self.status_update.emit("[System: Asking Planner AI to create modification plan and coder instructions...]")
        self.request_llm_call.emit(PLANNER_BACKEND_ID, history_for_llm)

    def _extract_text_between_markers(self, text: str, start_marker: str, end_marker: str) -> Optional[str]:
        try:
            start_idx_find = text.find(start_marker)
            if start_idx_find == -1:
                raise ValueError(f"Start marker '{start_marker}' not found (exact).")

            start_idx = start_idx_find + len(start_marker)
            end_idx_find = text.find(end_marker, start_idx)
            if end_idx_find == -1:
                raise ValueError(f"End marker '{end_marker}' not found after start marker '{start_marker}'.")

            return text[start_idx:end_idx_find].strip()
        except ValueError as ve:
            return None
        except Exception as e:
            return None

    def _handle_plan_response(self, planner_response_text: str):
        self._is_awaiting_llm = False
        self._full_planner_output_text = planner_response_text.strip()

        parsed_files_list, error_msg_parse_files = self._parse_files_to_modify_list(self._full_planner_output_text)
        if error_msg_parse_files or parsed_files_list is None:
            err_msg_ui = f"Failed to parse FILES_TO_MODIFY list from Planner AI: {error_msg_parse_files}. Response preview: '{planner_response_text[:300] if planner_response_text else '[EMPTY RESPONSE]'}...'"
            self.modification_error.emit(err_msg_ui)
            self._handle_sequence_end("error_plan_parsing", err_msg_ui)
            return

        self._planned_files_list = parsed_files_list
        if not self._planned_files_list:
            self.status_update.emit("[System: Planner AI indicates no file modifications are needed.]")
            self._handle_sequence_end("completed_no_files_in_plan", "Planner found no files to modify.")
            return

        files_str_display = ", ".join([f"`{f}`" for f in self._planned_files_list])
        self.status_update.emit(f"[System: Planner AI will process: {files_str_display}]")

        self._coder_instructions_map = {}
        all_instructions_extracted = True
        current_search_pos = 0

        for filename_in_list in self._planned_files_list:
            normalized_filename_for_marker = filename_in_list.replace("\\", "/")
            start_marker = f"--- CODER_INSTRUCTIONS_START: {normalized_filename_for_marker} ---"
            end_marker = f"--- CODER_INSTRUCTIONS_END: {normalized_filename_for_marker} ---"
            instruction_text = None
            try:
                start_idx_find = self._full_planner_output_text.find(start_marker, current_search_pos)
                if start_idx_find == -1:
                    pass
                else:
                    start_idx_content = start_idx_find + len(start_marker)
                    end_idx_find = self._full_planner_output_text.find(end_marker, start_idx_content)
                    if end_idx_find == -1:
                        pass
                    else:
                        instruction_text = self._full_planner_output_text[start_idx_content:end_idx_find].strip()
                        current_search_pos = end_idx_find + len(end_marker)
            except Exception as e_extract:
                pass

            if instruction_text:
                self._coder_instructions_map[filename_in_list] = instruction_text
            else:
                self._coder_instructions_map[
                    filename_in_list] = f"[Error: Planner failed to provide Coder AI instructions for {filename_in_list}]"
                all_instructions_extracted = False

        if not all_instructions_extracted or len(self._coder_instructions_map) != len(self._planned_files_list):
            if all(val.startswith("[Error:") for val in self._coder_instructions_map.values()):
                self.modification_error.emit(
                    "Planner AI failed to provide valid instructions for ANY of the planned files. Cannot proceed.")
                self._handle_sequence_end("error_no_valid_coder_instructions",
                                          "No valid Coder AI instructions received.")
                return
            else:
                self.modification_error.emit(
                    "Planner AI failed to provide instructions for some planned files. Those files may be skipped or incomplete.")

        self._generated_file_data = {}
        if not any(val and not val.startswith("[Error:") for val in self._coder_instructions_map.values()):
            self.modification_error.emit("Planner AI did not provide any valid Coder AI instructions for any file.")
            self._handle_sequence_end("error_no_valid_coder_instructions_after_check",
                                      "No valid Coder AI instructions for any file.")
            return

        self._current_phase = ModPhase.GENERATING_CODE_CONCURRENTLY
        QTimer.singleShot(0, lambda: asyncio.create_task(self._process_all_files_concurrently()))

    async def _get_rag_snippets_for_coder(self, filename: str, coder_instruction_for_file: str) -> str:
        if "[RAG_EXAMPLES_REQUESTED_FOR_THIS_FILE]" not in coder_instruction_for_file:
            return ""

        if not self._rag_handler:
            return ""

        active_project_id = self._project_context_manager.get_active_project_id() if self._project_context_manager else None
        if not active_project_id:
            return ""

        query_for_rag = f"Relevant code examples for implementing: {filename}. Key aspects: {coder_instruction_for_file[:200]}"

        try:
            focus_paths_for_rag = [
                os.path.join(self._original_focus_prefix, filename)] if self._original_focus_prefix else [filename]

            context_str, _ = self._rag_handler.get_formatted_context(
                query=query_for_rag,
                query_entities=self._rag_handler.extract_code_entities(coder_instruction_for_file),
                project_id=active_project_id,
                focus_paths=focus_paths_for_rag,
                is_modification_request=True
            )

            if not context_str or context_str.startswith("[Error retrieving RAG context]"):
                return ""

            snippets = []

            # A more robust way to extract ```python ... ``` blocks
            code_block_pattern = re.compile(r"--- Snippet \d+ from `(.*?)` .*?---\s*```python\s*(.*?)\s*```", re.DOTALL)
            matches = code_block_pattern.finditer(context_str)

            for match_idx, match in enumerate(matches):
                if match_idx >= RAG_SNIPPET_COUNT_FOR_CODER:
                    break
                original_path = match.group(1)
                code_content = match.group(2).strip()

                if len(code_content) > RAG_MAX_SNIPPET_LENGTH:
                    code_content = code_content[:RAG_MAX_SNIPPET_LENGTH] + "\n# ... (snippet truncated) ..."

                snippets.append(f"Example from `{original_path}`:\n```python\n{code_content}\n```\n")

            if snippets:
                return "\n--- RELEVANT CODE EXAMPLES FROM KNOWLEDGE BASE ---\n" + "\n".join(
                    snippets) + "--- END OF RELEVANT CODE EXAMPLES ---\n\n"
            return ""
        except Exception as e:
            return ""

    async def _execute_single_code_generation_task(self, filename: str) -> Tuple[str, Optional[str], Optional[str]]:
        coder_instruction_for_file = self._coder_instructions_map.get(filename)
        if not coder_instruction_for_file or coder_instruction_for_file.startswith("[Error:"):
            return filename, None, coder_instruction_for_file or "Missing Coder AI instructions."

        rag_examples_str = await self._get_rag_snippets_for_coder(filename, coder_instruction_for_file)

        coder_instruction_for_file = coder_instruction_for_file.replace("[RAG_EXAMPLES_REQUESTED_FOR_THIS_FILE]",
                                                                        "").strip()

        original_file_content = self._read_original_file_content(filename)

        final_coder_prompt_parts = [
            f"You are an expert Python Coder AI. Your task is to generate or update the file `{filename}` based "
            f"on the following detailed instructions. Pay EXTREMELY close attention to all requirements, "
            f"especially the **CRITICAL OUTPUT FORMAT REMINDER** contained within the instructions for this file.\n\n"
        ]

        if rag_examples_str:
            final_coder_prompt_parts.append(rag_examples_str)

        is_new_file_from_planner_instr = "new file" in coder_instruction_for_file.lower() or \
                                         "is new" in coder_instruction_for_file.lower() or \
                                         "create it entirely from scratch" in coder_instruction_for_file.lower()

        if original_file_content is not None:
            if is_new_file_from_planner_instr:
                pass
            final_coder_prompt_parts.append(
                f"The file `{filename}` **EXISTS**. Its current content is:\n"
                f"```python_original_for_coder\n{original_file_content}\n```\n\n"
                f"You MUST use this original content as the foundation and apply the necessary modifications "
                f"as detailed in the instructions below.\n\n"
            )
        elif not is_new_file_from_planner_instr:
            final_coder_prompt_parts.append(
                f"The file `{filename}` was planned (possibly as an update), but no original content was found. "
                f"Therefore, treat this as a NEW file creation, following the instructions below.\n\n"
            )
        else:
            final_coder_prompt_parts.append(
                f"The file `{filename}` is NEW. Create it from scratch based on the instructions below.\n\n"
            )

        final_coder_prompt_parts.extend([
            "--- DETAILED INSTRUCTIONS FOR THIS FILE (from Planner AI) ---\n",
            coder_instruction_for_file,
            "\n--- END OF DETAILED INSTRUCTIONS ---\n\n",
            f"Proceed to generate the complete, production-quality Python code for `{filename}` now, strictly adhering to all instructions and the output format reminder."
        ])

        final_coder_instruction = "".join(final_coder_prompt_parts)
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[final_coder_instruction])]
        coder_options = {"temperature": 0.2}

        request_id = f"mc_coder_{filename.replace('/', '_').replace('.', '_')}_{uuid.uuid4().hex[:8]}"
        request_metadata = {"purpose": f"mc_request_code_generation_{filename}", "mc_internal_id": request_id,
                            "backend_id_for_mc": GENERATOR_BACKEND_ID}

        response_future = asyncio.Future()
        temp_on_response_slot = None
        temp_on_error_slot = None

        def on_response(resp_request_id, completed_message, usage_stats):
            nonlocal temp_on_response_slot, temp_on_error_slot
            if resp_request_id == request_id:
                if not response_future.done():
                    response_future.set_result(completed_message.text.strip())
                try:
                    if temp_on_response_slot: self._backend_coordinator.response_completed.disconnect(
                        temp_on_response_slot)
                    if temp_on_error_slot: self._backend_coordinator.response_error.disconnect(temp_on_error_slot)
                except TypeError:
                    pass
                temp_on_response_slot = temp_on_error_slot = None

        def on_error(err_request_id, error_message_str):
            nonlocal temp_on_response_slot, temp_on_error_slot
            if err_request_id == request_id:
                if not response_future.done():
                    response_future.set_exception(RuntimeError(f"Coder AI error for {filename}: {error_message_str}"))
                try:
                    if temp_on_response_slot: self._backend_coordinator.response_completed.disconnect(
                        temp_on_response_slot)
                    if temp_on_error_slot: self._backend_coordinator.response_error.disconnect(temp_on_error_slot)
                except TypeError:
                    pass
                temp_on_response_slot = temp_on_error_slot = None

        temp_on_response_slot = on_response
        temp_on_error_slot = on_error

        try:
            self._backend_coordinator.response_completed.connect(temp_on_response_slot)
            self._backend_coordinator.response_error.connect(temp_on_error_slot)
        except Exception as e_conn:
            return filename, None, f"Internal error setting up Coder AI request: {e_conn}"

        self._backend_coordinator.request_response_stream(
            target_backend_id=GENERATOR_BACKEND_ID,
            request_id=request_id,
            history_to_send=history_for_llm,
            is_modification_response_expected=True,
            options=coder_options,
            request_metadata=request_metadata
        )
        self.status_update.emit(f"[System: Coder AI processing `{filename}`...]")

        try:
            timeout_seconds = 900.0
            generated_code_text = await asyncio.wait_for(response_future, timeout=timeout_seconds)
            return filename, generated_code_text, None
        except asyncio.TimeoutError:
            if not response_future.done(): response_future.cancel()
            return filename, None, "Coder AI request timed out."
        except RuntimeError as e:
            return filename, None, str(e)
        except asyncio.CancelledError:
            return filename, None, "Coder AI task cancelled."
        except Exception as e_task:
            return filename, None, f"Unexpected error: {e_task}"
        finally:
            try:
                if temp_on_response_slot: self._backend_coordinator.response_completed.disconnect(temp_on_response_slot)
                if temp_on_error_slot: self._backend_coordinator.response_error.disconnect(temp_on_error_slot)
            except TypeError:
                pass
    async def _process_all_files_concurrently(self):
        self.status_update.emit(
            f"[System: Coder AI is now generating code for {len(self._planned_files_list)} files concurrently...]")

        tasks_to_run = []
        for filename_to_process in self._planned_files_list:
            if filename_to_process in self._coder_instructions_map and \
               not self._coder_instructions_map[filename_to_process].startswith("[Error:"):
                task = asyncio.create_task(self._execute_single_code_generation_task(filename_to_process))
                tasks_to_run.append(task)
                self._active_code_generation_tasks[filename_to_process] = task
            else:
                self._generated_file_data[filename_to_process] = (None, self._coder_instructions_map.get(filename_to_process, "Instructions unavailable."))

        if not tasks_to_run:
            self.status_update.emit("[System: No files could be prepared for code generation due to instruction errors.]")
            if any(self._generated_file_data.values()):
                 self._current_phase = ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION
                 self.status_update.emit(
                    f"[System: All {len(self._planned_files_list)} files processed. Check logs for errors. "
                    "Review in Code Viewer. Provide overall feedback or type 'accept'.]"
                 )
            else:
                 self._handle_sequence_end("error_no_files_to_process_concurrently", "No files to process.")
            return

        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        self._active_code_generation_tasks.clear()

        files_successfully_generated_count = 0
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, Exception):
                self.status_update.emit(f"[System Error: Exception during code generation: {str(result)[:100]}...]")
                continue
            if not isinstance(result, tuple) or len(result) != 3:
                continue

            filename, generated_content, error_msg = result
            self._generated_file_data[filename] = (generated_content, error_msg)

            if error_msg:
                self.status_update.emit(f"[System Error: Coder AI failed for `{filename}`: {error_msg}]")
            elif generated_content is not None:
                if self._handler.process_llm_code_generation_response(generated_content, filename):
                    parsed_filename_content_tuple = self._handler.get_last_emitted_filename_and_content()
                    if parsed_filename_content_tuple and parsed_filename_content_tuple[0] == filename:
                        actual_filename, actual_content = parsed_filename_content_tuple
                        original_content_for_compare = self._read_original_file_content(actual_filename)
                        is_new_file = original_content_for_compare is None
                        if not is_new_file and actual_content.strip() == (original_content_for_compare or "").strip():
                            self.status_update.emit(f"[System: No effective changes applied to `{actual_filename}`.]")
                        elif is_new_file and not actual_content.strip():
                            self.status_update.emit(f"[System: No content generated for new file `{actual_filename}`.]")
                        else:
                            files_successfully_generated_count += 1
                            lines = actual_content.splitlines()
                            if len(lines) > self.MAX_LINES_BEFORE_SPLIT:
                                split_point = len(lines) // 2
                                part1_content = "\n".join(lines[:split_point])
                                part2_content = "\n".join(lines[split_point:])
                                self.file_ready_for_display.emit(actual_filename + " (Part 1/2)", part1_content)
                                self.file_ready_for_display.emit(actual_filename + " (Part 2/2)", part2_content)
                            else:
                                self.file_ready_for_display.emit(actual_filename, actual_content)
                            self.status_update.emit(f"[System: Code for `{actual_filename}` generated.]")
                            self._generated_file_data[filename] = (actual_content, None)
                    else:
                        mismatch_err = f"Filename mismatch after MH parsing for '{filename}'. Expected '{filename}', got '{parsed_filename_content_tuple[0] if parsed_filename_content_tuple else 'None'}'."
                        self.status_update.emit(f"[System Warning: Output issue for `{filename}`.]")
                        self._generated_file_data[filename] = (generated_content, mismatch_err)
                else:
                    parsing_error_msg = f"Coder AI output format error for `{filename}`."
                    self.status_update.emit(f"[System Error: {parsing_error_msg}]")
                    self._generated_file_data[filename] = (generated_content, parsing_error_msg)
            else:
                no_content_msg = f"Coder AI returned no content for `{filename}`."
                self.status_update.emit(f"[System: {no_content_msg}]")
                self._generated_file_data[filename] = (None, no_content_msg)

        num_errors = sum(1 for _, err in self._generated_file_data.values() if err)
        planned_count = len(self._planned_files_list)
        final_status_msg = ""
        if files_successfully_generated_count == planned_count and num_errors == 0:
            final_status_msg = f"[System: All {planned_count} files generated successfully! Review in Code Viewer. Provide overall feedback or type 'accept'.]"
        elif files_successfully_generated_count > 0:
            final_status_msg = (f"[System: {files_successfully_generated_count}/{planned_count} files generated/updated. "
                                f"{num_errors} file(s) had issues. Review in Code Viewer and logs. Provide overall feedback or type 'accept'.]")
        else:
            final_status_msg = (f"[System: All {planned_count} planned files encountered issues during generation. "
                                f"Check logs. Review in Code Viewer. Provide overall feedback or type 'accept'.]")
        self.status_update.emit(final_status_msg)
        self._current_phase = ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION
        self._is_awaiting_llm = False

    def _read_original_file_content(self, relative_filename: str) -> Optional[str]:
        content: Optional[str] = None
        full_path: Optional[str] = None
        norm_relative_filename = os.path.normpath(relative_filename)

        if self._original_focus_prefix and os.path.isdir(self._original_focus_prefix):
            full_path = os.path.join(self._original_focus_prefix, norm_relative_filename)
        elif os.path.isabs(norm_relative_filename):
            full_path = norm_relative_filename
        elif not self._original_focus_prefix:
             return None
        else:
            return None

        if full_path:
            full_path = os.path.normpath(full_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    pass
            else:
                pass
        return content

    def process_llm_response(self, backend_id: str, response_message: ChatMessage):
        if not self._is_active or not self._is_awaiting_llm:
            return
        response_text = response_message.text.strip()
        try:
            if self._current_phase == ModPhase.AWAITING_PLAN_AND_ALL_CODER_INSTRUCTIONS and backend_id == PLANNER_BACKEND_ID:
                self._handle_plan_response(response_text)
            else:
                self._is_awaiting_llm = False
                err_msg = f"Unexpected LLM response from '{backend_id}' for phase {self._current_phase}."
                self.modification_error.emit(err_msg)
                self._handle_sequence_end("error_unexpected_response_phase", err_msg)
        except Exception as e:
            self._is_awaiting_llm = False
            self._handle_sequence_end("error_processing_llm_response", f"Error processing {backend_id} response: {e}")

    def process_user_input(self, user_command: str):
        if not self._is_active: return
        if self._is_awaiting_llm or self._current_phase == ModPhase.GENERATING_CODE_CONCURRENTLY:
            self.status_update.emit("[System: Please wait for the current AI processing to complete.]")
            return
        command_lower = user_command.lower().strip()
        if command_lower in ["cancel", "stop", "abort"]:
            self._handle_sequence_end("cancelled_by_user", "User cancelled the modification process.")
            return
        if self._current_phase == ModPhase.ALL_FILES_GENERATED_AWAITING_USER_ACTION:
            if command_lower in ["accept", "done", "looks good", "ok", "okay", "proceed", "complete", "finalize"]:
                self._handle_sequence_end("completed_by_user_acceptance", "User accepted all generated files.")
            else:
                self.status_update.emit(f"[System: Received overall feedback: \"{user_command[:50]}...\". Requesting full re-plan...]")
                self._original_query = f"The initial request was: '{self._original_query_at_start}'. Based on the generated files, the user now provides this overall feedback for refinement: '{user_command}'"
                self._is_awaiting_llm = False
                self._planned_files_list = []
                self._coder_instructions_map = {}
                self._generated_file_data = {}
                self._request_initial_plan_and_all_coder_instructions()
        else:
            self.status_update.emit(f"[System: Currently processing. Please wait until all files are generated to provide overall feedback or type 'cancel'.]")

    @pyqtSlot(str)
    def _handle_mh_parsing_error(self, error_message: str):
        if not self._is_active: return
        filename_match = re.search(r"for '([^']*)'", error_message)
        filename_affected = filename_match.group(1) if filename_match else "unknown file"
        self.status_update.emit(f"[System Error: Coder AI output for `{filename_affected}` was not in the expected format. This file may be incomplete or incorrect.]")
        if filename_affected in self._generated_file_data:
            existing_content, _ = self._generated_file_data[filename_affected]
            self._generated_file_data[filename_affected] = (existing_content, error_message)
        else:
            self._generated_file_data[filename_affected] = (None, error_message)

    def _parse_files_to_modify_list(self, response_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
        marker = "FILES_TO_MODIFY:"
        try:
            marker_pos = response_text.index(marker)
            list_str_start = marker_pos + len(marker)
        except ValueError:
            match_marker = re.search(r"FILES_TO_MODIFY\s*:", response_text, re.IGNORECASE)
            if not match_marker:
                return None, f"Marker '{marker}' or similar variant not found in Planner response."
            list_str_start = match_marker.end()

        potential_list_str = response_text[list_str_start:].strip()
        first_line_of_potential_list = potential_list_str.split('\n', 1)[0].strip()
        list_str_for_eval = None
        list_match = re.search(r"(\[.*?\])", first_line_of_potential_list)

        if list_match:
            list_str_for_eval = list_match.group(1)
        elif first_line_of_potential_list.startswith('[') and first_line_of_potential_list.endswith(']'):
            list_str_for_eval = first_line_of_potential_list
        else:
            multiline_list_match = re.search(r"^\s*(\[.*?\])", potential_list_str, re.MULTILINE | re.DOTALL)
            if multiline_list_match:
                list_str_for_eval = multiline_list_match.group(1)
            else:
                return None, "FILES_TO_MODIFY list not found or not correctly formatted with brackets on the first line (or subsequent lines) after the marker."
        try:
            if list_str_for_eval.lower().startswith("python"):
                list_str_for_eval = re.sub(r"^[Pp][Yy][Tt][Hh][Oo][Nn]\s*", "", list_str_for_eval)
            parsed_list = ast.literal_eval(list_str_for_eval)
            if not isinstance(parsed_list, list):
                return None, "Parsed data for FILES_TO_MODIFY is not a list."
            cleaned_list = [str(f).strip().replace("\\", "/") for f in parsed_list if isinstance(f, (str, int, float))]
            return [f_item for f_item in cleaned_list if f_item], None
        except (ValueError, SyntaxError, TypeError) as e:
            return None, f"Error parsing FILES_TO_MODIFY list string '{list_str_for_eval}': {e}"

    def cancel_sequence(self, reason: str = "cancelled_externally"):
        if not self._is_active: return
        self._handle_sequence_end(reason, f"Sequence cancelled: {reason}")

    def _handle_sequence_end(self, reason: str, details: Optional[str] = None):
        if not self._is_active and reason != "error_processing_llm_response":
            return

        log_message = f"MC: Ending sequence. Reason: {reason}."
        if details: log_message += f" Details: {details}"

        for filename, task in list(self._active_code_generation_tasks.items()):
            if task and not task.done():
                task.cancel()
        self._active_code_generation_tasks.clear()

        original_query_summary = self._original_query_at_start[:75] + '...' if self._original_query_at_start and len(
            self._original_query_at_start) > 75 else self._original_query_at_start or "User's request"

        self.modification_sequence_complete.emit(reason, original_query_summary)
        self._reset_state()