# services/code_analysis_service.py
import ast
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CodeAnalysisService:
    """
    Provides services for analyzing source code structure, initially focusing on AST parsing.
    """

    def __init__(self):
        logger.info("CodeAnalysisService initialized.")
        # No complex state needed for now

    def parse_python_structures(self, code_content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses Python code using AST to extract function and class definitions.

        Args:
            code_content: The Python code as a string.
            file_path: The path to the file (used for logging).

        Returns:
            A list of dictionaries, each representing a function or class structure
            with its name, type, start line, and end line. Returns empty list on error.
            Example: [{"name": "my_func", "type": "function", "start_line": 10, "end_line": 25}, ...]
        """
        structures = []
        if not code_content:
            return structures

        try:
            logger.debug(f"Attempting AST parse for: {os.path.basename(file_path)}")
            # Ensure Python 3.8+ for end_lineno. Add type_comments=False if needed for compatibility.
            tree = ast.parse(code_content, filename=file_path)
            logger.debug(f"AST parsing successful for: {os.path.basename(file_path)}")

            class StructureVisitor(ast.NodeVisitor):
                def __init__(self, file_path_for_log: str):
                    self.structures = []
                    self.file_path_for_log = file_path_for_log
                    super().__init__()

                def _get_end_line(self, node: ast.AST) -> Optional[int]:
                    """Safely get the end line number (requires Python 3.8+)."""
                    if hasattr(node, 'end_lineno'):
                        return node.end_lineno
                    else:
                        # Basic fallback if end_lineno not available
                        logger.debug(f"Node '{getattr(node, 'name', 'Unnamed')}' in {os.path.basename(self.file_path_for_log)} lacks 'end_lineno'. Estimating.")
                        # A more complex estimation could walk child nodes max line, but keep simple for now.
                        return node.lineno # Simplest fallback: ends on start line

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    end_line = self._get_end_line(node)
                    if end_line is not None:
                        self.structures.append({
                            "name": node.name,
                            "type": "function",
                            "start_line": node.lineno,
                            "end_line": end_line
                        })
                    else:
                         logger.warning(f"Could not determine end line for function '{node.name}' in {os.path.basename(self.file_path_for_log)}.")
                    self.generic_visit(node) # Continue visiting children

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    # Treat async functions the same as regular functions
                    end_line = self._get_end_line(node)
                    if end_line is not None:
                        self.structures.append({
                            "name": node.name,
                            "type": "function", # Keep type consistent as 'function'
                            "start_line": node.lineno,
                            "end_line": end_line
                        })
                    else:
                         logger.warning(f"Could not determine end line for async function '{node.name}' in {os.path.basename(self.file_path_for_log)}.")
                    self.generic_visit(node) # Continue visiting children

                def visit_ClassDef(self, node: ast.ClassDef):
                    end_line = self._get_end_line(node)
                    if end_line is not None:
                        self.structures.append({
                            "name": node.name,
                            "type": "class",
                            "start_line": node.lineno,
                            "end_line": end_line
                        })
                    else:
                         logger.warning(f"Could not determine end line for class '{node.name}' in {os.path.basename(self.file_path_for_log)}.")
                    self.generic_visit(node) # Continue visiting children

            # Instantiate and run the visitor
            visitor = StructureVisitor(file_path)
            visitor.visit(tree)
            structures = visitor.structures
            logger.info(f"Extracted {len(structures)} functions/classes from: {os.path.basename(file_path)}")

        except SyntaxError as e:
            logger.warning(f"AST SyntaxError parsing {os.path.basename(file_path)}: {e}. Skipping structure extraction for this file.")
            # Return empty list, don't stop processing other files
        except Exception as e:
            # Catch other potential AST errors (e.g., recursion depth)
            logger.error(f"Unexpected AST error parsing {os.path.basename(file_path)}: {e}", exc_info=True)
            # Return empty list

        return structures

    # --- Future methods for code analysis could go here ---
    # def analyze_dependencies(self, code_content: str) -> List[str]:
    #     pass