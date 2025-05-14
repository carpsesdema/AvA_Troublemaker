# services/project_intelligence_service.py
import logging
import os
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Any

# Assuming VectorDBService is importable for type hinting
# from .vector_db_service import VectorDBService # Or adjust path as needed
# For now, let's use a forward reference string for VectorDBService type hint
# if the actual import causes circular issues or isn't available yet in this context.

logger = logging.getLogger(__name__)

# --- Constants for Scoring & Selection ---
FILENAME_SCORE_BOOSTS: Dict[str, int] = {
    "main.py": 15, "app.py": 12, "manage.py": 10, "wsgi.py": 8, "asgi.py": 8,
    "__init__.py": 7,  # Important for structure but maybe not always "key" content
    "settings.py": 10, "config.py": 10, "urls.py": 9, "routes.py": 9,
    "models.py": 10, "views.py": 10, "controllers.py": 10, "handlers.py": 9,
    "services.py": 9, "utils.py": 7, "helpers.py": 7, "tasks.py": 8,
    "admin.py": 7, "forms.py": 7,
    # Common test files
    "test_*.py": 6, "*_test.py": 6, "tests.py": 6, "conftest.py": 6,
    # Docker/CI
    "dockerfile": 5, "docker-compose.yml": 5, ".gitlab-ci.yml": 5, "requirements.txt": 5,
    "readme.md": 8, "readme.rst": 8,
}

CORE_DIR_KEYWORDS: Set[str] = {
    "src", "app", "core", "lib", "services", "api", "apis", "components",
    "utils", "helpers", "conf", "config", "controllers", "views", "models",
    "routes", "handlers", "tasks", "db", "database", "domain", "application",
    "tests", "test"
}
CORE_DIR_SCORE_BOOST: int = 8
ROOT_FILE_SCORE_BOOST: int = 5  # Slightly higher boost for root files

CHUNK_COUNT_THRESHOLDS_SCORES: List[Tuple[int, int]] = [
    (100, 5), (50, 3), (20, 2), (10, 1)  # (chunk_threshold, score_bonus)
]
ENTITY_COUNT_THRESHOLDS_SCORES: List[Tuple[int, int]] = [
    (50, 5), (25, 3), (10, 2), (5, 1)  # (entity_threshold, score_bonus)
]

DEFAULT_MAX_FILES_TO_DETAIL = 7
DEFAULT_MAX_ENTITIES_PER_FILE = 10


class ProjectIntelligenceService:
    """
    Provides services to analyze and understand project structures
    based on RAG (VectorDB) content.
    """

    def __init__(self, vector_db_service: 'VectorDBService'):  # Forward reference string
        if not vector_db_service:
            logger.critical("ProjectIntelligenceService requires a valid VectorDBService instance.")
            raise ValueError("VectorDBService instance is required.")
        self._vector_db_service = vector_db_service
        logger.info("ProjectIntelligenceService initialized.")

    def get_condensed_rag_overview_for_summarization(
            self,
            project_id: str,
            max_files_to_detail: int = DEFAULT_MAX_FILES_TO_DETAIL,
            max_entities_per_file: int = DEFAULT_MAX_ENTITIES_PER_FILE
    ) -> str:
        """
        Generates a condensed textual overview of the project's RAG content,
        highlighting key files and entities, suitable for an LLM to summarize further.

        Args:
            project_id: The ID of the project/collection to analyze.
            max_files_to_detail: Maximum number of key files to detail.
            max_entities_per_file: Maximum number of key entities to list per detailed file.

        Returns:
            A string containing the condensed overview, or an error/info message.
        """
        logger.info(f"Generating RAG overview for project_id: '{project_id}'")
        if not self._vector_db_service.is_ready(project_id):
            msg = f"Project '{project_id}' is not indexed or RAG system is not ready. Cannot generate overview."
            logger.warning(msg)
            return f"[INFO: {msg}]"

        all_metadata: List[Dict[str, Any]] = self._vector_db_service.get_all_metadata(project_id)
        if not all_metadata:
            msg = f"No RAG metadata found for project '{project_id}'. The project might be empty or not indexed."
            logger.info(msg)
            return f"[INFO: {msg}]"

        # 1. Aggregate File Data
        # file_info_map: {filepath: {"chunk_count": int, "entities": set, "source_path": str, "filename": str, "score": int}}
        file_info_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"chunk_count": 0, "entities": set(), "score": 0}
        )
        all_project_entities: Set[str] = set()
        all_dir_paths: Set[str] = set()

        for chunk_meta in all_metadata:
            source_path = chunk_meta.get("source")
            if not source_path:
                logger.debug(f"Skipping chunk metadata with no source path: {chunk_meta.get('filename', 'N/A')}")
                continue

            # Ensure source_path is stored if not already
            if "source_path" not in file_info_map[source_path]:
                file_info_map[source_path]["source_path"] = source_path
                file_info_map[source_path]["filename"] = os.path.basename(source_path)
                # Record directory path
                dir_path = os.path.dirname(source_path)
                if dir_path and dir_path != ".":  # Exclude current dir representation
                    all_dir_paths.add(dir_path)

            file_info_map[source_path]["chunk_count"] += 1
            chunk_entities = chunk_meta.get("code_entities", [])
            if isinstance(chunk_entities, list):
                valid_entities = {str(e).strip() for e in chunk_entities if isinstance(e, str) and str(e).strip()}
                file_info_map[source_path]["entities"].update(valid_entities)
                all_project_entities.update(valid_entities)
            elif chunk_entities:  # If it's not a list but not empty
                logger.warning(f"Chunk entities for {source_path} is not a list: {type(chunk_entities)}. Skipping.")

        if not file_info_map:
            return "[INFO: No processable files found in the RAG metadata for this project.]"

        # 2. Score Files
        logger.debug(f"Scoring {len(file_info_map)} files...")
        for path, info in file_info_map.items():
            score = 0
            filename_lower = info["filename"].lower()
            filepath_lower = path.lower()

            # Filename convention score
            for pattern, boost in FILENAME_SCORE_BOOSTS.items():
                if "*" in pattern:  # Basic wildcard matching
                    if (pattern.startswith("*") and filename_lower.endswith(pattern[1:])) or \
                            (pattern.endswith("*") and filename_lower.startswith(pattern[:-1])):
                        score += boost
                        logger.debug(f"  File '{filename_lower}' got {boost} for pattern '{pattern}'")
                        break  # Avoid multiple wildcard matches if not intended
                elif filename_lower == pattern:
                    score += boost
                    logger.debug(f"  File '{filename_lower}' got {boost} for exact match '{pattern}'")
                    break  # Found exact match

            # Directory structure score
            path_parts = set(p.lower() for p in filepath_lower.split(os.sep) if p)
            if CORE_DIR_KEYWORDS.intersection(path_parts):
                score += CORE_DIR_SCORE_BOOST
                logger.debug(f"  File '{filename_lower}' got {CORE_DIR_SCORE_BOOST} for core dir.")

            # Root file score (path depth 1 means it's in the root of the indexed structure)
            # Depth 0 if filename is the full path (e.g. single file upload, not in a dir)
            depth = path.count(os.sep)
            if depth <= 1:  # Check if filename is same as path for single file upload
                if os.path.basename(path) == path:  # Handle cases like "myfile.py" as source
                    depth = 0
                if depth <= 1:
                    score += ROOT_FILE_SCORE_BOOST
                    logger.debug(f"  File '{filename_lower}' got {ROOT_FILE_SCORE_BOOST} for root (depth {depth}).")

            # Chunk count score
            for threshold, bonus in CHUNK_COUNT_THRESHOLDS_SCORES:
                if info["chunk_count"] >= threshold:
                    score += bonus
                    logger.debug(
                        f"  File '{filename_lower}' got {bonus} for chunks ({info['chunk_count']} >= {threshold}).")
                    break  # Apply highest applicable bonus

            # Entity count score
            num_entities = len(info["entities"])
            for threshold, bonus in ENTITY_COUNT_THRESHOLDS_SCORES:
                if num_entities >= threshold:
                    score += bonus
                    logger.debug(f"  File '{filename_lower}' got {bonus} for entities ({num_entities} >= {threshold}).")
                    break

            info["score"] = score

        # 3. Select Key Files
        sorted_files = sorted(file_info_map.values(), key=lambda x: x["score"], reverse=True)
        key_files_detailed_info = sorted_files[:max_files_to_detail]
        logger.info(f"Selected {len(key_files_detailed_info)} key files for detailed overview.")

        # 4. Select Key Entities for Selected Files
        detailed_files_output = []
        for file_data in key_files_detailed_info:
            entities = sorted(list(file_data["entities"]))
            classes = sorted([e for e in entities if e and e[0].isupper() and len(e) > 1 and not e.startswith("_")])
            public_funcs = sorted([e for e in entities if e and not e[0].isupper() and not e.startswith("_") and len(
                e) > 1 and e not in classes])
            private_funcs_or_vars = sorted([e for e in entities if e and e.startswith("_") and len(
                e) > 1 and e not in classes and e not in public_funcs])

            selected_entities_list = []
            entity_type_counts = {"classes": 0, "functions": 0}

            # Prioritize classes
            for cls_name in classes:
                if len(selected_entities_list) < max_entities_per_file:
                    selected_entities_list.append(f"{cls_name} (Class)")
                    entity_type_counts["classes"] += 1
                else:
                    break

            # Then public functions
            if len(selected_entities_list) < max_entities_per_file:
                for func_name in public_funcs:
                    if len(selected_entities_list) < max_entities_per_file:
                        selected_entities_list.append(func_name)
                        entity_type_counts["functions"] += 1
                    else:
                        break

            # Then private functions/variables if space
            if len(selected_entities_list) < max_entities_per_file:
                for item_name in private_funcs_or_vars:
                    if len(selected_entities_list) < max_entities_per_file:
                        selected_entities_list.append(item_name)
                        # Not strictly differentiating private funcs vs vars here for simplicity
                    else:
                        break

            file_summary_str = f"  File: {file_data['filename']} (Source: {file_data['source_path']}, Chunks: {file_data['chunk_count']}, Score: {file_data['score']})\n"
            if selected_entities_list:
                file_summary_str += f"    Key Entities ({len(selected_entities_list)}/{len(entities)} shown): " + ", ".join(
                    selected_entities_list)
                if len(entities) > len(selected_entities_list):
                    file_summary_str += f" ... (and {len(entities) - len(selected_entities_list)} more)"
                file_summary_str += "\n"
            else:
                file_summary_str += "    Key Entities: None identified or selected.\n"
            detailed_files_output.append(file_summary_str)

        # 5. Calculate Overall Stats & Format Output
        total_files = len(file_info_map)
        total_chunks = sum(info["chunk_count"] for info in file_info_map.values())
        # Identify key directories (e.g., top N most frequent unique parent directories)
        dir_counts = defaultdict(int)
        for path in file_info_map.keys():
            parent_dir = os.path.dirname(path)
            if parent_dir and parent_dir != ".":  # Exclude representing current dir explicitly
                dir_counts[parent_dir] += 1

        sorted_dirs = sorted(dir_counts.items(), key=lambda item: item[1], reverse=True)
        key_dirs_str = ", ".join([f"{d} ({c} files)" for d, c in sorted_dirs[:5]])  # Top 5 directories
        if not key_dirs_str: key_dirs_str = "N/A (likely flat structure or few files)"

        overview = [
            f"Project RAG Overview for Project ID: {project_id}\n",
            f"Overall Statistics:",
            f"  - Total Indexed Files: {total_files}",
            f"  - Total Indexed Chunks: {total_chunks}",
            f"  - Total Unique Code Entities Identified: {len(all_project_entities)}",
            f"  - Key Directory Paths Observed: {key_dirs_str}\n",
            f"Key File Details (Top {len(key_files_detailed_info)} of {total_files} files based on heuristics):"
        ]
        overview.extend(detailed_files_output)
        if not detailed_files_output:
            overview.append("  No specific files were selected for detailed listing based on current criteria.\n")

        overview.append(
            "\nNote: This is a high-level, heuristically generated overview of the RAG content. "
            "It aims to highlight potentially important files and entities. 'Entities' are typically "
            "class and function names extracted from code chunks."
        )

        final_overview_str = "\n".join(overview)
        logger.info(f"Generated overview string length: {len(final_overview_str)}")
        return final_overview_str


if __name__ == '__main__':
    # --- Mock VectorDBService and data for testing ---
    logging.basicConfig(level=logging.DEBUG)


    class MockVectorDBService:
        def __init__(self):
            self.collections: Dict[str, List[Dict[str, Any]]] = {}

        def is_ready(self, project_id: str) -> bool:
            return project_id in self.collections

        def get_all_metadata(self, project_id: str) -> List[Dict[str, Any]]:
            return self.collections.get(project_id, [])

        def add_collection_data(self, project_id: str, data: List[Dict[str, Any]]):
            self.collections[project_id] = data


    mock_vdb = MockVectorDBService()
    test_project_id = "test_project_123"
    mock_data = [
        {"source": "src/main.py", "filename": "main.py", "code_entities": ["MyApp", "run_app", "helper_func"],
         "chunk_index": 0, "start_line": 1, "end_line": 10},
        {"source": "src/main.py", "filename": "main.py", "code_entities": ["helper_func", "another_one"],
         "chunk_index": 1, "start_line": 11, "end_line": 20},
        {"source": "src/utils/helpers.py", "filename": "helpers.py", "code_entities": ["format_data", "calculate_sum"],
         "chunk_index": 0, "start_line": 1, "end_line": 15},
        {"source": "src/utils/helpers.py", "filename": "helpers.py", "code_entities": ["_internal_util"],
         "chunk_index": 1, "start_line": 16, "end_line": 25},
        {"source": "core/models.py", "filename": "models.py",
         "code_entities": ["User", "Product", "Order", "_BaseModel"], "chunk_index": 0, "start_line": 1,
         "end_line": 30},
        {"source": "core/models.py", "filename": "models.py", "code_entities": ["User", "get_user_by_id"],
         "chunk_index": 1, "start_line": 31, "end_line": 50},
        {"source": "tests/test_main.py", "filename": "test_main.py", "code_entities": ["TestMyApp", "test_run"],
         "chunk_index": 0, "start_line": 1, "end_line": 12},
        {"source": "requirements.txt", "filename": "requirements.txt", "code_entities": [], "chunk_index": 0,
         "start_line": 1, "end_line": 5},  # No code entities
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 0, "start_line": 1,
         "end_line": 100},  # Many chunks for Readme
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 1, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 2, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 3, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 4, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 5, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 6, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 7, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 8, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 9, "start_line": 1,
         "end_line": 100},
        {"source": "README.md", "filename": "README.md", "code_entities": [], "chunk_index": 10, "start_line": 1,
         "end_line": 100},  # 11 chunks for README
        {"source": "single_root_file.py", "filename": "single_root_file.py",
         "code_entities": ["root_function", "RootClass"], "chunk_index": 0, "start_line": 1, "end_line": 50},
    ]
    # Add more chunks for some files to test thresholds
    for _ in range(25):  # main.py will have 2 + 25 = 27 chunks
        mock_data.append(
            {"source": "src/main.py", "filename": "main.py", "code_entities": [f"gen_entity_{_}"], "chunk_index": _ + 2,
             "start_line": 1, "end_line": 10})
    for _ in range(60):  # models.py will have 2 + 60 = 62 chunks and more entities
        mock_data.append({"source": "core/models.py", "filename": "models.py",
                          "code_entities": [f"ModelEntity_{_}", f"AnotherClass{_}"], "chunk_index": _ + 2,
                          "start_line": 1, "end_line": 10})

    mock_vdb.add_collection_data(test_project_id, mock_data)

    # Test the service
    intelligence_service = ProjectIntelligenceService(vector_db_service=mock_vdb)
    overview_text = intelligence_service.get_condensed_rag_overview_for_summarization(
        project_id=test_project_id,
        max_files_to_detail=5,  # Test with fewer files
        max_entities_per_file=6  # Test with fewer entities
    )

    print("\n--- Generated RAG Overview ---")
    print(overview_text)
    print("--- End of Overview ---\n")

    overview_empty = intelligence_service.get_condensed_rag_overview_for_summarization("non_existent_project")
    print("\n--- Overview for Non-Existent Project ---")
    print(overview_empty)
    print("--- End of Overview ---\n")

    mock_vdb.add_collection_data("empty_project", [])
    overview_no_meta = intelligence_service.get_condensed_rag_overview_for_summarization("empty_project")
    print("\n--- Overview for Empty Project (No Metadata) ---")
    print(overview_no_meta)
    print("--- End of Overview ---\n")