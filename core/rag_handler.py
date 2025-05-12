# core/rag_handler.py
# UPDATED - Modified get_formatted_context to accept focus_paths list.
# UPDATED - Implemented boosting for chunks from focused files during re-ranking.

import logging
import re
import os # Added for path normalization
from typing import List, Optional, Dict, Any, Set, Tuple

# Assuming services are in the parent directory 'services' relative to 'core'
# Adjust import paths if your structure differs
try:
    from services.upload_service import UploadService
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
    from utils import constants
except ImportError as e:
    logging.critical(f"RagHandler: Failed to import services/utils: {e}")
    # Define fallbacks if necessary for type hinting or basic functionality
    UploadService = type("UploadService", (object,), {})
    VectorDBService = type("VectorDBService", (object,), {})
    GLOBAL_COLLECTION_ID = "global_collection"
    constants = type("constants", (object,), {"RAG_NUM_RESULTS": 5, "RAG_CHUNK_SIZE": 1000, "RAG_CHUNK_OVERLAP": 150}) # Dummy constants


logger = logging.getLogger(__name__)

class RagHandler:
    """
    Handles RAG-specific logic: determining need, extracting entities,
    querying, re-ranking (considering focus paths), and formatting context.
    """

    _TECHNICAL_KEYWORDS = { # Keywords indicating RAG might be useful
        'python', 'code', 'error', 'fix', 'implement', 'explain', 'how to',
        'def ', 'class ', 'import ', ' module', ' function', ' method',
        ' attribute', ' bug', ' issue', ' traceback', ' install', ' pip',
        ' library', ' package', ' api', ' request', ' data', ' typeerror',
        ' indexerror', ' keyerror', ' exception', ' syntax', ' logic', ' algorithm',
        ' self.', ' args', ' kwargs', ' return', ' yield', ' async', ' await',
        ' decorator', ' lambda', ' list', ' dict', ' tuple', ' set', ' numpy', ' pandas',
        'pyqt6', 'pyqt', 'qwidget', 'qapplication', 'qdialog', 'qlabel', 'qpixmap',
        'rag', 'vector db', 'embedding', 'chunking', 'context', 'prompt', 'llm', 'ollama', 'faiss',
        'collection', 'document', 'similarity', 'query', 'index',
        'my code', 'my project', 'refactor this', 'debug this', 'in my implementation',
        'according to the file', 'based on the documents', 'in these files', 'the provided context',
        'summarize this', 'search', 'find', 'lookup', 'relevant context',
        'change', 'update', 'modify',
    }
    _GREETING_PATTERNS = re.compile(r"^\s*(hi|hello|hey|yo|sup|good\s+(morning|afternoon|evening)|how\s+are\s+you)\b.*", re.IGNORECASE)
    _CODE_FENCE_PATTERN = re.compile(r"```")


    def __init__(self, upload_service: UploadService, vector_db_service: VectorDBService):
        if not isinstance(upload_service, UploadService):
            raise TypeError("RagHandler requires a valid UploadService instance.")
        if not isinstance(vector_db_service, VectorDBService):
             raise TypeError("RagHandler requires a valid VectorDBService instance.")

        self._upload_service = upload_service
        self._vector_db_service = vector_db_service
        logger.info("RagHandler initialized.")

    def should_perform_rag(self, query: str, rag_available: bool, rag_initialized: bool) -> bool:
        """Checks if the query likely requires RAG based on keywords and structure."""
        if not rag_available or not rag_initialized:
            return False
        if not query:
            return False

        query_lower = query.lower().strip()
        if len(query) < 15 and self._GREETING_PATTERNS.match(query_lower):
            return False
        if len(query) < 10:
            return False
        if self._CODE_FENCE_PATTERN.search(query):
            return True
        if any(keyword in query_lower for keyword in self._TECHNICAL_KEYWORDS):
            return True
        if re.search(r"[_.(){}\[\]=:]", query) and len(query) > 15:
            return True
        return False

    def extract_code_entities(self, query: str) -> Set[str]:
        """Extracts potential function/class names from a user query using simple regex."""
        entities = set()
        if not query:
            return entities
        call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(' # Function calls
        def_class_pattern = r'\b(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)' # Definitions
        try:
            for match in re.finditer(call_pattern, query): entities.add(match.group(1))
            for match in re.finditer(def_class_pattern, query): entities.add(match.group(1))
        except Exception as e:
            logger.warning(f"Regex error during query entity extraction: {e}")

        entities = {e for e in entities if len(e) > 2 and e not in ['def', 'class', 'self']}
        if entities:
            logger.debug(f"Extracted potential code entities from query: {entities}")
        return entities

    def get_formatted_context(
        self,
        query: str,
        query_entities: Set[str],
        project_id: Optional[str],
        focus_paths: Optional[List[str]] = None, # MODIFIED: Added focus_paths parameter
        is_modification_request: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Retrieves, re-ranks (considering focus paths), and formats RAG context.

        Args:
            query: The user's query text.
            query_entities: Pre-extracted potential code entities from the query.
            project_id: The current project ID (or None for global).
            focus_paths: Optional list of absolute file/directory paths to prioritize.
            is_modification_request: Flag indicating if more context might be needed.

        Returns:
            A tuple containing:
            - The formatted RAG context string (empty if no context found/error).
            - A list of collection IDs that were successfully queried.
        """
        context_str = ""
        queried_collections = []

        # Normalize focus paths for reliable comparison
        normalized_focus_paths = set()
        if focus_paths:
            try:
                # Normalize paths (e.g., resolve '..', handle slashes) and make absolute
                # Using os.path.normpath and os.path.abspath might be needed depending on input format
                normalized_focus_paths = {os.path.normcase(os.path.abspath(p)) for p in focus_paths}
                logger.info(f"Normalized focus paths for RAG: {normalized_focus_paths}")
            except Exception as e_norm:
                logger.error(f"Error normalizing focus paths {focus_paths}: {e_norm}")
                normalized_focus_paths = set() # Clear on error

        # Determine collections to query
        collections_to_query = []
        if self._vector_db_service.is_ready(GLOBAL_COLLECTION_ID):
             collections_to_query.append(GLOBAL_COLLECTION_ID)
        if project_id and project_id != GLOBAL_COLLECTION_ID and self._vector_db_service.is_ready(project_id):
             collections_to_query.append(project_id)
        elif project_id and project_id != GLOBAL_COLLECTION_ID:
             logger.warning(f"Project collection '{project_id}' not ready, skipping.")

        if not collections_to_query:
            logger.warning("RAG context requested but no ready collections to query.")
            return "", []

        logger.info(f"Attempting RAG retrieval from collections: {collections_to_query}...")
        try:
            if not hasattr(self._upload_service, 'query_vector_db'):
                raise TypeError("UploadService missing required 'query_vector_db' method.")

            num_initial_results = constants.RAG_NUM_RESULTS * (3 if is_modification_request else 2)
            num_final_results = constants.RAG_NUM_RESULTS

            # 1. Retrieve relevant chunks (semantic search)
            relevant_chunks = self._upload_service.query_vector_db(
                query,
                collection_ids=collections_to_query,
                n_results=num_initial_results
            )
            queried_collections = list(set(c.get("metadata", {}).get("collection_id", "N/A") for c in relevant_chunks if c.get("metadata", {}).get("collection_id") != "N/A"))

            # 2. Re-ranking based on code entities AND focus paths
            entity_boost_factor = 0.80 # Boost for matching code entities
            focus_boost_factor = 0.60 # Stronger boost for being in a focused file
            boosted_by_entity_count = 0
            boosted_by_focus_count = 0

            if relevant_chunks:
                logger.debug(f"Re-ranking {len(relevant_chunks)} chunks. Entities: {query_entities}, Focus Paths: {normalized_focus_paths}")
                for chunk in relevant_chunks:
                    metadata = chunk.get('metadata')
                    distance = chunk.get('distance') # Original semantic distance

                    if not isinstance(metadata, dict) or not isinstance(distance, (int, float)):
                        logger.warning(f"Skipping chunk with invalid metadata or distance: {chunk}")
                        continue

                    boost_applied = False

                    # --- Apply Focus Boost (Highest Priority) ---
                    chunk_source_path = metadata.get('source') # 'source' should hold the full path
                    if normalized_focus_paths and chunk_source_path:
                        try:
                            # Normalize the chunk's source path for comparison
                            norm_chunk_path = os.path.normcase(os.path.abspath(chunk_source_path))
                            # Check if the chunk's path is one of the focused paths
                            # or if it's within a focused directory
                            is_focused = False
                            for focus_path in normalized_focus_paths:
                                if os.path.isdir(focus_path):
                                     # Check if chunk path starts with the directory path
                                     if norm_chunk_path.startswith(focus_path + os.sep):
                                         is_focused = True
                                         break
                                elif norm_chunk_path == focus_path: # Direct file match
                                     is_focused = True
                                     break

                            if is_focused:
                                chunk['distance'] *= focus_boost_factor # Apply focus boost
                                chunk['boost_reason'] = 'focus' # Add reason for debugging
                                boosted_by_focus_count += 1
                                boost_applied = True
                                # logger.debug(f"  Applied FOCUS boost to chunk from '{chunk_source_path}'. New dist: {chunk['distance']:.4f}")

                        except Exception as e_focus_boost:
                             logger.error(f"Error applying focus boost for chunk path '{chunk_source_path}': {e_focus_boost}")


                    # --- Apply Entity Boost (Lower Priority - only if not already focus-boosted) ---
                    if not boost_applied and query_entities and 'code_entities' in metadata:
                        chunk_entities = set(metadata.get('code_entities', [])) # Ensure it's a list/set
                        if not query_entities.isdisjoint(chunk_entities):
                            chunk['distance'] *= entity_boost_factor # Apply entity boost
                            chunk['boost_reason'] = 'entity' # Add reason for debugging
                            boosted_by_entity_count += 1
                            boost_applied = True
                            # logger.debug(f"  Applied ENTITY boost to chunk from '{chunk_source_path}'. New dist: {chunk['distance']:.4f}")


                if boosted_by_focus_count > 0 or boosted_by_entity_count > 0:
                    logger.info(f"Applied RAG boost: Focus={boosted_by_focus_count}, Entity={boosted_by_entity_count} chunks.")


            # 3. Sort and select final results based on potentially modified distances
            if relevant_chunks:
                valid_chunks = [res for res in relevant_chunks if isinstance(res.get('distance'), (int, float))]
                # Sort by distance (lower is better)
                sorted_results = sorted(valid_chunks, key=lambda x: x.get('distance', float('inf')))
                final_results = sorted_results[:num_final_results]

                # 4. Assemble context string
                context_parts = []
                retrieved_chunks_details = []
                for i, chunk in enumerate(final_results):
                    metadata = chunk.get("metadata", {})
                    filename = metadata.get("filename", "unknown_source") # Use filename for display
                    collection_id = metadata.get("collection_id", "N/A")
                    code_content = metadata.get("content", "[Content Missing]") # Get content from metadata
                    distance = chunk.get('distance', -1.0)
                    boost_reason = chunk.get('boost_reason') # Get boost reason if applied

                    # Optional debug info
                    debug_info = f"(Dist: {distance:.4f}"
                    if boost_reason: debug_info += f", Boost: {boost_reason}"
                    if query_entities and isinstance(metadata.get('code_entities'), list):
                        matches = query_entities.intersection(set(metadata['code_entities']))
                        if matches: debug_info += f", Matches: {', '.join(matches)}"
                    debug_info += ")"

                    context_parts.append(f"--- Snippet {i+1} from `{filename}` (Collection: {collection_id}) {debug_info} ---\n```python\n{code_content}\n```\n")
                    retrieved_chunks_details.append(f"{filename} {debug_info}")

                if context_parts:
                    context_str = ("--- Relevant Code Context Start ---\n" + "\n".join(context_parts) + "--- Relevant Code Context End ---")
                    logger.info(f"Final RAG context includes {len(final_results)} chunks: [{', '.join(retrieved_chunks_details)}]")
                else:
                    logger.info("No valid chunks remained after processing/sorting.")
            else:
                logger.info(f"No relevant RAG context found in collections {collections_to_query}.")

        except Exception as e_rag:
            logger.exception("Error retrieving/re-ranking RAG context:")
            context_str = "[Error retrieving RAG context]"

        return context_str, queried_collections
