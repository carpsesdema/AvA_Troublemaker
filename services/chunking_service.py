# services/chunking_service.py
# UPDATED FILE - Added start_line and end_line calculation to chunk metadata

import os
import logging
import bisect # Using bisect for efficient line number lookup
from typing import List, Dict, Any

# --- LangChain imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, PythonCodeTextSplitter

# --- Local Imports ---
from utils import constants

logger = logging.getLogger(__name__)

class ChunkingService:
    """
    Handles chunking of documents using LangChain splitters.
    Adds start_line and end_line numbers to chunk metadata.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        logger.info(f"ChunkingService initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            fallback_size = 1000
            logger.warning(f"Invalid chunk_size ({chunk_size}), using fallback: {fallback_size}")
            chunk_size = fallback_size
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
             fallback_overlap = int(chunk_size * 0.15) # 15% overlap as fallback
             logger.warning(f"Invalid chunk_overlap ({chunk_overlap}) for chunk_size {chunk_size}, using fallback: {fallback_overlap}")
             chunk_overlap = fallback_overlap

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # --- Instantiate the LangChain splitters ---
        # 1. Default/Fallback Recursive Splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        logger.info(f"Using LangChain RecursiveCharacterTextSplitter (size={self.chunk_size}, overlap={self.chunk_overlap}) as default.")

        # 2. Python Code Splitter
        try:
            self.python_splitter = PythonCodeTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
                # PythonCodeTextSplitter automatically uses appropriate Python separators
            )
            logger.info(f"Initialized LangChain PythonCodeTextSplitter (size={self.chunk_size}, overlap={self.chunk_overlap}).")
        except Exception as e:
            logger.error(f"Failed to initialize PythonCodeTextSplitter: {e}. Falling back to recursive for Python.", exc_info=True)
            self.python_splitter = None # Mark as unavailable if init fails


    def _get_line_start_indices(self, text: str) -> List[int]:
        """Calculates the starting character index of each line."""
        indices = [0] # Line 1 starts at index 0
        current_pos = 0
        while True:
            next_newline = text.find('\n', current_pos)
            if next_newline == -1:
                break
            indices.append(next_newline + 1)
            current_pos = next_newline + 1
        return indices

    def _get_line_number(self, char_index: int, line_start_indices: List[int]) -> int:
        """Finds the 1-based line number for a given character index."""
        # bisect_right finds the insertion point to maintain sorted order.
        # The insertion point corresponds to the line number (add 1 for 1-based index).
        line_num_0_based = bisect.bisect_right(line_start_indices, char_index) - 1
        return max(1, line_num_0_based + 1) # Ensure minimum line number is 1

    def chunk_document(self, content: str, source_id: str, file_ext: str) -> List[Dict[str, Any]]:
        """
        Chunks a document using the appropriate LangChain splitter based on file extension.
        Adds start_line and end_line to metadata.

        Args:
            content: The text content of the document.
            source_id: The original identifier (e.g., file path) of the document.
            file_ext: The lowercased file extension (e.g., '.py', '.txt').

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and contains 'content' and 'metadata' (including start/end line). Returns empty list on error.
        """
        filename_base = os.path.basename(source_id) if source_id else "unknown_source"
        logger.debug(f"Chunking document: {filename_base} (ext: {file_ext})")
        if not isinstance(content, str) or not content.strip():
            logger.warning(f"Skipping chunking for empty content: {filename_base}")
            return []

        # --- Choose the appropriate splitter ---
        splitter_to_use = None
        if file_ext == '.py' and self.python_splitter:
            logger.debug(f"Using PythonCodeTextSplitter for '{filename_base}'")
            splitter_to_use = self.python_splitter
        else:
            if file_ext == '.py': # Log fallback specifically for Python if splitter failed init
                 logger.warning(f"PythonCodeTextSplitter not available, falling back to RecursiveCharacterTextSplitter for '{filename_base}'.")
            logger.debug(f"Using default RecursiveCharacterTextSplitter for '{filename_base}'")
            splitter_to_use = self.recursive_splitter
        # ------------------------------------

        if splitter_to_use is None: # Should not happen if recursive_splitter is always initialized
             logger.error(f"No valid text splitter available for '{filename_base}'. Cannot chunk.")
             return []

        try:
            # --- Pre-calculate line start indices ---
            line_start_indices = self._get_line_start_indices(content)
            logger.debug(f"Calculated {len(line_start_indices)} line start indices for '{filename_base}'.")
            # ---------------------------------------

            logger.debug(f"Splitting text for '{filename_base}' (length: {len(content)}) using {type(splitter_to_use).__name__}")
            split_texts = splitter_to_use.split_text(content)
            logger.debug(f"Split into {len(split_texts)} chunks for '{filename_base}'")

            # --- Format output ---
            chunks = []
            current_pos = 0 # Track position in original content for approximate start_index search

            for i, text_chunk in enumerate(split_texts):
                if not text_chunk.strip(): # Skip empty chunks if splitter produces them
                     logger.debug(f"Skipping empty chunk {i} for '{filename_base}'")
                     continue

                # --- Find approximate character start/end indices ---
                chunk_start_in_original = -1
                # Try finding from current_pos first for better accuracy with overlaps
                try:
                    chunk_start_in_original = content.find(text_chunk, current_pos)
                except Exception:
                    pass # Ignore potential errors in find

                if chunk_start_in_original == -1:
                    # Fallback: search from the beginning (less accurate with identical chunks)
                    try:
                        chunk_start_in_original = content.find(text_chunk)
                    except Exception:
                        pass

                if chunk_start_in_original == -1:
                     # If still not found, use the previous position as a rough estimate
                     chunk_start_in_original = current_pos
                     logger.warning(f"Could not reliably find start index for chunk {i} of '{filename_base}'. Using estimated position {current_pos}.")

                chunk_end_in_original = chunk_start_in_original + len(text_chunk)
                # ----------------------------------------------------

                # --- Calculate start_line and end_line ---
                start_line = self._get_line_number(chunk_start_in_original, line_start_indices)
                # For end_line, we need the line number containing the *last character* of the chunk.
                last_char_index = max(0, chunk_end_in_original - 1)
                end_line = self._get_line_number(last_char_index, line_start_indices)
                # ---------------------------------------

                metadata = {
                    "source": str(source_id),
                    "filename": filename_base,
                    "chunk_index": i, # Index of the chunk within this document
                    "start_index": chunk_start_in_original, # Approximate start char index
                    "content": text_chunk, # IMPORTANT: Store original chunk content in metadata for DB lookup/display
                    "start_line": start_line, # <-- ADDED
                    "end_line": end_line, # <-- ADDED
                }
                chunks.append({"content": text_chunk, "metadata": metadata})

                # Update position for next search (move past the start of this chunk)
                # Add a small increment to handle potential overlaps correctly
                current_pos = chunk_start_in_original + 1

            logger.info(f"{type(splitter_to_use).__name__} created {len(chunks)} non-empty chunks for {filename_base} (with line numbers)")
            return chunks

        except Exception as e:
            logger.exception(f"Error using {type(splitter_to_use).__name__} for {filename_base}: {e}")
            return [] # Return empty list on error