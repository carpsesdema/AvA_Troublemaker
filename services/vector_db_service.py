# services/vector_db_service.py
import logging
import os
import shutil # For deleting collection directories
import pickle # For saving/loading metadata (simple persistence)
# Ensure faiss and numpy are imported
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None # type: ignore
    np = None # type: ignore
    FAISS_AVAILABLE = False
    logging.critical("VectorDBService: FAISS or NumPy library not found. RAG DB cannot function. Install: pip install faiss-cpu numpy")

from typing import List, Dict, Any, Optional, Tuple

# --- Local Imports ---
from utils import constants

# Define GLOBAL_COLLECTION_ID, typically imported from constants
GLOBAL_COLLECTION_ID = constants.GLOBAL_COLLECTION_ID

logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Manages interactions with FAISS vector indices, supporting multiple collections.
    Each collection is stored in a separate directory containing a FAISS index file
    and a pickled metadata file.
    Relies on external embedding generation (embeddings passed to add_embeddings).
    """

    def __init__(self, index_dimension: int, base_persist_directory: Optional[str] = None):
        """
        Initializes the VectorDBService and loads existing collections.
        Args:
            index_dimension (int): The dimension of the embedding vectors.
            base_persist_directory (Optional[str]): The base directory for storing collection data.
                                                    Defaults to constants.RAG_COLLECTIONS_PATH.
        """
        logger.info("VectorDBService initializing (FAISS implementation)...")
        # Initialize _service_ready to False initially
        self._service_ready = False
        self._collections_data: Dict[str, Tuple[faiss.Index, List[Dict[str, Any]]]] = {}

        if not FAISS_AVAILABLE:
            logger.critical("FAISS or NumPy library not available. VectorDBService cannot initialize.")
            return

        if not isinstance(index_dimension, int) or index_dimension <= 0:
            logger.critical(f"Invalid index dimension provided: {index_dimension}. Cannot initialize FAISS.")
            return
        self._index_dim = index_dimension
        logger.info(f"Using index dimension: {self._index_dim}")

        self.base_persist_directory = base_persist_directory or constants.RAG_COLLECTIONS_PATH
        logger.info(f"Using FAISS base persist directory: {self.base_persist_directory}")
        try:
            os.makedirs(self.base_persist_directory, exist_ok=True)
            logger.info(f"FAISS base directory ensured: {self.base_persist_directory}")
        except OSError as e:
            logger.critical(f"Failed to create FAISS base directory '{self.base_persist_directory}': {e}")
            return

        # --- MODIFICATION START: Set _service_ready earlier ---
        # If FAISS is available and basic checks passed, consider the service's core ready
        # for collection management operations like loading or creating.
        if FAISS_AVAILABLE: # Redundant check as it's done above, but good for clarity here
            self._service_ready = True
            logger.info("VectorDBService: Core dependencies (FAISS) ready. Proceeding with collection loading.")
        else:
            # This case should ideally be caught by the `if not FAISS_AVAILABLE:` check at the top of __init__.
            # If it somehow reaches here, it's a critical failure.
            logger.critical("VectorDBService: FAISS not available at a critical point. Cannot set service as ready.")
            return # Service cannot function without FAISS
        # --- MODIFICATION END ---

        # --- Load Existing Collections ---
        self._load_all_collections_from_disk()

        # --- Ensure the global collection exists ---
        # This implicitly creates and loads it if it doesn't exist.
        # Now, get_or_create_collection can correctly check _service_ready.
        global_coll_exists_and_loaded = self.get_or_create_collection(GLOBAL_COLLECTION_ID)

        # Final check for overall service readiness (specifically if global collection is OK)
        if global_coll_exists_and_loaded and GLOBAL_COLLECTION_ID in self._collections_data:
             # _service_ready is already True if we reached here.
             logger.info("VectorDBService initialized successfully and global collection is ready.")
        else:
             # If global collection failed despite service being "core ready", log an error.
             # The service itself (_service_ready=True) might still be usable for other collections if they loaded correctly.
             logger.error(
                 "VectorDBService initialized, but the global collection ('%s') could not be properly created or loaded. "
                 "RAG functionality relying on the global context may be impaired.", GLOBAL_COLLECTION_ID
             )


    def _load_all_collections_from_disk(self):
        """Scans the base directory and attempts to load all collection data."""
        logger.info(f"Scanning directory '{self.base_persist_directory}' for existing collections...")
        loaded_count = 0
        try:
            if os.path.isdir(self.base_persist_directory):
                for item_name in os.listdir(self.base_persist_directory):
                    collection_dir = os.path.join(self.base_persist_directory, item_name)
                    if os.path.isdir(collection_dir):
                        collection_id = item_name # Use directory name as collection ID
                        logger.debug(f"  Attempting to load collection '{collection_id}' from {collection_dir}...")
                        loaded_data = self._load_collection_data(collection_id)
                        if loaded_data:
                            self._collections_data[collection_id] = loaded_data
                            loaded_count += 1
                            logger.debug(f"  Successfully loaded collection '{collection_id}'.")
                        else:
                            logger.warning(f"  Failed to load collection '{collection_id}'. Skipping.")

            logger.info(f"Finished scanning. Loaded {loaded_count} existing collections.")
        except Exception as e:
            logger.exception(f"Error scanning or loading collections from disk: {e}")


    def _load_collection_data(self, collection_id: str) -> Optional[Tuple[faiss.Index, List[Dict[str, Any]]]]:
        """Loads a single collection's FAISS index and metadata from disk."""
        collection_dir = os.path.join(self.base_persist_directory, collection_id)
        index_path = os.path.join(collection_dir, "faiss.index")
        metadata_path = os.path.join(collection_dir, "metadata.pkl") # Using pickle

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.debug(f"Collection files not found for '{collection_id}' at {collection_dir}")
            return None

        try:
            if faiss is None:
                logger.error("FAISS library not available for reading index.")
                return None
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            if not isinstance(metadata, list):
                 logger.error(f"Loaded metadata for '{collection_id}' is not a list ({type(metadata)}). Corrupt?")
                 return None

            if index.ntotal != len(metadata):
                logger.warning(f"Mismatch between FAISS index size ({index.ntotal}) and metadata count ({len(metadata)}) for collection '{collection_id}'. Data might be inconsistent.")

            if index.d != self._index_dim:
                 logger.error(f"Loaded FAISS index dimension ({index.d}) for collection '{collection_id}' does not match service dimension ({self._index_dim}). Cannot load.")
                 return None

            return index, metadata
        except Exception as e:
            logger.error(f"Error loading collection data for '{collection_id}': {e}")
            return None

    def _save_collection_data(self, collection_id: str, index: faiss.Index, metadata: List[Dict[str, Any]]) -> bool:
        """Saves a single collection's FAISS index and metadata to disk."""
        if not FAISS_AVAILABLE:
             logger.error("Cannot save collection data: FAISS not available.")
             return False

        collection_dir = os.path.join(self.base_persist_directory, collection_id)
        index_path = os.path.join(collection_dir, "faiss.index")
        metadata_path = os.path.join(collection_dir, "metadata.pkl")

        try:
            os.makedirs(collection_dir, exist_ok=True)
            faiss.write_index(index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.debug(f"Saved collection '{collection_id}' to {collection_dir}")
            return True
        except Exception as e:
            logger.exception(f"Error saving collection data for '{collection_id}': {e}")
            return False


    def get_or_create_collection(self, collection_id: str) -> bool:
        """
        Ensures a collection exists and is loaded into memory.
        If it doesn't exist on disk, creates a new empty one.
        """
        if not FAISS_AVAILABLE or not self._service_ready:
             logger.error(f"Cannot get/create collection '{collection_id}': FAISS not available ({FAISS_AVAILABLE}) or Service not ready ({self._service_ready}).")
             return False

        if not isinstance(collection_id, str) or not collection_id.strip():
             logger.error("Cannot get/create collection: Invalid or empty collection_id provided.")
             return False

        if collection_id in self._collections_data:
             logger.debug(f"Collection '{collection_id}' already loaded in memory.")
             return True

        loaded_data = self._load_collection_data(collection_id)
        if loaded_data:
             self._collections_data[collection_id] = loaded_data
             logger.info(f"Loaded collection '{collection_id}' from disk.")
             return True

        logger.info(f"Collection '{collection_id}' not found. Creating new...")
        try:
            if faiss is None:
                logger.error("FAISS library became unavailable during get_or_create_collection.")
                return False
            new_index = faiss.IndexFlatL2(self._index_dim)
            new_metadata: List[Dict[str, Any]] = []
            if self._save_collection_data(collection_id, new_index, new_metadata):
                self._collections_data[collection_id] = (new_index, new_metadata)
                logger.info(f"Created and loaded new collection '{collection_id}'.")
                return True
            else:
                 logger.error(f"Failed to save newly created collection '{collection_id}' to disk.")
                 return False
        except Exception as e:
            logger.exception(f"Error creating new collection '{collection_id}': {e}")
            return False

    def is_ready(self, collection_id: Optional[str] = None) -> bool:
        """
        Checks if the service is overall ready (FAISS available, initialized)
        and optionally if a specific collection is loaded in memory.
        """
        logger.debug(
            f"[RAG_VIEW_DEBUG] is_ready called. collection_id: '{collection_id}', "
            f"_service_ready: {self._service_ready}, FAISS_AVAILABLE: {FAISS_AVAILABLE}"
        )
        logger.debug(
            f"[RAG_VIEW_DEBUG] Current _collections_data keys: {list(self._collections_data.keys())}"
        )

        if not FAISS_AVAILABLE or not self._service_ready:
            logger.debug(
                f"[RAG_VIEW_DEBUG] is_ready returning False (FAISS_AVAILABLE={FAISS_AVAILABLE}, "
                f"_service_ready={self._service_ready})"
            )
            return False

        if collection_id is None:
            is_global_loaded = GLOBAL_COLLECTION_ID in self._collections_data
            logger.debug(
                f"[RAG_VIEW_DEBUG] is_ready (global check for '{GLOBAL_COLLECTION_ID}') returning {is_global_loaded}."
            )
            return is_global_loaded

        is_specific_loaded = collection_id in self._collections_data
        logger.debug(
            f"[RAG_VIEW_DEBUG] is_ready (specific check for '{collection_id}') returning {is_specific_loaded}."
        )
        return is_specific_loaded

    def get_available_collections(self) -> List[str]:
        return list(self._collections_data.keys())

    def delete_collection(self, collection_id: str) -> bool:
        if not self._service_ready:
             logger.error(f"Cannot delete collection '{collection_id}': Service not ready."); return False
        if collection_id == GLOBAL_COLLECTION_ID:
            logger.error(f"Cannot delete the global collection '{GLOBAL_COLLECTION_ID}'."); return False

        collection_dir = os.path.join(self.base_persist_directory, collection_id)
        removed_from_memory = False
        if collection_id in self._collections_data:
            del self._collections_data[collection_id]
            removed_from_memory = True
            logger.debug(f"Collection '{collection_id}' removed from memory.")
        else:
            logger.warning(f"Attempted to delete collection '{collection_id}', but it's not loaded in memory.")

        disk_deleted = False
        if os.path.isdir(collection_dir):
             logger.info(f"Attempting to delete collection directory from disk: {collection_dir}")
             try:
                 shutil.rmtree(collection_dir)
                 logger.info(f"Collection directory '{collection_id}' deleted successfully from disk.")
                 disk_deleted = True
             except Exception as e:
                  logger.exception(f"Error deleting collection directory '{collection_id}': {e}")
        else:
             logger.warning(f"Collection directory '{collection_id}' not found on disk either.")
             if not removed_from_memory: return False

        return removed_from_memory or disk_deleted

    def add_embeddings(self, collection_id: str, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> bool:
        if not self.is_ready(collection_id):
            logger.error(f"Cannot add embeddings: Collection '{collection_id}' not loaded or service not ready."); return False
        if not isinstance(embeddings, np.ndarray) or not isinstance(metadatas, list):
            logger.error("Invalid input: embeddings must be a NumPy array, metadatas a list."); return False
        if embeddings.shape[0] != len(metadatas):
            logger.error(f"Input length mismatch: embeddings count ({embeddings.shape[0]}) != metadata count ({len(metadatas)})."); return False
        if embeddings.ndim != 2 or embeddings.shape[1] != self._index_dim: # Added ndim check
             logger.error(f"Embedding dimension mismatch or incorrect shape: Got {embeddings.shape}, expected (n, {self._index_dim})."); return False
        if embeddings.shape[0] == 0:
             logger.warning(f"No embeddings provided to add to collection '{collection_id}'."); return True

        collection_index, collection_metadata = self._collections_data[collection_id]
        logger.info(f"Adding {embeddings.shape[0]} embeddings to collection '{collection_id}'...")
        try:
            collection_index.add(embeddings)
            collection_metadata.extend(metadatas)
            if self._save_collection_data(collection_id, collection_index, collection_metadata):
                 logger.info(f"Successfully added and saved {embeddings.shape[0]} embeddings to '{collection_id}'.")
                 return True
            else:
                 logger.error(f"Successfully added embeddings to '{collection_id}' index but FAILED TO SAVE metadata/index.")
                 return False
        except Exception as e:
            logger.exception(f"Error adding embeddings to collection '{collection_id}': {e}")
            return False

    def search(self, collection_id: str, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready(collection_id):
            logger.error(f"Cannot search: Collection '{collection_id}' not loaded or service not ready."); return []
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 2 or query_embedding.shape[0] != 1 or query_embedding.shape[1] != self._index_dim :
            logger.error(f"Invalid query embedding format/dimension: Expected (1, {self._index_dim}), got {query_embedding.shape}."); return []
        if not isinstance(k, int) or k <= 0:
             logger.warning(f"Invalid k value ({k}), using default 5."); k = 5

        collection_index, collection_metadata = self._collections_data[collection_id]
        if collection_index.ntotal == 0:
             logger.info(f"Collection '{collection_id}' is empty. Returning no search results."); return []
        effective_k = min(k, collection_index.ntotal)
        if effective_k == 0: return []

        logger.info(f"Searching collection '{collection_id}' ({collection_index.ntotal} items) with k={effective_k}...")
        try:
            distances, indices = collection_index.search(query_embedding, effective_k)
            results = []
            for i in range(effective_k):
                vector_index_in_collection = indices[0][i]
                distance_val = distances[0][i]
                if vector_index_in_collection == -1: continue
                if 0 <= vector_index_in_collection < len(collection_metadata):
                    metadata = collection_metadata[vector_index_in_collection]
                    content = metadata.get('content', '[Content not found in metadata]')
                    if "collection_id" not in metadata: metadata["collection_id"] = collection_id
                    results.append({'content': content, 'metadata': metadata, 'distance': float(distance_val)})
                else:
                    logger.warning(f"Search returned index {vector_index_in_collection} for collection '{collection_id}' out of bounds for metadata ({len(collection_metadata)}).")
            logger.info(f"Search completed for collection '{collection_id}'. Found {len(results)} relevant items.")
            return results
        except Exception as e:
            logger.exception(f"Error searching collection '{collection_id}': {e}")
            return []

    def get_all_metadata(self, collection_id: str) -> List[Dict[str, Any]]:
        if not self.is_ready(collection_id):
            logger.error(f"Cannot get metadata: Collection '{collection_id}' not loaded or service not ready."); return []
        _, collection_metadata = self._collections_data[collection_id]
        logger.info(f"Retrieving all metadata ({len(collection_metadata)} items) from collection '{collection_id}'.")
        return list(collection_metadata)

    def get_collection_size(self, collection_id: str) -> int:
        logger.debug(f"[RAG_VIEW_DEBUG] get_collection_size called for ID: '{collection_id}'")
        if not self.is_ready(collection_id):
            logger.warning(f"[RAG_VIEW_DEBUG] Cannot get size for '{collection_id}': is_ready() returned False."); return -1
        try:
            if collection_id not in self._collections_data:
                 logger.warning(f"[RAG_VIEW_DEBUG] Cannot get size: Collection ID '{collection_id}' not found in _collections_data, though is_ready passed for it. This is unexpected."); return -1

            collection_index, _ = self._collections_data[collection_id]
            count = collection_index.ntotal
            logger.debug(f"[RAG_VIEW_DEBUG] Collection '{collection_id}' FAISS index.ntotal = {count}")
            return count
        except KeyError:
            logger.warning(f"[RAG_VIEW_DEBUG] Cannot get size: Collection ID '{collection_id}' not found in _collections_data (KeyError during access)."); return -1
        except Exception as e:
            logger.exception(f"[RAG_VIEW_DEBUG] Error getting count for collection '{collection_id}': {e}"); return -1

    def clear_collection(self, collection_id: str) -> bool:
        if collection_id == GLOBAL_COLLECTION_ID:
             logger.error(f"Clearing the global collection ('{GLOBAL_COLLECTION_ID}') is not permitted via this method."); return False
        if not self.is_ready(collection_id):
             logger.error(f"Cannot clear collection '{collection_id}': Not loaded or service not ready."); return False

        logger.warning(f"Attempting to clear all items from collection '{collection_id}' by re-creating index and metadata...")
        try:
            if faiss is None:
                logger.error("FAISS library not available for clearing collection.")
                return False
            new_index = faiss.IndexFlatL2(self._index_dim)
            new_metadata: List[Dict[str, Any]] = []
            if self._save_collection_data(collection_id, new_index, new_metadata):
                 self._collections_data[collection_id] = (new_index, new_metadata)
                 logger.info(f"Collection '{collection_id}' cleared successfully (by recreating).")
                 return True
            else:
                 logger.error(f"Failed to save cleared collection '{collection_id}' to disk.")
                 return False
        except Exception as e:
            logger.exception(f"Error during the process of clearing collection '{collection_id}': {e}")
            return False