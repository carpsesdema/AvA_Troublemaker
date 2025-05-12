# services/session_service.py
# UPDATED - Adapts to save/load a consolidated `project_context_data` dictionary.
#         - `_load_from_file` now reconstructs this dictionary.
#         - `_save_to_file` (via callers) now accepts this dictionary.
#         - Method signatures for public session methods updated.
#         - Enhanced backward compatibility for old session formats.

import os
import json
import re
import datetime
import logging
from typing import Dict, Any, Optional, Tuple, List

# Use constants for paths and filenames
from utils import constants
from core.models import ChatMessage # For ChatMessage type hints and instantiation

logger = logging.getLogger(__name__)

class SessionService:
    """
    Handles loading, saving, listing, and deleting chat sessions.
    Now works with a consolidated project_context_data dictionary.
    """

    def __init__(self):
        try:
            os.makedirs(constants.USER_DATA_DIR, exist_ok=True)
            logger.info(f"User data directory ensured: {constants.USER_DATA_DIR}")
        except OSError as e:
            logger.critical(f"CRITICAL: Could not create base data directory {constants.USER_DATA_DIR}: {e}")
        try:
            os.makedirs(constants.CONVERSATIONS_DIR, exist_ok=True)
            logger.info(f"Conversations directory ensured: {constants.CONVERSATIONS_DIR}")
        except OSError as e:
             logger.error(f"Could not create conversations directory {constants.CONVERSATIONS_DIR}: {e}")
        logger.info("SessionService initialized.")
        logger.info(f"  Conversations Path: {constants.CONVERSATIONS_DIR}")
        logger.info(f"  Last Session Path: {constants.LAST_SESSION_FILEPATH}")

    # --- Internal Load Helper ---
    def _load_from_file(self, filepath: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Internal helper to load and parse session data.
        Reconstructs a project_context_data dictionary.

        Returns:
            A tuple: (model_name, personality_prompt, project_context_data_dict)
            project_context_data_dict is None if loading fails or file is empty/invalid.
        """
        logger.debug(f"  Internal load: Reading file {filepath}")
        model_name, personality_prompt = None, None
        project_context_data: Optional[Dict[str, Any]] = None # Initialize to None

        try:
            with open(filepath, "r", encoding="utf-8") as f: file_content = f.read()
            if not file_content.strip():
                logger.warning(f"Session file is empty: {filepath}"); return model_name, personality_prompt, None

            data = json.loads(file_content)
            if not isinstance(data, dict):
                logger.error(f"Invalid format: Session data not dict in {filepath}"); return model_name, personality_prompt, None

            model_name = data.get("model_name")
            personality_prompt = data.get("personality_prompt")

            # --- Reconstruct project_context_data ---
            # Attempt to load new format first
            if "project_context_data" in data and isinstance(data["project_context_data"], dict):
                loaded_pcd = data["project_context_data"]
                # Basic validation of the loaded PCD structure
                if isinstance(loaded_pcd.get("project_histories"), dict) and \
                   isinstance(loaded_pcd.get("project_names"), dict):
                    project_context_data = {
                        "project_histories": {}, # Will be filled with ChatMessage objects
                        "project_names": loaded_pcd.get("project_names", {}),
                        "current_project_id": loaded_pcd.get("current_project_id")
                    }
                    # Deserialize ChatMessage objects within project_histories
                    raw_histories = loaded_pcd.get("project_histories", {})
                    for pid, history_list_raw in raw_histories.items():
                        if isinstance(history_list_raw, list):
                            deserialized_history = []
                            for item_dict in history_list_raw:
                                try:
                                    # Ensure all necessary fields are present for ChatMessage
                                    role = item_dict.get('role')
                                    parts_raw = item_dict.get('parts')
                                    timestamp = item_dict.get('timestamp')
                                    metadata = item_dict.get('metadata')

                                    if role is None or parts_raw is None:
                                        logger.warning(f"Skipping history item in project '{pid}' due to missing role/parts in {filepath}.")
                                        continue

                                    if isinstance(parts_raw, str): parts_list = [parts_raw]
                                    elif isinstance(parts_raw, list): parts_list = [p for p in parts_raw if isinstance(p, (str, dict))]
                                    else: parts_list = [] # Invalid parts format

                                    deserialized_history.append(ChatMessage(role=str(role), parts=parts_list, timestamp=timestamp, metadata=metadata))
                                except Exception as e_msg:
                                    logger.warning(f"Error deserializing ChatMessage item in project '{pid}' from {filepath}: {e_msg}. Skipping item.")
                            project_context_data["project_histories"][pid] = deserialized_history
                        else:
                             logger.warning(f"History for project '{pid}' in {filepath} is not a list. Skipping.")
                             project_context_data["project_histories"][pid] = []
                else:
                    logger.warning(f"Loaded 'project_context_data' from {filepath} has invalid structure. Attempting backward compatibility.")
                    project_context_data = None # Fallback to old format parsing

            # --- Backward Compatibility: If new format failed or not present, try old format ---
            if project_context_data is None:
                logger.info(f"Attempting to load old session format from {filepath} (no valid 'project_context_data' found).")
                raw_project_histories = data.get("project_histories") # From previous refactor
                raw_history = data.get("history") # From very old format
                last_active_project_id_old = data.get("project_id") # Old key for active project

                temp_project_histories = {}
                temp_project_names = {} # Names were not saved in very old formats

                if isinstance(raw_project_histories, dict): # Previous refactor format
                    for pid, history_list_raw in raw_project_histories.items():
                        deserialized_history = []
                        if isinstance(history_list_raw, list):
                            for item_dict in history_list_raw:
                                try:
                                    role = item_dict.get('role')
                                    parts_raw = item_dict.get('parts')
                                    timestamp = item_dict.get('timestamp')
                                    metadata = item_dict.get('metadata')
                                    if role is None or parts_raw is None: continue
                                    if isinstance(parts_raw, str): parts_list = [parts_raw]
                                    elif isinstance(parts_raw, list): parts_list = [p for p in parts_raw if isinstance(p, (str, dict))]
                                    else: parts_list = []
                                    deserialized_history.append(ChatMessage(role=str(role), parts=parts_list, timestamp=timestamp, metadata=metadata))
                                except Exception: pass # Skip malformed messages
                        temp_project_histories[pid] = deserialized_history
                        temp_project_names[pid] = pid # Use ID as name if not found
                    current_project_id_to_use = last_active_project_id_old
                elif isinstance(raw_history, list): # Very old format (single history)
                    deserialized_history = []
                    for item_dict in raw_history:
                        try:
                            role = item_dict.get('role')
                            parts_raw = item_dict.get('parts')
                            timestamp = item_dict.get('timestamp')
                            metadata = item_dict.get('metadata')
                            if role is None or parts_raw is None: continue
                            if isinstance(parts_raw, str): parts_list = [parts_raw]
                            elif isinstance(parts_raw, list): parts_list = [p for p in parts_raw if isinstance(p, (str, dict))]
                            else: parts_list = []
                            deserialized_history.append(ChatMessage(role=str(role), parts=parts_list, timestamp=timestamp, metadata=metadata))
                        except Exception: pass
                    # Assume this single history belongs to the old project_id or global
                    pid_for_old_history = last_active_project_id_old or constants.GLOBAL_COLLECTION_ID
                    temp_project_histories[pid_for_old_history] = deserialized_history
                    temp_project_names[pid_for_old_history] = pid_for_old_history
                    current_project_id_to_use = pid_for_old_history
                else: # No recognizable history format
                    current_project_id_to_use = None

                # Ensure global collection exists if we loaded something
                if temp_project_histories and constants.GLOBAL_COLLECTION_ID not in temp_project_histories:
                    temp_project_histories[constants.GLOBAL_COLLECTION_ID] = []
                    temp_project_names[constants.GLOBAL_COLLECTION_ID] = "Global Context" # Default name

                if temp_project_histories: # If any history was parsed from old formats
                    project_context_data = {
                        "project_histories": temp_project_histories,
                        "project_names": temp_project_names,
                        "current_project_id": current_project_id_to_use
                    }
                    logger.info(f"Successfully converted old session format from {filepath} to new project_context_data structure.")
                else:
                    logger.warning(f"Could not parse any history from old format in {filepath}.")
                    # project_context_data remains None

            # Final check: if project_context_data is still None, create a default empty one
            if project_context_data is None:
                 logger.warning(f"No valid project context data could be loaded or converted from {filepath}. Initializing default empty PCD.")
                 project_context_data = {
                     "project_histories": {constants.GLOBAL_COLLECTION_ID: []},
                     "project_names": {constants.GLOBAL_COLLECTION_ID: "Global Context"},
                     "current_project_id": constants.GLOBAL_COLLECTION_ID
                 }

            loaded_projects_count = len(project_context_data.get("project_histories", {}))
            total_messages = sum(len(h) for h in project_context_data.get("project_histories", {}).values())
            logger.info(f"Session loaded from {os.path.basename(filepath)}. Model: {model_name}, Pers: {'Set' if personality_prompt else 'None'}, ActiveProj: {project_context_data.get('current_project_id')}, Projects: {loaded_projects_count}, TotalMsgs: {total_messages}")

        except json.JSONDecodeError as e: logger.error(f"JSON decode error in {filepath}: {e}"); return None, None, None
        except OSError as e: logger.error(f"OS error reading {filepath}: {e}"); return None, None, None
        except Exception as e: logger.exception(f"Unexpected error loading {filepath}: {e}"); return None, None, None

        return model_name, personality_prompt, project_context_data

    # --- Internal Save Helper ---
    def _chatmessage_to_dict(self, msg: ChatMessage) -> Dict[str, Any]:
        serializable_metadata = None
        if isinstance(msg.metadata, dict):
             serializable_metadata = {k: v for k, v in msg.metadata.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        return {"role": msg.role, "parts": msg.parts, "timestamp": msg.timestamp, "metadata": serializable_metadata}

    def _save_to_file(self, filepath: str, data_to_save: Dict[str, Any]) -> bool:
        logger.debug(f"  Internal save: Writing to file {filepath}")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Before saving project_context_data, ensure ChatMessages are dicts
            if "project_context_data" in data_to_save and isinstance(data_to_save["project_context_data"], dict):
                pcd = data_to_save["project_context_data"]
                if "project_histories" in pcd and isinstance(pcd["project_histories"], dict):
                    serializable_histories = {}
                    for pid, history_list_obj in pcd["project_histories"].items():
                        if isinstance(history_list_obj, list):
                            serializable_histories[pid] = [self._chatmessage_to_dict(msg) for msg in history_list_obj if isinstance(msg, ChatMessage)]
                        else:
                            serializable_histories[pid] = [] # or log error
                    pcd["project_histories"] = serializable_histories # Replace with serialized version

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Session data saved to {os.path.basename(filepath)}.")
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.exception(f"Error saving session file {filepath}: {e}")
            return False

    # --- Last Session State ---
    def get_last_session(self) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Loads the state from the last session file.
        Returns model_name, personality, and project_context_data dictionary.
        """
        logger.info(f"Attempting to load last session state from: {constants.LAST_SESSION_FILEPATH}")
        if not os.path.exists(constants.LAST_SESSION_FILEPATH):
             logger.info("Last session file not found. Starting fresh.")
             # Return None for project_context_data to indicate fresh start
             return None, None, None
        return self._load_from_file(constants.LAST_SESSION_FILEPATH)

    def save_last_session(self,
                          model_name: Optional[str],
                          personality: Optional[str],
                          project_context_data: Dict[str, Any]): # <<< CHANGE: Accept project_context_data dict >>>
        """Saves the current state to the last session file."""
        logger.info(f"Attempting to save last session state to: {constants.LAST_SESSION_FILEPATH}")
        if not isinstance(project_context_data, dict):
            logger.error(f"Project context data for saving last session is not a dict ({type(project_context_data)}). Aborting save.")
            return False

        data_to_save = {
            "model_name": model_name,
            "personality_prompt": personality,
            # <<< CHANGE: Store the entire project_context_data dictionary >>>
            "project_context_data": project_context_data,
            "metadata": {
                "save_timestamp": datetime.datetime.now().isoformat(),
                "source": "last_session"
            }
        }
        return self._save_to_file(constants.LAST_SESSION_FILEPATH, data_to_save)

    def clear_last_session_file(self) -> bool: # No change
        logger.info(f"Attempting to clear last session state file: {constants.LAST_SESSION_FILEPATH}")
        try:
            if os.path.exists(constants.LAST_SESSION_FILEPATH): os.remove(constants.LAST_SESSION_FILEPATH); logger.info("Last session state file deleted.")
            else: logger.info("Last session state file did not exist.")
            return True
        except OSError as e: logger.error(f"Error deleting last session file {constants.LAST_SESSION_FILEPATH}: {e}"); return False
        except Exception as e: logger.exception(f"Unexpected error clearing last session file: {e}"); return False

    # --- Named Conversation Management ---
    def list_sessions(self) -> List[str]: # No change
        logger.info(f"Listing conversations in: {constants.CONVERSATIONS_DIR}")
        full_paths = []
        try:
            if not os.path.isdir(constants.CONVERSATIONS_DIR): logger.warning(f"Conversations directory not found: {constants.CONVERSATIONS_DIR}"); return []
            filenames = [f for f in os.listdir(constants.CONVERSATIONS_DIR) if os.path.isfile(os.path.join(constants.CONVERSATIONS_DIR, f)) and f.lower().endswith(".json")]
            full_paths = [os.path.join(constants.CONVERSATIONS_DIR, f) for f in filenames]
            try: full_paths.sort(key=os.path.getmtime, reverse=True)
            except Exception as sort_e: logger.warning(f"Could not sort by mtime: {sort_e}"); full_paths.sort(key=os.path.basename, reverse=True)
            logger.info(f"Found {len(full_paths)} conversation files.")
        except OSError as e: logger.error(f"Error listing conversations in {constants.CONVERSATIONS_DIR}: {e}")
        return full_paths

    def load_session(self, filepath: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Loads session data from a specific file path.
        Returns model_name, personality, and project_context_data dictionary.
        """
        if not filepath or not isinstance(filepath, str) or not os.path.isabs(filepath): logger.error(f"Invalid path for loading: {filepath}"); return None, None, None
        if not filepath.lower().endswith(".json"): logger.error(f"Not a .json file: {filepath}"); return None, None, None
        if not os.path.exists(filepath): logger.error(f"File not found: {filepath}"); return None, None, None
        if not self._is_path_safe(filepath): logger.error(f"Attempt to load outside conversations dir: {filepath}"); return None, None, None
        logger.info(f"Attempting to load conversation from: {filepath}")
        return self._load_from_file(filepath)

    def save_session(self,
                     filepath: str,
                     model_name: Optional[str],
                     personality: Optional[str],
                     project_context_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]: # <<< CHANGE: Accept project_context_data dict >>>
        """Saves session data to a specific conversation file path."""
        if not filepath or not isinstance(filepath, str) or not os.path.isabs(filepath): logger.error(f"Invalid path for saving: {filepath}"); return False, None
        if not filepath.lower().endswith(".json"): filepath += ".json"
        if not self._is_path_safe(filepath): logger.error(f"Attempt to save outside conversations dir: {filepath}"); return False, None
        logger.info(f"Saving conversation to: {filepath}")
        if not isinstance(project_context_data, dict):
            logger.error(f"Project context data for saving session is not a dict ({type(project_context_data)}). Aborting save.")
            return False, None

        data_to_save = {
            "model_name": model_name,
            "personality_prompt": personality,
            # <<< CHANGE: Store the entire project_context_data dictionary >>>
            "project_context_data": project_context_data,
            "metadata": {
                "save_timestamp": datetime.datetime.now().isoformat(),
                "source": "named_conversation",
                "saved_filename": os.path.basename(filepath)
            }
        }
        success = self._save_to_file(filepath, data_to_save)
        return success, filepath if success else None

    def delete_session(self, filepath: str) -> bool: # No change
        if not filepath or not os.path.isabs(filepath): logger.error(f"Invalid path for deletion: {filepath}"); return False
        if not self._is_path_safe(filepath): logger.error(f"Attempt to delete outside conversations dir: {filepath}"); return False
        logger.info(f"Attempting to delete conversation file: {filepath}")
        if not os.path.exists(filepath): logger.error("File not found for deletion."); return False
        try:
            os.remove(filepath); logger.info("Conversation file deleted successfully."); return True
        except OSError as e: logger.error(f"Error deleting file {filepath}: {e}"); return False
        except Exception as e: logger.exception(f"Unexpected error deleting file {filepath}: {e}"); return False

    @staticmethod
    def sanitize_filename(filename: str) -> str: # No change
        if not filename or not isinstance(filename, str): return ""
        name = filename.strip();
        if not name: return ""
        base, ext = os.path.splitext(name)
        if not ext: name += ".json"
        elif ext.lower() != ".json": name = base + ".json"
        invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
        sanitized = re.sub(invalid_chars, '_', name).strip('_')
        if not sanitized or sanitized in ['.', '..'] or sanitized.upper() in ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'LPT1', 'COM2', 'LPT2', 'COM3', 'LPT3', 'COM4', 'LPT4']:
             return f"session_invalid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        max_len = 200
        if len(sanitized) > max_len: sanitized = base[:max_len - len(ext)] + ext
        return sanitized

    def _is_path_safe(self, filepath: str) -> bool: # No change
        try:
            safe_dir = os.path.abspath(constants.CONVERSATIONS_DIR)
            target_file = os.path.abspath(filepath)
            return os.path.commonpath([safe_dir]) == os.path.commonpath([safe_dir, target_file])
        except Exception as e: logger.error(f"Error during path safety check for '{filepath}': {e}"); return False