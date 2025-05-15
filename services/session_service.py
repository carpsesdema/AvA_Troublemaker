# services/session_service.py
import os
import json
import re
import datetime
import logging
from typing import Dict, Any, Optional, Tuple, List

from utils import constants
from core.models import ChatMessage # For deserializing ChatMessage objects
# Import MessageLoadingState for deserialization if it was saved.
# from core.message_enums import MessageLoadingState (If you decide to save/load it)

logger = logging.getLogger(__name__)


class SessionService:
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

    def _load_from_file(self, filepath: str) -> Tuple[
        Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        logger.debug(f"  Internal load: Reading file {filepath}")
        model_name, personality_prompt = None, None
        project_context_data: Optional[Dict[str, Any]] = None
        session_extra_data_loaded: Dict[str, Any] = {} # Initialize as dict

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()
            if not file_content.strip():
                logger.warning(f"Session file is empty: {filepath}");
                return None, None, None, None

            data = json.loads(file_content)
            if not isinstance(data, dict):
                logger.error(f"Invalid format: Session data not dict in {filepath}");
                return None, None, None, None

            model_name = data.get("model_name")
            personality_prompt = data.get("personality_prompt")

            # Load main project_context_data (same as before)
            if "project_context_data" in data and isinstance(data["project_context_data"], dict):
                loaded_pcd = data["project_context_data"]
                # ... (exact same ChatMessage deserialization logic as before)
                if isinstance(loaded_pcd.get("project_histories"), dict) and \
                        isinstance(loaded_pcd.get("project_names"), dict):
                    project_context_data = {
                        "project_histories": {},
                        "project_names": loaded_pcd.get("project_names", {}),
                        "current_project_id": loaded_pcd.get("current_project_id")
                    }
                    raw_histories = loaded_pcd.get("project_histories", {})
                    for pid, history_list_raw in raw_histories.items():
                        if isinstance(history_list_raw, list):
                            deserialized_history = []
                            for item_dict in history_list_raw:
                                try:
                                    role = item_dict.get('role')
                                    parts_raw = item_dict.get('parts')
                                    timestamp = item_dict.get('timestamp')
                                    metadata = item_dict.get('metadata')
                                    msg_id = item_dict.get('id')
                                    # loading_state_str = item_dict.get('loading_state')
                                    # loading_state_enum = MessageLoadingState[loading_state_str] if loading_state_str and hasattr(MessageLoadingState, loading_state_str) else MessageLoadingState.IDLE

                                    if role is None or parts_raw is None: continue
                                    if isinstance(parts_raw, str): parts_list = [parts_raw]
                                    elif isinstance(parts_raw, list): parts_list = [p for p in parts_raw if isinstance(p, (str, dict))]
                                    else: parts_list = []
                                    deserialized_history.append(
                                        ChatMessage(role=str(role), parts=parts_list, timestamp=timestamp,
                                                    metadata=metadata, id=msg_id)) #, loading_state=loading_state_enum))
                                except Exception as e_msg:
                                    logger.warning(f"Error deserializing ChatMessage item in project '{pid}' from {filepath}: {e_msg}. Skipping item.")
                            project_context_data["project_histories"][pid] = deserialized_history
                        else:
                            project_context_data["project_histories"][pid] = []
                else:
                    project_context_data = None # Indicates old format or error
            # ... (end of ChatMessage deserialization)

            # Load session_extra_data (includes new generator_model_name)
            # These are assumed to be top-level keys in the JSON file
            active_chat_backend_id_val = data.get("active_chat_backend_id")
            if active_chat_backend_id_val is not None:
                session_extra_data_loaded["active_chat_backend_id"] = active_chat_backend_id_val

            chat_temperature_val = data.get("chat_temperature")
            if chat_temperature_val is not None:
                session_extra_data_loaded["chat_temperature"] = chat_temperature_val

            generator_model_name_val = data.get("generator_model_name") # <-- LOAD NEW
            if generator_model_name_val is not None:
                session_extra_data_loaded["generator_model_name"] = generator_model_name_val
            # Add other fields from session_extra_data here if they were saved at top level

            # Backward Compatibility for project_context_data (same as before)
            if project_context_data is None:
                logger.info(f"Attempting to load old session format from {filepath} for project_context_data.")
                # ... (exact same backward compatibility logic for history as before)
                raw_project_histories = data.get("project_histories")
                raw_history = data.get("history")
                last_active_project_id_old = data.get("project_id")
                temp_project_histories = {}
                temp_project_names = {}
                current_project_id_to_use = None

                if isinstance(raw_project_histories, dict):
                    for pid, history_list_raw in raw_project_histories.items():
                        deserialized_history = []
                        if isinstance(history_list_raw, list):
                            for item_dict in history_list_raw:
                                try:
                                    role = item_dict.get('role'); parts_raw = item_dict.get('parts'); timestamp = item_dict.get('timestamp'); metadata = item_dict.get('metadata'); msg_id = item_dict.get('id')
                                    if role is None or parts_raw is None: continue
                                    if isinstance(parts_raw, str): parts_list = [parts_raw]
                                    elif isinstance(parts_raw, list): parts_list = [p for p in parts_raw if isinstance(p, (str, dict))]
                                    else: parts_list = []
                                    deserialized_history.append(ChatMessage(role=str(role), parts=parts_list, timestamp=timestamp, metadata=metadata, id=msg_id))
                                except Exception: pass
                        temp_project_histories[pid] = deserialized_history
                        temp_project_names[pid] = pid
                    current_project_id_to_use = last_active_project_id_old
                elif isinstance(raw_history, list):
                    deserialized_history = []
                    for item_dict in raw_history:
                        try:
                            role = item_dict.get('role'); parts_raw = item_dict.get('parts'); timestamp = item_dict.get('timestamp'); metadata = item_dict.get('metadata'); msg_id = item_dict.get('id')
                            if role is None or parts_raw is None: continue
                            if isinstance(parts_raw, str): parts_list = [parts_raw]
                            elif isinstance(parts_raw, list): parts_list = [p for p in parts_raw if isinstance(p, (str, dict))]
                            else: parts_list = []
                            deserialized_history.append(ChatMessage(role=str(role), parts=parts_list, timestamp=timestamp, metadata=metadata, id=msg_id))
                        except Exception: pass
                    pid_for_old_history = last_active_project_id_old or constants.GLOBAL_COLLECTION_ID
                    temp_project_histories[pid_for_old_history] = deserialized_history
                    temp_project_names[pid_for_old_history] = pid_for_old_history
                    current_project_id_to_use = pid_for_old_history

                if temp_project_histories:
                    if constants.GLOBAL_COLLECTION_ID not in temp_project_histories:
                        temp_project_histories[constants.GLOBAL_COLLECTION_ID] = []
                        temp_project_names[constants.GLOBAL_COLLECTION_ID] = constants.GLOBAL_CONTEXT_DISPLAY_NAME
                    project_context_data = {
                        "project_histories": temp_project_histories,
                        "project_names": temp_project_names,
                        "current_project_id": current_project_id_to_use
                    }
                    logger.info(f"Successfully converted old session format from {filepath} to PCD.")
                else:
                    logger.warning(f"Could not parse any history from old format in {filepath}.")
            # ... (end of backward compatibility for history)

            if project_context_data is None: # Ensure it's always initialized
                project_context_data = {
                    "project_histories": {constants.GLOBAL_COLLECTION_ID: []},
                    "project_names": {constants.GLOBAL_COLLECTION_ID: constants.GLOBAL_CONTEXT_DISPLAY_NAME},
                    "current_project_id": constants.GLOBAL_COLLECTION_ID
                }
            loaded_projects_count = len(project_context_data.get("project_histories", {}))
            total_messages = sum(len(h) for h in project_context_data.get("project_histories", {}).values())
            logger.info(
                f"Session loaded from {os.path.basename(filepath)}. Model: {model_name}, Pers: {'Set' if personality_prompt else 'None'}, ActiveProj: {project_context_data.get('current_project_id')}, Projects: {loaded_projects_count}, TotalMsgs: {total_messages}, ExtraData: {session_extra_data_loaded if session_extra_data_loaded else 'None'}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}"); return None, None, None, None
        except OSError as e:
            logger.error(f"OS error reading {filepath}: {e}"); return None, None, None, None
        except Exception as e:
            logger.exception(f"Unexpected error loading {filepath}: {e}"); return None, None, None, None

        # Return None for session_extra_data if it's empty after loading
        final_session_extra_data = session_extra_data_loaded if session_extra_data_loaded else None
        return model_name, personality_prompt, project_context_data, final_session_extra_data

    def _chatmessage_to_dict(self, msg: ChatMessage) -> Dict[str, Any]:
        serializable_metadata = None
        if isinstance(msg.metadata, dict):
            serializable_metadata = {k: v for k, v in msg.metadata.items() if
                                     isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        # loading_state_str = msg.loading_state.name if hasattr(msg, 'loading_state') and msg.loading_state else None # MessageLoadingState.IDLE.name
        return {"role": msg.role, "parts": msg.parts, "timestamp": msg.timestamp, "metadata": serializable_metadata, "id": msg.id} # , "loading_state": loading_state_str}

    def _save_to_file(self, filepath: str, data_to_save: Dict[str, Any]) -> bool:
        logger.debug(f"  Internal save: Writing to file {filepath}")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Serialize ChatMessage objects within project_context_data
            if "project_context_data" in data_to_save and isinstance(data_to_save["project_context_data"], dict):
                pcd = data_to_save["project_context_data"]
                if "project_histories" in pcd and isinstance(pcd["project_histories"], dict):
                    serializable_histories = {}
                    for pid, history_list_obj in pcd["project_histories"].items():
                        if isinstance(history_list_obj, list):
                            serializable_histories[pid] = [self._chatmessage_to_dict(msg) for msg in history_list_obj if
                                                           isinstance(msg, ChatMessage)]
                        else:
                            serializable_histories[pid] = [] # Should not happen if types are correct
                    pcd["project_histories"] = serializable_histories
            # data_to_save now contains model_name, personality_prompt, project_context_data (with serialized histories),
            # and any top-level keys from session_extra_data (like active_chat_backend_id, chat_temperature, generator_model_name)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Session data saved to {os.path.basename(filepath)}.")
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.exception(f"Error saving session file {filepath}: {e}")
            return False

    def get_last_session(self) -> Tuple[
        Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        logger.info(f"Attempting to load last session state from: {constants.LAST_SESSION_FILEPATH}")
        if not os.path.exists(constants.LAST_SESSION_FILEPATH):
            logger.info("Last session file not found. Starting fresh.")
            return None, None, None, None
        return self._load_from_file(constants.LAST_SESSION_FILEPATH)

    def save_last_session(self,
                          model_name: Optional[str],
                          personality: Optional[str],
                          project_context_data: Dict[str, Any],
                          session_extra_data: Optional[Dict[str, Any]] = None):
        logger.info(f"Attempting to save last session state to: {constants.LAST_SESSION_FILEPATH}")
        if not isinstance(project_context_data, dict):
            logger.error(f"PCD for saving last session is not a dict ({type(project_context_data)}). Aborting save.")
            return False

        data_to_save: Dict[str, Any] = {
            "model_name": model_name, # Chat model name
            "personality_prompt": personality, # Chat model personality
            "project_context_data": project_context_data, # Project histories, names, active PID
            "metadata": { # General session metadata
                "save_timestamp": datetime.datetime.now().isoformat(),
                "source": "last_session"
            }
        }
        # Merge session_extra_data (active_chat_backend_id, chat_temperature, generator_model_name)
        # These will be saved as top-level keys in the JSON.
        if session_extra_data and isinstance(session_extra_data, dict):
            data_to_save.update(session_extra_data)
            logger.debug(f"Merged session_extra_data into save data for last session: {list(session_extra_data.keys())}")

        return self._save_to_file(constants.LAST_SESSION_FILEPATH, data_to_save)

    # list_sessions, load_session, save_session, delete_session, sanitize_filename, _is_path_safe
    # remain largely the same but the calls to _load_from_file and _save_to_file
    # will now handle the session_extra_data correctly.

    def clear_last_session_file(self) -> bool:
        logger.info(f"Attempting to clear last session state file: {constants.LAST_SESSION_FILEPATH}")
        try:
            if os.path.exists(constants.LAST_SESSION_FILEPATH):
                os.remove(constants.LAST_SESSION_FILEPATH); logger.info("Last session state file deleted.")
            else:
                logger.info("Last session state file did not exist.")
            return True
        except OSError as e:
            logger.error(f"Error deleting last session file {constants.LAST_SESSION_FILEPATH}: {e}"); return False
        except Exception as e:
            logger.exception(f"Unexpected error clearing last session file: {e}"); return False

    def list_sessions(self) -> List[str]:
        logger.info(f"Listing conversations in: {constants.CONVERSATIONS_DIR}")
        full_paths = []
        try:
            if not os.path.isdir(constants.CONVERSATIONS_DIR): logger.warning(
                f"Conversations directory not found: {constants.CONVERSATIONS_DIR}"); return []
            filenames = [f for f in os.listdir(constants.CONVERSATIONS_DIR) if
                         os.path.isfile(os.path.join(constants.CONVERSATIONS_DIR, f)) and f.lower().endswith(".json")]
            full_paths = [os.path.join(constants.CONVERSATIONS_DIR, f) for f in filenames]
            try:
                full_paths.sort(key=os.path.getmtime, reverse=True)
            except Exception as sort_e:
                logger.warning(f"Could not sort by mtime: {sort_e}"); full_paths.sort(key=os.path.basename,
                                                                                      reverse=True)
            logger.info(f"Found {len(full_paths)} conversation files.")
        except OSError as e:
            logger.error(f"Error listing conversations in {constants.CONVERSATIONS_DIR}: {e}")
        return full_paths

    def load_session(self, filepath: str) -> Tuple[
        Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not filepath or not isinstance(filepath, str) or not os.path.isabs(filepath): logger.error(
            f"Invalid path for loading: {filepath}"); return None, None, None, None
        if not filepath.lower().endswith(".json"): logger.error(
            f"Not a .json file: {filepath}"); return None, None, None, None
        if not os.path.exists(filepath): logger.error(f"File not found: {filepath}"); return None, None, None, None
        if not self._is_path_safe(filepath): logger.error(
            f"Attempt to load outside conversations dir: {filepath}"); return None, None, None, None
        logger.info(f"Attempting to load conversation from: {filepath}")
        return self._load_from_file(filepath) # This now returns session_extra_data

    def save_session(self,
                     filepath: str,
                     model_name: Optional[str], # Chat model name
                     personality: Optional[str], # Chat model personality
                     project_context_data: Dict[str, Any],
                     session_extra_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        if not filepath or not isinstance(filepath, str) or not os.path.isabs(filepath): logger.error(
            f"Invalid path for saving: {filepath}"); return False, None
        if not filepath.lower().endswith(".json"): filepath += ".json"
        if not self._is_path_safe(filepath): logger.error(
            f"Attempt to save outside conversations dir: {filepath}"); return False, None
        logger.info(f"Saving conversation to: {filepath}")
        if not isinstance(project_context_data, dict):
            logger.error(f"PCD for saving session is not a dict ({type(project_context_data)}). Aborting save.")
            return False, None

        data_to_save: Dict[str, Any] = {
            "model_name": model_name,
            "personality_prompt": personality,
            "project_context_data": project_context_data,
            "metadata": {
                "save_timestamp": datetime.datetime.now().isoformat(),
                "source": "named_conversation",
                "saved_filename": os.path.basename(filepath)
            }
        }
        if session_extra_data and isinstance(session_extra_data, dict):
            data_to_save.update(session_extra_data) # Merge as top-level keys
            logger.debug(f"Merged session_extra_data into named save data: {list(session_extra_data.keys())}")

        success = self._save_to_file(filepath, data_to_save)
        return success, filepath if success else None

    def delete_session(self, filepath: str) -> bool:
        if not filepath or not os.path.isabs(filepath): logger.error(
            f"Invalid path for deletion: {filepath}"); return False
        if not self._is_path_safe(filepath): logger.error(
            f"Attempt to delete outside conversations dir: {filepath}"); return False
        logger.info(f"Attempting to delete conversation file: {filepath}")
        if not os.path.exists(filepath): logger.error("File not found for deletion."); return False
        try:
            os.remove(filepath);
            logger.info("Conversation file deleted successfully.");
            return True
        except OSError as e:
            logger.error(f"Error deleting file {filepath}: {e}"); return False
        except Exception as e:
            logger.exception(f"Unexpected error deleting file {filepath}: {e}"); return False

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        if not filename or not isinstance(filename, str): return ""
        name = filename.strip();
        if not name: return ""
        base, ext = os.path.splitext(name)
        if not ext:
            name += ".json"
        elif ext.lower() != ".json":
            name = base + ".json"
        invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
        sanitized = re.sub(invalid_chars, '_', name).strip('_')
        if not sanitized or sanitized in ['.', '..'] or sanitized.upper() in ['CON', 'PRN', 'AUX', 'NUL', 'COM1',
                                                                              'LPT1', 'COM2', 'LPT2', 'COM3', 'LPT3',
                                                                              'COM4', 'LPT4']:
            return f"session_invalid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        max_len = 200
        if len(sanitized) > max_len: sanitized = base[:max_len - len(ext)] + ext
        return sanitized

    def _is_path_safe(self, filepath: str) -> bool:
        try:
            safe_dir = os.path.abspath(constants.CONVERSATIONS_DIR)
            target_file = os.path.abspath(filepath)
            return os.path.commonpath([safe_dir]) == os.path.commonpath([safe_dir, target_file])
        except Exception as e:
            logger.error(f"Error during path safety check for '{filepath}': {e}"); return False