# utils/constants.py

import os
import sys
import logging

logger = logging.getLogger(__name__)

# --- Core Application Settings ---
APP_NAME = "SynapseChat" # Or your new app name "AvA"
APP_VERSION = "3.4-FAISS-Multi" # Or current version

# --- API & Model Configuration ---
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-pro-preview-05-06" # Example, use your actual default
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-2.5-pro-preview-05-06" # Example
DEFAULT_OLLAMA_MODEL = "codellama:13b" # Example
DEFAULT_GPT_MODEL = "gpt-4-turbo-preview" # Example

# --- Backend IDs ---
DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
OLLAMA_CHAT_BACKEND_ID = "ollama_chat"
GPT_CHAT_BACKEND_ID = "gpt_chat"
PLANNER_BACKEND_ID = "gemini_planner"
GENERATOR_BACKEND_ID = "ollama_generator"

# --- UI Appearance ---
CHAT_FONT_FAMILY = "SansSerif" # Or your preferred font
CHAT_FONT_SIZE = 12 # Or your preferred size
MAX_CHAT_AREA_WIDTH = 900
LOADING_GIF_FILENAME = "loading.gif"

# --- File Paths & Storage ---
if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(sys.executable)
else:
    APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes constants.py is in utils/

USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".ava_desktop_data") # Changed to .ava_desktop_data
CONVERSATIONS_DIR_NAME = "conversations"
CONVERSATIONS_DIR = os.path.join(USER_DATA_DIR, CONVERSATIONS_DIR_NAME)
LAST_SESSION_FILENAME = ".last_session_state.json"
LAST_SESSION_FILEPATH = os.path.join(USER_DATA_DIR, LAST_SESSION_FILENAME)

ASSETS_DIR_NAME = "assets"
ASSETS_PATH = os.path.join(APP_BASE_DIR, ASSETS_DIR_NAME)

STYLESHEET_FILENAME = "style.qss"
BUBBLE_STYLESHEET_FILENAME = "bubble_style.qss"
UI_DIR_NAME = "ui"
UI_DIR_PATH = os.path.join(APP_BASE_DIR, UI_DIR_NAME)
STYLE_PATHS_TO_CHECK = [os.path.join(UI_DIR_PATH, STYLESHEET_FILENAME)]
BUBBLE_STYLESHEET_PATH = os.path.join(UI_DIR_PATH, BUBBLE_STYLESHEET_FILENAME)

# --- Upload Handling (General) ---
MAX_SCAN_DEPTH = 5
ALLOWED_TEXT_EXTENSIONS = {
    '.txt', '.py', '.md', '.json', '.js', '.html', '.css', '.c', '.cpp', '.h',
    '.java', '.cs', '.xml', '.yaml', '.yml', '.sh', '.bat', '.ps1', '.log',
    '.csv', '.tsv', '.ini', '.cfg', '.sql', '.rb', '.php', '.go', '.rs', '.swift',
    '.kt', '.kts', '.scala', '.lua', '.pl', '.pm', '.r', '.dart', '.tex', '.toml',
    '.pdf', '.docx' # Keep these as they are handled by FileHandlerService
}
DEFAULT_IGNORED_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', '.env', 'env',
    'node_modules', 'build', 'dist', '.vscode', '.idea', '.pytest_cache',
    'site-packages', '.mypy_cache', 'lib', 'include', 'bin', 'Scripts', '.dist',
    '.history', '.vscode-test', '.idea_modules', '*.egg-info', '*.tox', '.nox'
}

# --- RAG Specific Configuration ---
RAG_DB_PATH_NAME = "faiss_db_ava" # Changed to _ava
RAG_COLLECTIONS_PATH = os.path.join(USER_DATA_DIR, RAG_DB_PATH_NAME)
GLOBAL_COLLECTION_ID = "ava_global_collection" # Changed
GLOBAL_CONTEXT_DISPLAY_NAME = "AvA Global Knowledge" # Changed
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 150
RAG_NUM_RESULTS = 15 # Or your preferred number of RAG results
RAG_MAX_FILE_SIZE_MB = 50

# --- Logging Configuration ---
LOG_LEVEL = "INFO" # Or "INFO" for less verbose logs
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'

# --- REMOVED AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES ---
# AUTO_SUMMARY_TRIGGER_THRESHOLD_FILES = 5