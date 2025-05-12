# utils/constants.py
# UPDATED: Added GLOBAL_CONTEXT_DISPLAY_NAME and user's preferred Gemini models

import os
import sys
import logging

logger = logging.getLogger(__name__)

# --- Core Application Settings ---
APP_NAME = "SynapseChat"
APP_VERSION = "3.4-FAISS-Multi" # Or whatever your current version is

# --- API & Model Configuration ---
# User's preferred preview models
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-pro-preview-05-06"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-2.5-pro-preview-05-06"
DEFAULT_OLLAMA_MODEL = "codellama:13b" # Default for generator or fallback Ollama chat

# --- UI Appearance ---
CHAT_FONT_FAMILY = "SansSerif"
CHAT_FONT_SIZE = 12
MAX_CHAT_AREA_WIDTH = 900
LOADING_GIF_FILENAME = "loading.gif"

# --- File Paths & Storage ---
if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(sys.executable)
    logger.info(f"[Constants] Running frozen. APP_BASE_DIR: {APP_BASE_DIR}")
else:
    APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"[Constants] Running as script. APP_BASE_DIR: {APP_BASE_DIR}")

USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".synapse_chat_data")
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
    '.pdf', '.docx'
}
DEFAULT_IGNORED_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', '.env', 'env',
    'node_modules', 'build', 'dist', '.vscode', '.idea', '.pytest_cache',
    'site-packages', '.mypy_cache', 'lib', 'include', 'bin', 'Scripts', '.dist',
    '.history', '.vscode-test', '.idea_modules', '*.egg-info', '*.tox', '.nox'
}

# --- RAG Specific Configuration ---
RAG_DB_PATH_NAME = "faiss_db"
RAG_COLLECTIONS_PATH = os.path.join(USER_DATA_DIR, RAG_DB_PATH_NAME)
GLOBAL_COLLECTION_ID = "global_collection"
GLOBAL_CONTEXT_DISPLAY_NAME = "Knowledge Database" # <-- DEFINED HERE
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 150
RAG_NUM_RESULTS = 15
RAG_MAX_FILE_SIZE_MB = 50

# --- Logging Configuration --- #INFO OR DEBUG
LOG_LEVEL = "DEBUG"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'