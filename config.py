# SynChat/config.py
# UPDATED FILE
import os
import logging
import sys

import dotenv # Ensure dotenv is imported if load_dotenv is used
from typing import Optional

logger = logging.getLogger(__name__)

# Determine the base path (works for frozen and non-frozen)
if getattr(sys, 'frozen', False):
    # If frozen, the executable is usually in the root, or a subfolder.
    # Using dirname(sys.executable) is often correct.
    _BASE_DIR = os.path.dirname(sys.executable)
    logger.info(f"[config] Running frozen. _BASE_DIR: {_BASE_DIR}")
else:
    # If not frozen, assume config.py is in the project root directory (SynaChat).
    # Get the directory containing *this* file (config.py).
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"[config] Running as script. _BASE_DIR: {_BASE_DIR}")

# Construct the path to the .env file based on the corrected _BASE_DIR
_DOTENV_PATH = os.path.join(_BASE_DIR, '.env')
logger.info(f"[config] Calculated .env path: {_DOTENV_PATH}")


def load_config() -> dict:
    """Loads configuration from .env file."""
    config = {}
    # Use the calculated _DOTENV_PATH
    dotenv_path_to_load = _DOTENV_PATH

    if os.path.exists(dotenv_path_to_load):
        logger.info(f"Loading configuration from: {dotenv_path_to_load}")
        try:
            # Explicitly pass the calculated path to load_dotenv
            dotenv.load_dotenv(dotenv_path=dotenv_path_to_load)
            config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
            if not config['GEMINI_API_KEY']:
                logger.warning("GEMINI_API_KEY not found or empty in .env file.")
            else:
                logger.info("GEMINI_API_KEY loaded successfully from .env.")
        except Exception as e:
            logger.exception(f"Error loading .env file at {dotenv_path_to_load}: {e}")
            config['GEMINI_API_KEY'] = None # Ensure key is None on error
    else:
        logger.warning(f".env file not found at expected location: {dotenv_path_to_load}. Checking environment variables as fallback.")
        # Check environment variables as fallback if .env is missing
        config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
        if not config['GEMINI_API_KEY']:
            logger.warning("GEMINI_API_KEY not found in environment variables either.")
        else:
             logger.info("GEMINI_API_KEY loaded from environment variables.")


    # Add other configuration loading here if needed (e.g., from JSON/YAML)
    # config['DEFAULT_MODEL'] = ...

    return config

# Load config once on import
# This ensures load_config runs with the corrected paths when the module loads.
APP_CONFIG = load_config()

def get_api_key() -> Optional[str]:
    """Returns the loaded Gemini API key."""
    # The key is retrieved from the already loaded APP_CONFIG dictionary
    return APP_CONFIG.get("GEMINI_API_KEY")