# main.py
import sys
import os
import traceback
import logging
import asyncio

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase # Keep for potential font loading in future

try:
    import qasync
except ImportError:
    print("[CRITICAL] qasync library not found. Please install it: pip install qasync", file=sys.stderr)
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Missing Dependency",
                             "Required library 'qasync' is not installed.\nPlease run: pip install qasync")
    except Exception as e:
        print(f"Failed to show missing dependency message: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from ui.main_window import MainWindow
    from core.chat_manager import ChatManager
    # --- NEW: Import ApplicationOrchestrator ---
    from core.application_orchestrator import ApplicationOrchestrator
    # --- END NEW ---
    from services.session_service import SessionService
    from services.upload_service import UploadService
    # Backend adapters are no longer directly instantiated here by ChatManager or main.
    from utils.constants import CHAT_FONT_FAMILY, LOG_LEVEL, LOG_FORMAT, APP_VERSION, APP_NAME
except ImportError as e:
    print(f"[CRITICAL] Failed to import core components in main.py: {e}", file=sys.stderr)
    print(f"PYTHONPATH: {sys.path}", file=sys.stderr)
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Import Error", f"Failed to import core components:\n{e}\nCheck PYTHONPATH.")
    except Exception as e_qm:
        print(f"Failed to show import error message: {e_qm}", file=sys.stderr)
    sys.exit(1)

# Configure logging
log_level_actual = getattr(logging, LOG_LEVEL.upper(), logging.INFO) # Renamed to avoid conflict
logging.basicConfig(level=log_level_actual, format=LOG_FORMAT, handlers=[logging.StreamHandler()], force=True)
logger = logging.getLogger(__name__) # Main application logger


async def async_main():
    logger.info(f"--- Starting {APP_NAME} v{APP_VERSION} (Async with Orchestrator) ---")

    app = QApplication.instance()
    if app is None:
        if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Application base path: {application_path}")

    logger.info("--- Font Setup: Relying on system fonts (Using CHAT_FONT_FAMILY constant) ---")
    # If specific font loading was needed, it would go here, e.g.:
    # QFontDatabase.addApplicationFont(os.path.join(constants.ASSETS_PATH, "YourFont.ttf"))

    app.setStyle("Fusion") # Or your preferred style
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    logger.info("--- Instantiating Application Components ---")
    main_window = None
    try:
        # --- MODIFIED: Instantiate Orchestrator first ---
        session_service = SessionService()
        upload_service = UploadService() # VectorDBService is initialized within UploadService

        # Create the ApplicationOrchestrator, which handles creation of other core components
        app_orchestrator = ApplicationOrchestrator(
            session_service=session_service,
            upload_service=upload_service
        )
        logger.info("ApplicationOrchestrator instantiated.")

        # ChatManager now receives the orchestrator
        chat_manager = ChatManager(
            orchestrator=app_orchestrator
            # No longer directly passes session_service or upload_service
        )
        logger.info("ChatManager instantiated with orchestrator.")
        # --- END MODIFIED ---

        main_window = MainWindow(chat_manager=chat_manager, app_base_path=application_path)
        logger.info("--- Core Components Instantiated ---")
    except Exception as e:
        logger.exception(" ***** FATAL ERROR DURING COMPONENT INSTANTIATION ***** ")
        try:
            QMessageBox.critical(None, "Fatal Init Error", f"Failed during component setup:\n{e}\n\nCheck logs.")
        except Exception:
            # This ensures the error is printed if QMessageBox fails (e.g., if app hasn't started fully)
            print(f"[CRITICAL] Component Init Failed: {e}\nTraceback:\n{traceback.format_exc()}", file=sys.stderr)
        await app.quit() # type: ignore
        return 1

    # Initialize ChatManager after UI is created
    # ChatManager's initialize() will configure the backends using the components from the orchestrator
    QTimer.singleShot(100, chat_manager.initialize)
    logger.info("Scheduled ChatManager late initialization.")

    if main_window:
        main_window.setGeometry(100, 100, 1100, 850) # Or your preferred default size/pos
        main_window.show()
        logger.info("--- Main Window Shown ---")
    else:
        logger.error("MainWindow instance not created, cannot show window.")
        await app.quit() # type: ignore
        return 1

    logger.info("--- Starting Application Event Loop (via qasync) ---")
    await asyncio.Future() # This keeps the event loop running until quit
    logger.info(f"--- Application Event Loop Finished ---")
    return 0


if __name__ == "__main__":
    try:
        # Ensure qasync event loop is set for asyncio
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Get or create QApplication instance BEFORE creating QEventLoop
        q_app_instance = QApplication.instance() or QApplication(sys.argv)

        event_loop = qasync.QEventLoop(q_app_instance)
        asyncio.set_event_loop(event_loop)

        with event_loop:
            exit_code = event_loop.run_until_complete(async_main())
        sys.exit(exit_code)

    except RuntimeError as e:
        if "cannot be nested" in str(e) or "already running" in str(e):
            logger.warning(
                f"qasync event loop issue: {e}. This might happen during re-runs in some IDEs. Attempting to proceed if app is already running.")
            if QApplication.instance() and QApplication.instance().property("activeWindow"):
                pass
            else:
                sys.exit(1)
        else:
            logger.critical(f"RuntimeError during qasync execution: {e}", exc_info=True)
            try:
                _dummy_app = QApplication.instance() or QApplication(sys.argv) # Ensure app for dialog
                QMessageBox.critical(None, "Runtime Error", f"Application failed to run:\n{e}\n\nCheck logs.")
            except Exception: pass # Ignore if dialog fails
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception during application startup/run: {e}", exc_info=True)
        try:
            _dummy_app = QApplication.instance() or QApplication(sys.argv) # Ensure app for dialog
            QMessageBox.critical(None, "Unhandled Exception", f"An unexpected error occurred:\n{e}\n\nCheck logs.")
        except Exception: pass # Ignore if dialog fails
        sys.exit(1)