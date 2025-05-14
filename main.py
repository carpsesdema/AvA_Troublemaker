# main.py
import sys
import os
import traceback
import logging
import asyncio
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QStyle
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase, QIcon # QKeyEvent, QKeySequence (if you re-add shortcuts)

try:
    import qasync
except ImportError:
    print("[CRITICAL] qasync library not found. Please install it: pip install qasync", file=sys.stderr)
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv);
        QMessageBox.critical(None, "Missing Dependency",
                             "Required library 'qasync' is not installed.\nPlease run: pip install qasync")
    except Exception as e:
        print(f"Failed to show missing dependency message: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from ui.main_window import MainWindow
    from core.chat_manager import ChatManager
    from core.application_orchestrator import ApplicationOrchestrator
    from services.session_service import SessionService
    from services.upload_service import UploadService
    from core.chat_message_state_handler import ChatMessageStateHandler
    from utils import constants
    from utils.constants import CHAT_FONT_FAMILY, LOG_LEVEL, LOG_FORMAT, APP_VERSION, APP_NAME, \
        ASSETS_PATH
except ImportError as e:
    print(f"[CRITICAL] Failed to import core components in main.py: {e}", file=sys.stderr)
    print(f"PYTHONPATH: {sys.path}", file=sys.stderr)
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv);
        QMessageBox.critical(None, "Import Error",
                             f"Failed to import core components:\n{e}\nCheck PYTHONPATH.")
    except Exception as e_qm:
        print(f"Failed to show import error message: {e_qm}", file=sys.stderr)
    sys.exit(1)

log_level_actual = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level_actual, format=LOG_FORMAT, handlers=[logging.StreamHandler()], force=True)
logger = logging.getLogger(__name__)


async def async_main():
    logger.info(f"--- Starting {APP_NAME} v{APP_VERSION} (Async with Orchestrator & StateHandler) ---")

    app = QApplication.instance()
    if app is None:
        if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Application base path: {application_path}")
    logger.info("--- Font Setup: Relying on system fonts ---")
    app.setStyle("Fusion");
    app.setApplicationName(APP_NAME);
    app.setApplicationVersion(APP_VERSION)

    try:
        app_icon_path = os.path.join(ASSETS_PATH, "Synchat.ico")
        std_fallback_icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        app_icon = QIcon(app_icon_path) if os.path.exists(app_icon_path) else std_fallback_icon
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
        elif not std_fallback_icon.isNull():
            app.setWindowIcon(std_fallback_icon)
        else:
            logger.warning("Could not load custom or standard fallback icon.")
    except Exception as e:
        logger.error(f"Error setting application icon: {e}", exc_info=True)

    logger.info("--- Instantiating Application Components ---")
    main_window: Optional[MainWindow] = None
    chat_message_state_handler: Optional[ChatMessageStateHandler] = None
    chat_manager: Optional[ChatManager] = None # Ensure chat_manager is defined in this scope

    try:
        session_service = SessionService()
        upload_service = UploadService()
        app_orchestrator = ApplicationOrchestrator(session_service=session_service, upload_service=upload_service)
        logger.info("ApplicationOrchestrator instantiated.")
        chat_manager = ChatManager(orchestrator=app_orchestrator)
        logger.info("ChatManager instantiated with orchestrator.")

        main_window = MainWindow(chat_manager=chat_manager, app_base_path=application_path)
        logger.info("MainWindow instantiated.")

        if main_window and main_window.chat_tab_manager:
            active_chat_tab = main_window.chat_tab_manager.get_active_chat_tab_instance()
            chat_list_model_instance = None
            if active_chat_tab:
                chat_display_area = active_chat_tab.get_chat_display_area()
                if chat_display_area:
                    chat_list_model_instance = chat_display_area.get_model()
                    if chat_display_area.chat_item_delegate:
                        # Pass the view to the delegate for QMovie updates
                        chat_display_area.chat_item_delegate.setView(chat_display_area.chat_list_view)
                        logger.info("Set view reference for initial active tab's delegate.")


            if chat_list_model_instance:
                backend_coordinator_instance = chat_manager.get_backend_coordinator()
                if backend_coordinator_instance:
                    chat_message_state_handler = ChatMessageStateHandler(
                        model=chat_list_model_instance,
                        backend_coordinator=backend_coordinator_instance,
                        parent=app # Parent to app for lifecycle management
                    )
                    logger.info("ChatMessageStateHandler instantiated and wired to initial active model.")
                else:
                    logger.error("Critical: BackendCoordinator not available. ChatMessageStateHandler NOT created.")
            else:
                logger.warning(
                    "ChatMessageStateHandler NOT created: Could not get a ChatListModel instance from initial active tab (this is normal if no tabs are open on startup).")
        logger.info("--- Core Components Instantiated ---")
    except Exception as e:
        logger.exception(" ***** FATAL ERROR DURING COMPONENT INSTANTIATION ***** ")
        try:
            QMessageBox.critical(None, "Fatal Init Error", f"Failed during component setup:\n{e}\n\nCheck logs.")
        except Exception:
            print(f"[CRITICAL] Component Init Failed: {e}\nTraceback:\n{traceback.format_exc()}", file=sys.stderr)
        if app:
            # Use synchronous quit here as await might not be safe if loop isn't fully up
            app.quit()
        return 1

    if chat_manager:
        QTimer.singleShot(100, chat_manager.initialize)
        logger.info("Scheduled ChatManager late initialization.")
    else:
        logger.critical("ChatManager failed to instantiate. Application cannot continue.")
        if app: app.quit()
        return 1

    if main_window:
        main_window.setGeometry(100, 100, 1100, 850)
        main_window.show()
        logger.info("--- Main Window Shown ---")
    else:
        logger.error("MainWindow instance not created, cannot show window.")
        if app: app.quit()
        return 1

    logger.info("--- async_main: Entering main blocking phase (await asyncio.Future()) ---")
    if app:
        # This future is what keeps async_main alive.
        # qasync's event loop will run, processing Qt events.
        # When the Qt app quits (e.g., last window closed), qasync's loop should stop,
        # which in turn should cause this Future to complete or be cancelled.
        await asyncio.Future()
        logger.info("--- async_main: asyncio.Future() completed. Application is shutting down. ---")
    else:
        # This case should ideally not be reached if app initialization was successful
        logger.error("--- async_main: app instance is None. Cannot block. Application will likely exit. ---")

    logger.info(f"--- async_main returning, Application Event Loop should be finishing ---")
    return 0


if __name__ == "__main__":
    q_app_instance = QApplication.instance()
    if q_app_instance is None:
        if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        q_app_instance = QApplication(sys.argv)

    event_loop: Optional[qasync.QEventLoop] = None
    exit_code = 1  # Default to error

    try:
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        event_loop = qasync.QEventLoop(q_app_instance)
        asyncio.set_event_loop(event_loop)

        with event_loop:
            exit_code = event_loop.run_until_complete(async_main())

    except RuntimeError as e:
        if "cannot be nested" in str(e).lower() or "already running" in str(e).lower():
            logger.warning(f"qasync event loop issue: {e}. Loop may already be running (e.g. interactive environment).")
            if QApplication.instance() and QApplication.instance().activeWindow():
                pass
            else:
                exit_code = 1
        else:
            logger.critical(f"RuntimeError during qasync execution: {e}", exc_info=True)
            try:
                QMessageBox.critical(None, "Runtime Error", f"Application failed to run:\n{e}\n\nCheck logs.")
            except Exception: pass
            exit_code = 1
    except Exception as e:
        logger.critical(f"Unhandled exception during application startup/run: {e}", exc_info=True)
        try:
            QMessageBox.critical(None, "Unhandled Exception", f"An unexpected error occurred:\n{e}\n\nCheck logs.")
        except Exception: pass
        exit_code = 1
    finally:
        logger.info(f"Application attempting to exit with code: {exit_code}")
        if event_loop and event_loop.is_running():
            logger.info("Event loop still running in finally block, attempting to close.")
            event_loop.close()
        sys.exit(exit_code)