# main.py
# UPDATED - Instantiates ChatMessageStateHandler and connects it.

import sys
import os
import traceback
import logging
import asyncio
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase, QIcon  # Added QIcon

try:
    import qasync
except ImportError:
    print("[CRITICAL] qasync library not found. Please install it: pip install qasync", file=sys.stderr)
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv); QMessageBox.critical(None, "Missing Dependency",
                                                                                             "Required library 'qasync' is not installed.\nPlease run: pip install qasync")  # type: ignore
    except Exception as e:
        print(f"Failed to show missing dependency message: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from ui.main_window import MainWindow
    from core.chat_manager import ChatManager
    from core.application_orchestrator import ApplicationOrchestrator
    from services.session_service import SessionService
    from services.upload_service import UploadService
    # --- NEW Import ---
    from core.chat_message_state_handler import ChatMessageStateHandler
    # --- END NEW ---
    from utils import constants  # Ensure constants is imported
    from utils.constants import CHAT_FONT_FAMILY, LOG_LEVEL, LOG_FORMAT, APP_VERSION, APP_NAME, \
        ASSETS_PATH  # Import ASSETS_PATH
except ImportError as e:
    print(f"[CRITICAL] Failed to import core components in main.py: {e}", file=sys.stderr)
    print(f"PYTHONPATH: {sys.path}", file=sys.stderr)
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv); QMessageBox.critical(None, "Import Error",
                                                                                             f"Failed to import core components:\n{e}\nCheck PYTHONPATH.")  # type: ignore
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
            Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)  # type: ignore
        if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)  # type: ignore
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

    # --- Setup Window Icon (moved earlier, before critical components if possible) ---
    try:
        app_icon_path = os.path.join(ASSETS_PATH, "Synchat.ico")  # Use constants.ASSETS_PATH
        std_fallback_icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)  # type: ignore
        app_icon = QIcon(app_icon_path) if os.path.exists(app_icon_path) else std_fallback_icon
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
        elif not std_fallback_icon.isNull():
            app.setWindowIcon(std_fallback_icon)
    except Exception as e:
        logger.error(f"Error setting application icon: {e}", exc_info=True)
    # --- End Icon Setup ---

    logger.info("--- Instantiating Application Components ---")
    main_window: Optional[MainWindow] = None
    chat_message_state_handler: Optional[ChatMessageStateHandler] = None

    try:
        session_service = SessionService()
        upload_service = UploadService()
        app_orchestrator = ApplicationOrchestrator(session_service=session_service, upload_service=upload_service)
        logger.info("ApplicationOrchestrator instantiated.")
        chat_manager = ChatManager(orchestrator=app_orchestrator)
        logger.info("ChatManager instantiated with orchestrator.")

        main_window = MainWindow(chat_manager=chat_manager, app_base_path=application_path)
        logger.info("MainWindow instantiated.")

        # --- Instantiate and wire ChatMessageStateHandler ---
        if main_window and main_window.chat_tab_manager:
            # The ChatListModel instances are created by ChatDisplayArea, which is part of ChatTabWidget.
            # ChatMessageStateHandler needs to be connected to *each* model if they are distinct per tab.
            # For now, let's assume a single primary ChatListModel instance for simplicity,
            # or that the handler will be re-initialized/re-connected when tabs change.
            # This part might need refinement based on how ChatListModel instances are managed with tabs.

            # A robust approach: ChatManager itself could provide access to the currently active model,
            # or ChatMessageStateHandler could listen to tab changes and get the model from the new active tab.

            # For now, this attempts to get the model of the *initially active tab* if one exists.
            # This assumes MainWindow or ChatTabManager ensures an initial tab and its model are ready.

            active_chat_tab = main_window.chat_tab_manager.get_active_chat_tab_instance()
            chat_list_model_instance = None

            # Set the view reference for the delegate of the active tab
            # This is important for QMovie updates.
            if active_chat_tab:
                chat_display_area = active_chat_tab.get_chat_display_area()
                if chat_display_area:
                    chat_list_model_instance = chat_display_area.get_model()
                    if chat_display_area.chat_item_delegate:
                        chat_display_area.chat_item_delegate.setView(
                            chat_display_area.chat_list_view)  # Pass the QListView
                        logger.info("Set view reference for initial active tab's delegate.")

            # Fallback or if no tab is initially active but a "default" model view exists (e.g. hidden global context view)
            # This part is speculative as the structure focuses on tabbed views for chat display.
            # if not chat_list_model_instance and hasattr(main_window, '_get_primary_chat_display_area_model'):
            #     chat_list_model_instance = main_window._get_primary_chat_display_area_model()

            if chat_list_model_instance:
                backend_coordinator_instance = chat_manager.get_backend_coordinator()
                if backend_coordinator_instance:
                    # Create the state handler, it will connect to BC signals
                    chat_message_state_handler = ChatMessageStateHandler(
                        model=chat_list_model_instance,
                        backend_coordinator=backend_coordinator_instance,
                        parent=app  # Make it a child of the app for lifecycle management
                    )
                    logger.info("ChatMessageStateHandler instantiated and wired to initial active model.")
                else:
                    logger.error("Critical: BackendCoordinator not available. ChatMessageStateHandler NOT created.")
            else:
                logger.warning(
                    "ChatMessageStateHandler NOT created: Could not get a ChatListModel instance from initial active tab. Loading indicators may not function until a tab is active and handler is initialized/connected.")
                # Consider: If no initial tab, state handler might need to be created/connected later
                # when the first tab is actually made active by ChatManager.
                # This could be done by ChatManager emitting a signal like "activeModelChanged"
                # that main.py or another coordinator listens to.

        # --- End Instantiate ChatMessageStateHandler ---

        logger.info("--- Core Components Instantiated ---")
    except Exception as e:
        logger.exception(" ***** FATAL ERROR DURING COMPONENT INSTANTIATION ***** ")
        try:
            QMessageBox.critical(None, "Fatal Init Error", f"Failed during component setup:\n{e}\n\nCheck logs.")
        except Exception:
            print(f"[CRITICAL] Component Init Failed: {e}\nTraceback:\n{traceback.format_exc()}", file=sys.stderr)
        await app.quit()  # type: ignore
        return 1

    QTimer.singleShot(100, chat_manager.initialize)  # ChatManager late init
    logger.info("Scheduled ChatManager late initialization.")

    if main_window:
        main_window.setGeometry(100, 100, 1100, 850)
        main_window.show()
        logger.info("--- Main Window Shown ---")
    else:
        logger.error("MainWindow instance not created, cannot show window.")
        await app.quit()  # type: ignore
        return 1

    logger.info("--- Starting Application Event Loop (via qasync) ---")
    await asyncio.Future()
    logger.info(f"--- Application Event Loop Finished ---")
    return 0


if __name__ == "__main__":
    try:
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        q_app_instance = QApplication.instance() or QApplication(sys.argv)
        event_loop = qasync.QEventLoop(q_app_instance)
        asyncio.set_event_loop(event_loop)
        with event_loop:
            exit_code = event_loop.run_until_complete(async_main())
        sys.exit(exit_code)
    except RuntimeError as e:
        if "cannot be nested" in str(e) or "already running" in str(e):
            logger.warning(f"qasync event loop issue: {e}.")
            if QApplication.instance() and QApplication.instance().property("activeWindow"):
                pass  # type: ignore
            else:
                sys.exit(1)
        else:
            logger.critical(f"RuntimeError during qasync execution: {e}", exc_info=True)
            try:
                _dummy_app = QApplication.instance() or QApplication(sys.argv); QMessageBox.critical(None,
                                                                                                     "Runtime Error",
                                                                                                     f"Application failed to run:\n{e}\n\nCheck logs.")  # type: ignore
            except Exception:
                pass
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception during application startup/run: {e}", exc_info=True)
        try:
            _dummy_app = QApplication.instance() or QApplication(sys.argv); QMessageBox.critical(None,
                                                                                                 "Unhandled Exception",
                                                                                                 f"An unexpected error occurred:\n{e}\n\nCheck logs.")  # type: ignore
        except Exception:
            pass
        sys.exit(1)