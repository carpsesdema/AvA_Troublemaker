# ui/main_window.py
import logging
import os
import re
import sys
from typing import Optional, List, Dict, Any
import datetime

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QApplication, QMessageBox,
    QDialog, QLabel, QStyle, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QSize, QEvent, QPoint, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QMovie, QCloseEvent, QShortcut, QKeyEvent, QKeySequence, QCursor

try:
    from core.chat_manager import ChatManager, DEFAULT_CHAT_BACKEND_ID
    from core.models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE, MODEL_ROLE, USER_ROLE
    from core.message_enums import MessageLoadingState # Added import
    from .left_panel import LeftControlPanel
    from .dialog_service import DialogService
    from .chat_tab_manager import ChatTabManager
    from utils import constants
except ImportError as e:
    logging.basicConfig(level=logging.DEBUG)
    logging.critical(f"CRITICAL IMPORT ERROR in main_window.py: {e}", exc_info=True)
    try:
        _dummy_app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()
        QMessageBox.critical(None, "Import Error",
                             f"Failed to import critical components:\n{e}\nPlease check installation and paths.")
    except Exception as msg_e:
        logging.critical(f"Failed to show import error message box: {msg_e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    def __init__(self, chat_manager: ChatManager, app_base_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        logger.info("MainWindow initializing...")
        if not isinstance(chat_manager, ChatManager):
            logger.critical("MainWindow requires a valid ChatManager instance.")
            raise TypeError("MainWindow requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        self.app_base_path = app_base_path
        # self._last_added_message_was_streaming = False # REPLACED by ID-based logic

        self.left_panel: Optional[LeftControlPanel] = None
        self.status_label: Optional[QLabel] = None
        self._status_clear_timer: Optional[QTimer] = None
        self.dialog_service: Optional[DialogService] = None
        self.main_tab_widget: Optional[QTabWidget] = None
        self.chat_tab_manager: Optional[ChatTabManager] = None

        self._last_token_display_str: Optional[str] = None
        self._current_base_status_text: str = "Status: Initializing..."
        self._current_base_status_color: str = "#abb2bf"

        try:
            self.dialog_service = DialogService(self, self.chat_manager)
        except Exception as e:
            logger.critical(f"Failed to initialize DialogService: {e}", exc_info=True)
            QApplication.quit(); return # type: ignore

        self._init_ui()

        if self.main_tab_widget is not None:
            try:
                self.chat_tab_manager = ChatTabManager(self.main_tab_widget, self.chat_manager, self)
            except Exception as e:
                logger.critical(f"Failed to initialize ChatTabManager: {e}", exc_info=True)
                QApplication.quit(); return # type: ignore
        else:
            logger.critical("main_tab_widget is None after _init_ui. Cannot initialize ChatTabManager.")
            QApplication.quit(); return # type: ignore

        self._apply_styles()
        self._connect_signals()
        self._setup_window()
        logger.info("MainWindow initialized successfully.")

    def _setup_window(self):
        self.setWindowTitle(constants.APP_NAME)
        try:
            app_icon_path = os.path.join(constants.ASSETS_PATH, "Synchat.ico")
            std_fallback_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            app_icon = QIcon(app_icon_path) if os.path.exists(app_icon_path) else std_fallback_icon
            if not app_icon.isNull(): self.setWindowIcon(app_icon)
            elif not std_fallback_icon.isNull(): self.setWindowIcon(std_fallback_icon)
        except Exception as e: logger.error(f"Error setting window icon: {e}", exc_info=True)
        self.update_window_title()

    def _init_ui(self):
        logger.debug("MainWindow initializing UI...")
        main_hbox_layout = QHBoxLayout(self)
        main_hbox_layout.setContentsMargins(0,0,0,0)
        main_hbox_layout.setSpacing(0)
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setHandleWidth(1)

        try:
            self.left_panel = LeftControlPanel(self)
            assert self.left_panel is not None, "LeftControlPanel creation failed."
            self.left_panel.setObjectName("LeftPanel")
            self.left_panel.setMinimumWidth(270)
        except Exception as e_lcp:
            logger.critical(f"CRITICAL ERROR creating LeftControlPanel: {e_lcp}", exc_info=True)
            raise RuntimeError("Failed to create LeftControlPanel") from e_lcp

        right_panel_widget = QWidget(self)
        right_panel_widget.setObjectName("RightPanelContainer")
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(5,5,5,5)
        right_panel_layout.setSpacing(5)
        right_panel_widget.setMinimumWidth(400)

        try:
            self.main_tab_widget = QTabWidget(self)
            assert self.main_tab_widget is not None, "QTabWidget creation failed."
            self.main_tab_widget.setObjectName("MainChatTabsWidget")
            self.main_tab_widget.setTabsClosable(True)
            self.main_tab_widget.setMovable(True)
            right_panel_layout.addWidget(self.main_tab_widget, 1)
        except Exception as e_tab:
            logger.critical(f"CRITICAL ERROR creating QTabWidget: {e_tab}", exc_info=True)
            raise RuntimeError("Failed to create QTabWidget") from e_tab

        status_bar_widget = QWidget(self)
        status_bar_widget.setObjectName("StatusBarWidget")
        status_bar_layout = QHBoxLayout(status_bar_widget)
        status_bar_layout.setContentsMargins(5,2,5,2)
        status_bar_layout.setSpacing(10)
        self.status_label = QLabel("Status: Initializing...", self)
        self.status_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1))
        self.status_label.setObjectName("StatusLabel")
        status_bar_layout.addWidget(self.status_label, 1)
        right_panel_layout.addWidget(status_bar_widget)

        main_splitter.addWidget(self.left_panel)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([270, 730])
        main_splitter.setStretchFactor(0,0)
        main_splitter.setStretchFactor(1,1)
        main_hbox_layout.addWidget(main_splitter)
        self.setLayout(main_hbox_layout)
        logger.debug("MainWindow UI layout complete.")

    def _apply_styles(self):
        stylesheet = ""
        logger.debug("Applying styles...")
        try:
            main_style_path = next((p for p in constants.STYLE_PATHS_TO_CHECK if os.path.exists(p)), None)
            if main_style_path:
                with open(main_style_path, "r", encoding="utf-8") as f: stylesheet += f.read() + "\n"
            bubble_style_path = constants.BUBBLE_STYLESHEET_PATH
            if os.path.exists(bubble_style_path):
                with open(bubble_style_path, "r", encoding="utf-8") as f: stylesheet += f.read()
            if stylesheet: self.setStyleSheet(stylesheet)
            else: self.setStyleSheet("QWidget { background-color: #282c34; color: #abb2bf; }")
        except Exception as e:
            logger.exception(f"Error loading/applying stylesheet(s): {e}")
            self.setStyleSheet("QWidget { background-color: #333; color: #EEE; }")

    def _connect_signals(self):
        logger.debug("MainWindow: Connecting signals...")
        if not all([self.chat_manager, self.left_panel, self.dialog_service, self.chat_tab_manager]):
            logger.critical("Cannot connect signals: Crucial components missing."); return

        lp = self.left_panel
        lp.newSessionClicked.connect(self._handle_new_session_for_active_tab)
        lp.manageSessionsClicked.connect(self._show_session_manager)
        lp.uploadFileClicked.connect(self._trigger_file_upload)
        lp.uploadDirectoryClicked.connect(self._trigger_dir_upload)
        lp.editPersonalityClicked.connect(self._show_personality_editor)
        lp.viewCodeBlocksClicked.connect(self._show_code_viewer)
        lp.viewRagContentClicked.connect(self._show_rag_viewer)
        lp.newProjectClicked.connect(self._handle_new_project_request)
        lp.uploadGlobalClicked.connect(self._trigger_global_upload_menu)
        lp.modelSelected.connect(self.chat_manager.set_model)
        lp.projectSelected.connect(self.chat_manager.set_current_project)
        lp.temperatureChanged.connect(self._handle_temperature_changed)

        ctm = self.chat_tab_manager
        ctm.text_copied_from_tab.connect(lambda pid, text, color: self._handle_text_copied(text, color))
        ctm.active_project_context_changed.connect(self._handle_tab_manager_active_context_change)

        cm = self.chat_manager
        cm.history_changed.connect(self._handle_history_changed)
        cm.new_message_added.connect(self._handle_new_message)
        cm.status_update.connect(self.update_status)
        cm.error_occurred.connect(self._handle_error)
        cm.busy_state_changed.connect(self._handle_busy_state)
        cm.config_state_changed.connect(self._handle_config_state)
        cm.stream_started.connect(self._handle_stream_started)
        cm.stream_chunk_received.connect(self._handle_stream_chunk)
        cm.stream_finished.connect(self._handle_stream_finished)
        cm.code_file_updated.connect(self._handle_code_file_update)
        cm.project_inventory_updated.connect(lp.handle_project_inventory_update)
        cm.current_project_changed.connect(ctm.ensure_tab_active_and_loaded)
        cm.available_models_changed.connect(self._handle_available_models_changed)
        cm.token_usage_updated.connect(self._handle_token_usage_update)

        shortcuts = {
            "Ctrl+N": self._handle_new_session_for_active_tab, "Ctrl+O": self._show_session_manager,
            "Ctrl+S": self._trigger_save_session, "Ctrl+Shift+S": self._trigger_save_as_session,
            "Ctrl+U": self._trigger_file_upload, "Ctrl+Shift+U": self._trigger_dir_upload,
            "Ctrl+G": self._trigger_global_upload_menu, "Ctrl+P": self._show_personality_editor,
            "Ctrl+B": self._show_code_viewer, "Ctrl+R": self._show_rag_viewer,
            "Ctrl+W": self._close_current_tab_action
        }
        for seq, func in shortcuts.items(): QShortcut(QKeySequence(seq), self).activated.connect(func)
        logger.debug("MainWindow: Signal connections complete.")

        if self.left_panel and hasattr(self.chat_manager, '_current_chat_temperature'):
            initial_temp = getattr(self.chat_manager, '_current_chat_temperature', 0.7)
            self.left_panel.set_temperature_ui(initial_temp)
            logger.info(f"Initialized LeftPanel temperature UI from ChatManager's default: {initial_temp:.2f}")

    @pyqtSlot(list)
    def _handle_history_changed(self, history: List[ChatMessage]):
        pcm = self.chat_manager.get_project_context_manager()
        active_project_id = pcm.get_active_project_id() if pcm else constants.GLOBAL_COLLECTION_ID
        logger.debug(f"MW Slot: Handling history changed for project '{active_project_id}' (count: {len(history)})")
        if not (self.chat_tab_manager and active_project_id): return
        target_tab = self.chat_tab_manager.get_chat_tab_instance(active_project_id)
        if target_tab and (display_area := target_tab.get_chat_display_area()):
            display_area.load_history_into_model(history)
            self._rescan_history_for_code_blocks(history)
        self.update_window_title()
        self._update_rag_button_state()

    @pyqtSlot(object)
    def _handle_new_message(self, message: ChatMessage):
        # ---- NEW DISTINCT LOG ----
        logger.error(f"!!!!!!!! MW _handle_new_message ENTRY: Role={message.role}, ID={message.id}, Text='{message.text[:100]}...' !!!!!")
        # ---- END NEW DISTINCT LOG ----

        pcm = self.chat_manager.get_project_context_manager()
        active_project_id = pcm.get_active_project_id() if pcm else constants.GLOBAL_COLLECTION_ID
        logger.info(f"MW Slot _handle_new_message for project '{active_project_id}' (ID: {message.id}, Role: {message.role})")

        if not (self.chat_tab_manager and active_project_id): return
        target_tab = self.chat_tab_manager.get_chat_tab_instance(active_project_id)
        if not target_tab:
            logger.error(f"No tab instance for active project '{active_project_id}' when handling new message.")
            return
        target_display_area = target_tab.get_chat_display_area()
        if not target_display_area:
            logger.critical(f"DisplayArea missing for project '{active_project_id}'.")
            return
        model = target_display_area.get_model()
        if not model:
            logger.critical(f"ChatListModel missing for project '{active_project_id}'.")
            return

        is_internal_system_message = message.metadata and message.metadata.get("is_internal", False)
        if is_internal_system_message:
            logger.debug(f"Skipping internal system message in _handle_new_message (ID: {message.id})")
            return

        if message.role == MODEL_ROLE:
            existing_row = model.find_message_row_by_id(message.id)
            if existing_row is not None:
                logger.debug(f"Found existing AI message (ID: {message.id}) at row {existing_row}. Updating.")
                if message.metadata and "is_streaming" in message.metadata:
                    del message.metadata["is_streaming"]
                if message.loading_state == MessageLoadingState.LOADING:
                    message.loading_state = MessageLoadingState.COMPLETED
                target_display_area.update_message_in_model(existing_row, message)
            else:
                logger.debug(f"Adding new AI message (ID: {message.id}) to model.")
                target_display_area.add_message_to_model(message)
        else:
            logger.debug(f"Adding new non-AI message (ID: {message.id}, Role: {message.role}) to model.")
            target_display_area.add_message_to_model(message)

        if message.text:
            self._scan_message_for_code_blocks(message)

        is_upload_summary = message.metadata and message.metadata.get("upload_summary") is not None
        if is_upload_summary:
            self._update_rag_button_state()

    @pyqtSlot(str) # request_id
    def _handle_stream_started(self, request_id: str):
        active_project_id = self.chat_manager.get_current_project_id()
        logger.info(f"MW SLOT _handle_stream_started for project '{active_project_id}', request_id '{request_id}'.")
        pass

    @pyqtSlot(str)
    def _handle_stream_chunk(self, chunk: str):
        active_project_id = self.chat_manager.get_current_project_id()
        if not (self.chat_tab_manager and active_project_id): return

        target_tab = self.chat_tab_manager.get_chat_tab_instance(active_project_id)
        if target_tab:
            display_area = target_tab.get_chat_display_area()
            if display_area:
                display_area.append_stream_chunk_to_model(chunk)
            else:
                logger.error(f"No display area in active tab for project '{active_project_id}' to append chunk.")
        else:
            logger.error(f"No active tab found for project '{active_project_id}' to append chunk.")


    @pyqtSlot()
    def _handle_stream_finished(self):
        active_project_id = self.chat_manager.get_current_project_id()
        logger.info(f"MW SLOT _handle_stream_finished for project '{active_project_id}'.")

        if not (self.chat_tab_manager and active_project_id): return
        target_tab = self.chat_tab_manager.get_chat_tab_instance(active_project_id)
        if target_tab:
            display_area = target_tab.get_chat_display_area()
            if display_area:
                try:
                    display_area.finalize_stream_in_model()
                except Exception as e:
                    logger.exception(f"ERROR finalizing stream for project '{active_project_id}'")
                    self.update_status(f"Error finalizing stream display: {e}", "#e06c75", True, 5000)
            else:
                logger.error(f"No display area in active tab for project '{active_project_id}' to finalize stream.")
        else:
            logger.error(f"No active tab found for project '{active_project_id}' to finalize stream.")

        self._update_rag_button_state()

    @pyqtSlot(bool)
    def _handle_busy_state(self, is_busy: bool):
        logger.debug(f"MW Slot: Handling global busy state: {is_busy}")
        api_ready = self.chat_manager.is_api_ready()
        if self.left_panel: self.left_panel.set_enabled_state(enabled=api_ready, is_busy=is_busy)
        if self.chat_tab_manager: self.chat_tab_manager.update_busy_state_for_active_tab(is_busy)

    @pyqtSlot(str, str, bool, int)
    def update_status(self, message: str, color: str = "#abb2bf", is_temporary: bool = False, duration_ms: int = 0):
        logger.debug(f"MW Slot: update_status called with: '{message}', color: {color}, temp: {is_temporary}")
        self._current_base_status_text = message
        self._current_base_status_color = color

        if "Error" in message or "Busy" in message or "Not Configured" in message or "Cancelled" in message or "failed" in message.lower():
            self._last_token_display_str = None

        self._refresh_full_status_display()

        if self._status_clear_timer and self._status_clear_timer.isActive():
            self._status_clear_timer.stop()

        if is_temporary and duration_ms > 0:
            if not self._status_clear_timer:
                self._status_clear_timer = QTimer(self)
                self._status_clear_timer.setSingleShot(True)
                self._status_clear_timer.timeout.connect(self.update_status_based_on_state)
            if self._status_clear_timer:
                self._status_clear_timer.start(duration_ms)

    def _refresh_full_status_display(self):
        if not self.status_label:
            return

        full_status_text = self._current_base_status_text
        if self._last_token_display_str:
            full_status_text += f"  |  {self._last_token_display_str}"

        try:
            if not isinstance(self._current_base_status_color, str) or not self._current_base_status_color.startswith('#') or len(self._current_base_status_color) not in [4, 7]:
                logger.warning(f"Invalid status color: {self._current_base_status_color}. Defaulting.")
                self._current_base_status_color = "#abb2bf"
            self.status_label.setStyleSheet(f"QLabel#StatusLabel {{ color: {self._current_base_status_color}; }}")
        except Exception as e:
            logger.exception(f"Error setting status label stylesheet: {e}")

        self.status_label.setText(full_status_text)

    @pyqtSlot()
    def update_status_based_on_state(self):
        if self.chat_manager: self.chat_manager.update_status_based_on_state()

    @pyqtSlot(str, bool)
    def _handle_config_state(self, model_name: str, personality_active: bool):
        logger.debug(f"MW Slot: Handling config state. Model: {model_name}, Pers Active: {personality_active}")
        if self.left_panel:
            self.left_panel.update_model_selection(model_name)
            self.left_panel.update_personality_tooltip(active=personality_active)
        self._update_rag_button_state()
        self.update_window_title()

    @pyqtSlot(str, bool)
    def _handle_error(self, error_message: str, is_critical: bool):
        logger.error(f"MW Slot: Handling error: '{error_message}', Critical: {is_critical}")
        self.update_status(f"Error: {error_message}", "#e06c75", True, 8000)
        critical_keywords = ["api", "config", "connection", "fatal", "critical", "service", "failed to load", "permission", "quota", "backend", "upload", "saving"]
        if is_critical or any(kw in error_message.lower() for kw in critical_keywords):
            QMessageBox.warning(self, "Application Error" if is_critical else "Warning", error_message)

    @pyqtSlot(str, str)
    def _handle_code_file_update(self, filename: str, content: str):
        logger.info(f"MW Slot: Received updated code for '{filename}'")
        if not self.dialog_service: return
        try:
            code_viewer = self.dialog_service.show_code_viewer(ensure_creation=True)
            if code_viewer: code_viewer.update_or_add_file(filename, content)
        except Exception as e:
            logger.exception(f"Error handling code file update for {filename}: {e}")
            self.update_status(f"Error showing code update: {e}", "#e06c75", True, 5000)

    @pyqtSlot(list)
    def _handle_available_models_changed(self, models: List[str]):
        logger.info(f"MW Slot: Handling available models changed ({len(models)} models).")
        if self.left_panel and self.chat_manager:
            try:
                current_model = self.chat_manager.get_current_model()
                self.left_panel.update_model_list(models, current_model)
            except Exception: logger.exception("Error updating model list in LeftPanel:")

    @pyqtSlot(float)
    def _handle_temperature_changed(self, temperature: float):
        logger.info(f"MainWindow: Temperature changed via UI to {temperature:.2f}, passing to ChatManager.")
        self.chat_manager.set_chat_temperature(temperature)

    @pyqtSlot(str, int, int, int)
    def _handle_token_usage_update(self, backend_id: str, prompt_tokens: int, completion_tokens: int, model_max_context: int):
        if backend_id != DEFAULT_CHAT_BACKEND_ID:
            return

        token_info_parts = []
        if prompt_tokens is not None:
            token_info_parts.append(f"P: {prompt_tokens}")
        if completion_tokens is not None:
            token_info_parts.append(f"C: {completion_tokens}")

        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if total_tokens > 0 or (prompt_tokens is not None and completion_tokens is not None) :
             token_info_parts.append(f"T: {total_tokens}")

        if not token_info_parts:
            self._last_token_display_str = None
        else:
            token_info_str = " ".join(token_info_parts)
            if model_max_context > 0:
                token_info_str += f" / Max: {model_max_context}"
            self._last_token_display_str = f"Tokens: {token_info_str}"

        self._refresh_full_status_display()
        logger.info(f"Status bar token info: {self._last_token_display_str or 'Cleared'}")

    @pyqtSlot(str)
    def _handle_tab_manager_active_context_change(self, active_project_id: str):
        logger.info(f"MW: TabManager confirmed active context: {active_project_id}")
        self.update_window_title()
        self._update_rag_button_state()
        if self.left_panel:
            self.left_panel.handle_active_project_ui_update(active_project_id)

    def _handle_new_session_for_active_tab(self):
        self.chat_manager.start_new_chat()

    def _close_current_tab_action(self):
        if self.main_tab_widget and self.chat_tab_manager and self.main_tab_widget.count() > 0:
            if (current_idx := self.main_tab_widget.currentIndex()) != -1:
                 self.chat_tab_manager._handle_tab_close_requested(current_idx)

    def update_window_title(self):
        try:
            base_title = constants.APP_NAME; details = []
            model_name = self.chat_manager.get_current_model()
            pers_active = bool(self.chat_manager.get_current_personality())
            pcm = self.chat_manager.get_project_context_manager()
            active_project_id = pcm.get_active_project_id() if pcm else constants.GLOBAL_COLLECTION_ID
            proj_display_name = "No Active Project"
            if self.chat_tab_manager and self.main_tab_widget and \
               (active_tab_instance := self.chat_tab_manager.get_active_chat_tab_instance()):
                if (active_tab_index := self.main_tab_widget.indexOf(active_tab_instance)) != -1:
                    proj_display_name = self.main_tab_widget.tabText(active_tab_index)
            elif active_project_id == constants.GLOBAL_COLLECTION_ID:
                proj_display_name = constants.GLOBAL_CONTEXT_DISPLAY_NAME

            if proj_display_name != "No Active Project": details.append(f"Ctx: {proj_display_name}")
            if model_name: model_short = model_name.split(':')[-1].split('/')[-1].replace("-latest", ""); details.append(f"M:{model_short}")
            if pers_active: details.append("P")
            if active_project_id and self.chat_manager.is_rag_available() and self.chat_manager.is_rag_context_initialized(active_project_id):
                details.append("RAG")
            self.setWindowTitle(base_title + (f" - [{ ' | '.join(details)}]" if details else ""))
        except Exception: logger.exception("Error updating window title:"); self.setWindowTitle(constants.APP_NAME)

    def _scan_message_for_code_blocks(self, message: ChatMessage):
        if not (self.dialog_service and hasattr(self.dialog_service, '_code_viewer_window') and \
                (viewer_instance := self.dialog_service._code_viewer_window)): return
        if (message.metadata and message.metadata.get("is_internal", False)) or not message.text: return
        try:
            code_pattern = re.compile(r"```(?:[a-zA-Z0-9_\-\.]*)?\s*\n?(.*?)```", re.DOTALL)
            matches_found = False
            for match in code_pattern.finditer(message.text):
                if code_content := match.group(1).strip():
                    block_name = f"Code Block ({datetime.datetime.now().strftime('%H%M%S_%f')[:-3]})"
                    if viewer_instance: viewer_instance.update_or_add_file(block_name, code_content)
                    matches_found = True
            if matches_found: logger.debug("Found and added code block(s) to viewer.")
        except Exception: logger.exception("Error processing code blocks for viewer:")

    def _rescan_history_for_code_blocks(self, history: List[ChatMessage]):
        if not (self.dialog_service and hasattr(self.dialog_service, '_code_viewer_window') and \
                self.dialog_service._code_viewer_window): return
        try:
            for message in history: self._scan_message_for_code_blocks(message)
        except Exception: logger.exception("Error during history rescan for code viewer:")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape and self.chat_manager and self.chat_manager.is_overall_busy():
            self.chat_manager._cancel_active_tasks()
            self.update_status("Attempting cancel...", "#e5c07b", True, 2000)
            event.accept()
        else: super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent):
        logger.info("MainWindow close event. Cleaning up...")
        if self.dialog_service:
            try: self.dialog_service.close_non_modal_dialogs()
            except Exception: logger.exception("Error closing dialogs:")
        if self.chat_manager:
            try: self.chat_manager.cleanup()
            except Exception: logger.exception("Error during ChatManager cleanup:")
        logger.info("Cleanup complete. Accepting close event.")
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        if self.chat_tab_manager: QTimer.singleShot(250, self._focus_active_input)
        self._update_rag_button_state()
        logger.info("MainWindow shown and initial focus scheduled.")

    def _focus_active_input(self):
         if self.chat_tab_manager and (active_tab := self.chat_tab_manager.get_active_chat_tab_instance()):
             if input_bar := active_tab.get_chat_input_bar(): input_bar.set_focus()

    def _update_rag_button_state(self):
        if not (self.left_panel and hasattr(self.left_panel, 'view_project_rag_button')): return
        rag_ready = False
        if self.chat_manager and (pcm := self.chat_manager.get_project_context_manager()):
            active_pid = pcm.get_active_project_id() or constants.GLOBAL_COLLECTION_ID
            if hasattr(self.chat_manager, 'is_rag_context_initialized'):
                rag_ready = self.chat_manager.is_rag_context_initialized(active_pid)
        self.left_panel.view_project_rag_button.setEnabled(rag_ready)

    @pyqtSlot(str, str)
    def _handle_text_copied(self, message: str, color: str):
        self.update_status(message, color, True, 2500)

    def _show_session_manager(self):
        if self.dialog_service: self.dialog_service.show_session_manager()

    def _trigger_save_session(self):
        current_session_file = None
        if self.chat_manager:
            sfm = self.chat_manager.get_session_flow_manager()
            if sfm:
                current_session_file = sfm.get_current_session_filepath()

        if current_session_file:
            self.chat_manager.save_current_chat_session(current_session_file)
        else:
            self._trigger_save_as_session()

    def _trigger_save_as_session(self):
        if self.dialog_service:
            filepath = self.dialog_service.get_save_as_path()
            if filepath: self.chat_manager.save_current_chat_session(filepath)

    def _get_upload_filter(self) -> str:
        filter_parts = ["All Files (*)"]
        text_ext = sorted([ext for ext in constants.ALLOWED_TEXT_EXTENSIONS if ext not in ['.pdf', '.docx']])
        if text_ext: filter_parts.append(f"Text/Code Files ({' '.join(['*' + e for e in text_ext])})")
        if '.pdf' in constants.ALLOWED_TEXT_EXTENSIONS: filter_parts.append("PDF Files (*.pdf)")
        if '.docx' in constants.ALLOWED_TEXT_EXTENSIONS: filter_parts.append("Word Documents (*.docx)")
        return ";;".join(filter_parts)

    def _trigger_file_upload(self):
        if self.dialog_service and self.chat_manager and not self.chat_manager.is_overall_busy():
            file_paths = self.dialog_service.get_upload_files_paths("Upload File(s) to Current Context")
            if file_paths:
                pcm = self.chat_manager.get_project_context_manager()
                active_pid = pcm.get_active_project_id() if pcm else constants.GLOBAL_COLLECTION_ID
                if active_pid == constants.GLOBAL_COLLECTION_ID: self.chat_manager.handle_global_file_upload(file_paths)
                else: self.chat_manager.handle_file_upload(file_paths)

    def _trigger_dir_upload(self):
        if self.dialog_service and self.chat_manager and not self.chat_manager.is_overall_busy():
            dir_path = self.dialog_service.get_upload_directory_path("Select Directory for Current Context")
            if dir_path:
                pcm = self.chat_manager.get_project_context_manager()
                active_pid = pcm.get_active_project_id() if pcm else constants.GLOBAL_COLLECTION_ID
                if active_pid == constants.GLOBAL_COLLECTION_ID: self.chat_manager.handle_global_directory_upload(dir_path)
                else: self.chat_manager.handle_directory_upload(dir_path)

    @pyqtSlot()
    def _trigger_global_upload_menu(self):
        if self.dialog_service and self.chat_manager and not self.chat_manager.is_overall_busy():
             choice = self.dialog_service.show_global_upload_menu()
             if choice == "file": self._trigger_global_file_upload_dialog()
             elif choice == "directory": self._trigger_global_dir_upload_dialog()

    def _trigger_global_file_upload_dialog(self):
        if self.dialog_service and self.chat_manager:
             file_paths = self.dialog_service.get_upload_files_paths("Upload File(s) to Global Knowledge")
             if file_paths: self.chat_manager.handle_global_file_upload(file_paths)

    def _trigger_global_dir_upload_dialog(self):
        if self.dialog_service and self.chat_manager:
             dir_path = self.dialog_service.get_upload_directory_path("Select Directory for Global Knowledge")
             if dir_path: self.chat_manager.handle_global_directory_upload(dir_path)

    def _show_personality_editor(self):
        if self.dialog_service and self.chat_manager and not self.chat_manager.is_overall_busy():
             new_prompt = self.dialog_service.show_edit_personality()
             if new_prompt is not None: self.chat_manager.set_personality(new_prompt)

    def _show_code_viewer(self, ensure_creation: bool = True):
        if not self.dialog_service: return
        code_viewer_instance = self.dialog_service.show_code_viewer(ensure_creation=ensure_creation)
        if code_viewer_instance and ensure_creation and self.chat_manager:
            pcm = self.chat_manager.get_project_context_manager()
            if pcm:
                active_pid = pcm.get_active_project_id() or constants.GLOBAL_COLLECTION_ID
                history = self.chat_manager.get_project_history(active_pid)
                self._rescan_history_for_code_blocks(history)

    def _show_rag_viewer(self):
        if self.dialog_service and self.chat_manager and self.chat_manager.is_rag_available():
             rag_viewer_instance = self.dialog_service.show_rag_viewer()
             if rag_viewer_instance and hasattr(rag_viewer_instance, 'focusRequested'):
                 try: rag_viewer_instance.focusRequested.disconnect(self.chat_manager.set_chat_focus)
                 except TypeError: pass
                 rag_viewer_instance.focusRequested.connect(self.chat_manager.set_chat_focus)
        elif self.chat_manager and not self.chat_manager.is_rag_available():
             QMessageBox.information(self, "RAG Not Ready", "RAG system not available/initialized. Upload documents to a context first.")

    @pyqtSlot()
    def _handle_new_project_request(self):
        if self.dialog_service:
            project_name = self.dialog_service.get_new_project_name()
            if project_name: self.chat_manager.create_project_collection(project_name)