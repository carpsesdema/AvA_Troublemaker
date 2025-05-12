# ui/chat_tab_manager.py
import logging
from typing import Optional, Dict, List, Any

from PyQt6.QtWidgets import QTabWidget, QWidget
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt

try:
    from .chat_tab_widget import ChatTabWidget
    from core.chat_manager import ChatManager
    from core.models import ChatMessage # Keep for type hinting if ChatTabWidget exposes it
    from utils import constants
except ImportError as e:
    logging.critical(f"ChatTabManager: Failed to import critical components: {e}")
    ChatTabWidget = type("ChatTabWidget", (object,), {
        "get_chat_input_bar": lambda: type("DummyChatInputBar", (object,), {
            "sendMessageRequested": type("DummySignal", (object,), {"connect": lambda x: None}),
            "handle_busy_state": lambda x: None, "set_focus": lambda: None})(),
        "get_chat_display_area": lambda: type("DummyChatDisplayArea", (object,), {
            "textCopied": type("DummySignal", (object,), {"connect": lambda x: None}),
            "load_history_into_model": lambda x: None, "clear_model_display": lambda: None})(),
        "deleteLater": lambda: None
    })
    ChatManager = type("ChatManager", (object,), {
        "get_current_project_id": lambda: None,
        "set_current_project": lambda x: None,
        "get_project_history": lambda x: [],
        "process_user_message": lambda x, y: None, # type: ignore
        "get_project_context_manager": lambda: type("DummyPCM", (object,),
                                                    {"get_project_name": lambda x: "DummyProject"})()
    })
    constants = type("constants", (object,), {"GLOBAL_COLLECTION_ID": "global_collection"})
    ChatMessage = type("ChatMessage", (object,), {})

logger = logging.getLogger(__name__)


class ChatTabManager(QObject):
    """
    Manages the QTabWidget for displaying multiple chat sessions.
    Handles creation, activation, and closing of chat tabs,
    and coordinates with ChatManager for history and active project state.
    Prevents the 'Global Context' from being shown as a dedicated tab.
    """
    text_copied_from_tab = pyqtSignal(str, str, str)  # project_id, message, color
    active_project_context_changed = pyqtSignal(str)  # project_id

    def __init__(self,
                 tab_widget: QTabWidget,
                 chat_manager: ChatManager,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(tab_widget, QTabWidget):
            raise TypeError("ChatTabManager requires a valid QTabWidget instance.")
        if not isinstance(chat_manager, ChatManager):
            raise TypeError("ChatTabManager requires a valid ChatManager instance.")

        self.tab_widget = tab_widget
        self.chat_manager = chat_manager

        self._project_id_to_chat_tab_instance: Dict[str, ChatTabWidget] = {}
        self._is_programmatic_tab_change: bool = False

        self._connect_tab_widget_signals()
        logger.info("ChatTabManager initialized.")

    def _connect_tab_widget_signals(self):
        self.tab_widget.currentChanged.connect(self._handle_ui_tab_changed)
        self.tab_widget.tabCloseRequested.connect(self._handle_tab_close_requested)

    @pyqtSlot(int)
    def _handle_ui_tab_changed(self, index: int):
        if self._is_programmatic_tab_change or index == -1:
            logger.debug(f"Tab UI change to index {index} ignored (programmatic or invalid).")
            return

        current_chat_tab_instance = self.tab_widget.widget(index)
        if isinstance(current_chat_tab_instance, ChatTabWidget):
            project_id_for_tab = self._get_project_id_for_tab_instance(current_chat_tab_instance)
            if project_id_for_tab:
                logger.info(
                    f"UI Tab changed to index {index}, Project ID: '{project_id_for_tab}'. Setting active in ChatManager.")
                if self.chat_manager.get_current_project_id() != project_id_for_tab:
                    self.chat_manager.set_current_project(project_id_for_tab)
                else:
                    self.active_project_context_changed.emit(project_id_for_tab)
            else:
                logger.warning(f"Could not find project_id for tab at index {index} during UI change.")
        else:
            logger.warning(
                f"Widget at tab index {index} is not a ChatTabWidget instance ({type(current_chat_tab_instance)}).")

    @pyqtSlot(int)
    def _handle_tab_close_requested(self, index: int):
        chat_tab_instance_to_close = self.tab_widget.widget(index)
        project_id_to_close = self._get_project_id_for_tab_instance(chat_tab_instance_to_close)

        if not project_id_to_close:
            logger.error(f"Cannot close tab at index {index}: No project_id mapping found. Removing orphan tab.")
            self.tab_widget.removeTab(index)
            if isinstance(chat_tab_instance_to_close, ChatTabWidget):
                chat_tab_instance_to_close.deleteLater()
            return

        # Global Context tab should not exist to be closed, but defensive check.
        if project_id_to_close == constants.GLOBAL_COLLECTION_ID:
            logger.warning("Attempt to close Global Context tab, which should not exist. Ignoring.")
            # If it somehow exists, remove it without further logic.
            if self.tab_widget.widget(index) is chat_tab_instance_to_close:
                self.tab_widget.removeTab(index)
            if isinstance(chat_tab_instance_to_close, ChatTabWidget):
                chat_tab_instance_to_close.deleteLater()
            return

        logger.info(f"Closing tab for project ID: '{project_id_to_close}' at index {index}.")
        self.tab_widget.removeTab(index)

        if project_id_to_close in self._project_id_to_chat_tab_instance:
            del self._project_id_to_chat_tab_instance[project_id_to_close]

        if isinstance(chat_tab_instance_to_close, ChatTabWidget):
            chat_tab_instance_to_close.deleteLater()

        if self.tab_widget.count() == 0:
            logger.info("All closable tabs closed. ChatManager will default to Global Context for RAG if no project is selected.")
            # MainWindow should ensure a sensible state, perhaps selecting Global in LeftPanel
            # or prompting for new project if no project context makes sense.
            # We emit the active project changed with Global to reflect this state if ChatManager defaults.
            if self.chat_manager.get_current_project_id() != constants.GLOBAL_COLLECTION_ID:
                 if self.chat_manager.get_project_context_manager().get_all_projects_info(): # Only if there are projects
                     # Try to select the first available project if one exists
                     all_pids = list(self.chat_manager.get_project_context_manager().get_all_projects_info().keys())
                     first_non_global = next((pid for pid in all_pids if pid != constants.GLOBAL_COLLECTION_ID), None)
                     if first_non_global:
                         self.chat_manager.set_current_project(first_non_global)
                         return # set_current_project will trigger ensure_tab_active_and_loaded

            # If no other project to switch to, ensure global context is signalled as active
            # This will NOT create a tab, but will update UI elements like window title.
            self.active_project_context_changed.emit(constants.GLOBAL_COLLECTION_ID)


    def _get_project_id_for_tab_instance(self, tab_instance: Optional[QWidget]) -> Optional[str]:
        if not isinstance(tab_instance, ChatTabWidget):
            return None
        for pid, instance in self._project_id_to_chat_tab_instance.items():
            if instance == tab_instance:
                return pid
        return None

    @pyqtSlot(str)
    def ensure_tab_active_and_loaded(self, project_id: str):
        logger.info(f"ChatTabManager: Ensuring tab active/loaded for Project ID: '{project_id}'")
        if not project_id:
            logger.error("ChatTabManager: Cannot ensure tab, project_id is None or empty.")
            return

        # --- MODIFICATION START: Do not create a visible tab for Global Context ---
        if project_id == constants.GLOBAL_COLLECTION_ID:
            logger.info(f"ChatTabManager: Global Context ('{project_id}') selected. No dedicated tab will be shown. RAG context is active.")
            self.active_project_context_changed.emit(project_id)
            # If no other tabs are open, user must select/create a project via LeftPanel for chat.
            # The UI should reflect that Global is active for RAG, but chat happens in a project tab.
            if self.tab_widget.count() == 0:
                logger.info("ChatTabManager: Global context is active, and no project tabs are open. User should select/create a project for chat.")
            return # Exit before creating/activating a tab for Global Context
        # --- MODIFICATION END ---

        project_name: Optional[str] = None
        pcm = self.chat_manager.get_project_context_manager()
        if pcm:
            project_name = pcm.get_project_name(project_id)

        if not project_name:
            project_name = project_id
            logger.warning(f"ChatTabManager: Project name for ID '{project_id}' not found in PCM. Using ID as name for tab.")

        effective_project_name = project_name
        chat_tab_instance = self._project_id_to_chat_tab_instance.get(project_id)

        if not chat_tab_instance:
            logger.debug(f"No existing tab for project '{project_id}'. Creating new one named '{effective_project_name}'.")
            chat_tab_instance = ChatTabWidget(self.tab_widget)
            input_bar = chat_tab_instance.get_chat_input_bar()
            if input_bar and hasattr(input_bar, 'sendMessageRequested'):
                input_bar.sendMessageRequested.connect(
                    lambda text, images, pid=project_id: self._handle_send_message_for_project(pid, text, images)
                )
            display_area = chat_tab_instance.get_chat_display_area()
            if display_area and hasattr(display_area, 'textCopied'):
                display_area.textCopied.connect(
                    lambda msg_text, color, pid=project_id: self.text_copied_from_tab.emit(pid, msg_text, color)
                )
            tab_index = self.tab_widget.addTab(chat_tab_instance, effective_project_name)
            self.tab_widget.setTabToolTip(tab_index, f"Chat for: {effective_project_name} (ID: {project_id})")
            self._project_id_to_chat_tab_instance[project_id] = chat_tab_instance
            logger.info(f"Created and added new tab for '{effective_project_name}' (ID: {project_id}).")

        if self.tab_widget.currentWidget() != chat_tab_instance:
            self._is_programmatic_tab_change = True
            self.tab_widget.setCurrentWidget(chat_tab_instance)
            self._is_programmatic_tab_change = False
            logger.debug(f"Programmatically set current tab to project '{project_id}'.")

        history: List[ChatMessage] = self.chat_manager.get_project_history(project_id)
        display_area_for_history = chat_tab_instance.get_chat_display_area()
        if display_area_for_history:
            display_area_for_history.load_history_into_model(history)
            logger.debug(f"Loaded/Refreshed history ({len(history)} msgs) for tab '{project_id}'.")
        else:
            logger.error(f"DisplayArea not found to load history for project '{project_id}'")

        input_bar_for_focus = chat_tab_instance.get_chat_input_bar()
        if input_bar_for_focus:
            input_bar_for_focus.set_focus()
        else:
            logger.error(f"InputBar not found to set focus for project '{project_id}'")

        self.active_project_context_changed.emit(project_id)

    def _handle_send_message_for_project(self, project_id: str, text: str, images: List[Dict[str, Any]]):
        # Ensure ChatManager's active project IS the one this tab represents,
        # as the user is explicitly sending from THIS tab.
        if self.chat_manager.get_current_project_id() != project_id:
            logger.info(
                f"Send event from tab '{project_id}', but ChatManager active project is '{self.chat_manager.get_current_project_id()}'. "
                f"Switching ChatManager active project to '{project_id}' before sending."
            )
            self.chat_manager.set_current_project(project_id)
            # Allow the signal chain from set_current_project to ensure_tab_active_and_loaded
            # to complete before processing the message. This might involve a small delay.
            # However, process_user_message should operate on ChatManager's current state.
        self.chat_manager.process_user_message(text, images)

    def get_chat_tab_instance(self, project_id: str) -> Optional[ChatTabWidget]:
        return self._project_id_to_chat_tab_instance.get(project_id)

    def get_active_chat_tab_instance(self) -> Optional[ChatTabWidget]:
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, ChatTabWidget):
            return current_widget
        return None

    def update_busy_state_for_active_tab(self, is_busy: bool):
        active_tab = self.get_active_chat_tab_instance()
        if active_tab:
            input_bar = active_tab.get_chat_input_bar()
            if input_bar and hasattr(input_bar, 'handle_busy_state'):
                input_bar.handle_busy_state(is_busy)

    def update_tab_name(self, project_id: str, new_name: str):
        instance = self._project_id_to_chat_tab_instance.get(project_id)
        if instance:
            # Do not update tab name for Global Context as it shouldn't have a tab
            if project_id == constants.GLOBAL_COLLECTION_ID:
                return

            for i in range(self.tab_widget.count()):
                if self.tab_widget.widget(i) == instance:
                    effective_name = new_name # Global context display name handled by not creating a tab
                    self.tab_widget.setTabText(i, effective_name)
                    self.tab_widget.setTabToolTip(i, f"Chat for: {effective_name} (ID: {project_id})")
                    logger.info(f"Updated tab name for project '{project_id}' to '{effective_name}'.")
                    if self.tab_widget.currentWidget() == instance:
                        self.active_project_context_changed.emit(project_id)
                    break
        else:
            logger.warning(f"Cannot update tab name: No tab instance found for project_id '{project_id}'.")