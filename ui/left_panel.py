# ui/left_panel.py
import logging
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QStyle, QSizePolicy,
    QSpacerItem, QTreeView, QAbstractItemView, QComboBox, QGroupBox,
    QHBoxLayout, QSlider, QDoubleSpinBox
)
from PyQt6.QtGui import QFont, QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal, Qt, QSize, pyqtSlot, QModelIndex

try:
    import qtawesome as qta

    QTAWESOME_AVAILABLE = True
except ImportError:
    QTAWESOME_AVAILABLE = False
    logging.warning("LeftControlPanel: qtawesome library not found. Icons will be limited.")

from utils.constants import (
    CHAT_FONT_FAMILY, CHAT_FONT_SIZE, GLOBAL_COLLECTION_ID, GLOBAL_CONTEXT_DISPLAY_NAME
)
from .widgets import load_icon

logger = logging.getLogger(__name__)


class LeftControlPanel(QWidget):
    newSessionClicked = pyqtSignal()
    manageSessionsClicked = pyqtSignal()
    uploadFileClicked = pyqtSignal()
    uploadDirectoryClicked = pyqtSignal()
    uploadGlobalClicked = pyqtSignal()
    editPersonalityClicked = pyqtSignal()
    viewCodeBlocksClicked = pyqtSignal()
    viewRagContentClicked = pyqtSignal()
    modelSelected = pyqtSignal(str)
    projectSelected = pyqtSignal(str)
    newProjectClicked = pyqtSignal()
    temperatureChanged = pyqtSignal(float)

    PROJECT_ID_ROLE = Qt.ItemDataRole.UserRole + 1
    TEMP_SLIDER_MIN = 0
    TEMP_SLIDER_MAX = 200
    TEMP_PRECISION_FACTOR = 100.0

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LeftControlPanel")
        self._active_project_id_in_lcp: str = GLOBAL_COLLECTION_ID
        self._is_programmatic_selection: bool = False
        self._is_programmatic_temp_change: bool = False
        self._projects_inventory: Dict[str, str] = {}
        self.project_item_tree_icon = load_icon("new_folder_icon.svg")
        self.global_context_tree_icon = QIcon()
        self._init_widgets()
        self._init_layout()
        self._connect_signals()
        self.set_temperature_ui(0.7)

    def _get_qta_icon(self, icon_name: str, color: str = "#00CFE8") -> QIcon:
        if QTAWESOME_AVAILABLE:
            try:
                return qta.icon(icon_name, color=color)
            except Exception as e:
                logger.warning(f"Could not load qtawesome icon '{icon_name}': {e}")
        return QIcon()

    def _init_widgets(self):
        self.button_font = QFont(CHAT_FONT_FAMILY, CHAT_FONT_SIZE - 1)
        button_style_sheet = "QPushButton { text-align: left; padding: 6px 8px; }"
        button_icon_size = QSize(16, 16)

        self.projects_group = QGroupBox("PROJECTS")
        self.chat_sessions_group = QGroupBox("CHAT SESSIONS")
        self.project_knowledge_group = QGroupBox("KNOWLEDGE")
        self.global_knowledge_group = QGroupBox("GLOBAL KNOWLEDGE BASE")
        self.tools_settings_group = QGroupBox("TOOLS & SETTINGS")

        for group_box in [self.projects_group, self.chat_sessions_group,
                          self.project_knowledge_group, self.global_knowledge_group,
                          self.tools_settings_group]:
            group_box.setFont(QFont(CHAT_FONT_FAMILY, CHAT_FONT_SIZE - 1, QFont.Weight.Bold))
            group_box.setStyleSheet(
                "QGroupBox { margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")

        self.create_project_context_button = QPushButton(" Create New Project")
        self.create_project_context_button.setFont(self.button_font);
        self.create_project_context_button.setIcon(self._get_qta_icon('fa5s.folder-plus'));
        self.create_project_context_button.setToolTip("Create a new project workspace (Ctrl+Shift+N)");
        self.create_project_context_button.setObjectName("createProjectContextButton");
        self.create_project_context_button.setStyleSheet(button_style_sheet);
        self.create_project_context_button.setIconSize(button_icon_size)

        self.project_tree_view = QTreeView();
        self.project_tree_view.setObjectName("ProjectTreeView");
        self.project_tree_view.setToolTip("Select the active project context for RAG and chat history");
        self.project_tree_view.setHeaderHidden(True);
        self.project_tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection);
        self.project_tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);
        self.project_tree_view.setItemsExpandable(False);
        self.project_tree_view.setRootIsDecorated(False);
        self.project_tree_view.setIndentation(10);
        self.project_tree_view.setFont(self.button_font);
        self.project_tree_model = QStandardItemModel(self);
        self.project_tree_view.setModel(self.project_tree_model)

        self.new_chat_button = QPushButton(" New Chat")
        self.new_chat_button.setFont(self.button_font);
        self.new_chat_button.setIcon(self._get_qta_icon('fa5.comment'));
        self.new_chat_button.setToolTip("Start a new chat in the current project (Ctrl+N)");
        self.new_chat_button.setObjectName("newChatButton");
        self.new_chat_button.setStyleSheet(button_style_sheet);
        self.new_chat_button.setIconSize(button_icon_size)

        self.manage_chats_button = QPushButton(" Manage Chats");
        self.manage_chats_button.setFont(self.button_font);
        self.manage_chats_button.setIcon(self._get_qta_icon('fa5.folder-open'));
        self.manage_chats_button.setToolTip("Load, save, or delete chat sessions for the current project (Ctrl+O)");
        self.manage_chats_button.setObjectName("manageChatsButton");
        self.manage_chats_button.setStyleSheet(button_style_sheet);
        self.manage_chats_button.setIconSize(button_icon_size)

        self.add_files_button = QPushButton(" Add File(s)")
        self.add_files_button.setFont(self.button_font);
        self.add_files_button.setIcon(self._get_qta_icon('fa5s.file-medical'));
        self.add_files_button.setToolTip("Add files to this project's RAG knowledge base (Ctrl+U)");
        self.add_files_button.setObjectName("addFilesButton");
        self.add_files_button.setStyleSheet(button_style_sheet);
        self.add_files_button.setIconSize(button_icon_size)

        self.add_folder_button = QPushButton(" Add Folder");
        self.add_folder_button.setFont(self.button_font);
        self.add_folder_button.setIcon(self._get_qta_icon('fa5s.folder-plus'));
        self.add_folder_button.setToolTip("Add a folder to this project's RAG knowledge base (Ctrl+Shift+U)");
        self.add_folder_button.setObjectName("addFolderButton");
        self.add_folder_button.setStyleSheet(button_style_sheet);
        self.add_folder_button.setIconSize(button_icon_size)

        self.view_project_rag_button = QPushButton(" View Project RAG");
        self.view_project_rag_button.setFont(self.button_font);
        self.view_project_rag_button.setIcon(self._get_qta_icon('fa5s.database'));
        self.view_project_rag_button.setToolTip("View RAG content for the current project (Ctrl+R)");
        self.view_project_rag_button.setObjectName("viewProjectRagButton");
        self.view_project_rag_button.setStyleSheet(button_style_sheet);
        self.view_project_rag_button.setIconSize(button_icon_size)

        self.manage_global_knowledge_button = QPushButton(" Manage Global Knowledge")
        self.manage_global_knowledge_button.setFont(self.button_font);
        self.manage_global_knowledge_button.setIcon(self._get_qta_icon('fa5s.globe'));
        self.manage_global_knowledge_button.setToolTip("Upload to or manage the Global RAG knowledge base (Ctrl+G)");
        self.manage_global_knowledge_button.setObjectName("manageGlobalKnowledgeButton");
        self.manage_global_knowledge_button.setStyleSheet(button_style_sheet);
        self.manage_global_knowledge_button.setIconSize(button_icon_size)

        self.view_code_blocks_button = QPushButton(" View Code Blocks")
        self.view_code_blocks_button.setFont(self.button_font);
        self.view_code_blocks_button.setIcon(self._get_qta_icon('fa5s.code'));
        self.view_code_blocks_button.setToolTip("View code blocks from the current chat (Ctrl+B)");
        self.view_code_blocks_button.setObjectName("viewCodeBlocksButton");
        self.view_code_blocks_button.setStyleSheet(button_style_sheet);
        self.view_code_blocks_button.setIconSize(button_icon_size)

        self.configure_ai_personality_button = QPushButton(" Configure AI Persona")
        self.configure_ai_personality_button.setFont(self.button_font);
        self.configure_ai_personality_button.setIcon(self._get_qta_icon('fa5s.sliders-h'));
        self.configure_ai_personality_button.setToolTip("Customize the AI's personality and system prompt (Ctrl+P)");
        self.configure_ai_personality_button.setObjectName("configureAiPersonalityButton");
        self.configure_ai_personality_button.setStyleSheet(button_style_sheet);
        self.configure_ai_personality_button.setIconSize(button_icon_size)

        self.model_label = QLabel("Model:")
        self.model_label.setFont(self.button_font)
        self.model_selector = QComboBox()
        self.model_selector.setFont(self.button_font);
        self.model_selector.setObjectName("ModelSelector");
        self.model_selector.setToolTip("Select the AI model to use");
        self.model_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.temperature_label = QLabel("Temperature:")
        self.temperature_label.setFont(self.button_font)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(self.TEMP_SLIDER_MIN, self.TEMP_SLIDER_MAX)
        self.temperature_slider.setSingleStep(1);
        self.temperature_slider.setPageStep(10)
        self.temperature_slider.setToolTip("Adjust AI response creativity (0.0 to 2.0)")
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setRange(0.0, 2.0);
        self.temperature_spinbox.setSingleStep(0.01)
        self.temperature_spinbox.setDecimals(2);
        self.temperature_spinbox.setFont(self.button_font)
        self.temperature_spinbox.setFixedWidth(60)

        std_global_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon)
        if not std_global_icon.isNull():
            self.global_context_tree_icon = std_global_icon
        else:
            self.global_context_tree_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        if self.project_item_tree_icon.isNull(): self.project_item_tree_icon = self._get_qta_icon('fa5s.folder',
                                                                                                  color="#DAA520")

    def _init_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10);
        main_layout.setSpacing(10)
        projects_group_layout = QVBoxLayout(self.projects_group)
        projects_group_layout.setSpacing(5);
        projects_group_layout.addWidget(self.create_project_context_button);
        projects_group_layout.addWidget(self.project_tree_view)
        main_layout.addWidget(self.projects_group)
        chat_sessions_group_layout = QVBoxLayout(self.chat_sessions_group)
        chat_sessions_group_layout.setSpacing(5);
        chat_sessions_group_layout.addWidget(self.new_chat_button);
        chat_sessions_group_layout.addWidget(self.manage_chats_button)
        main_layout.addWidget(self.chat_sessions_group)
        project_knowledge_group_layout = QVBoxLayout(self.project_knowledge_group)
        project_knowledge_group_layout.setSpacing(5);
        project_knowledge_group_layout.addWidget(self.add_files_button);
        project_knowledge_group_layout.addWidget(self.add_folder_button);
        project_knowledge_group_layout.addWidget(self.view_project_rag_button)
        main_layout.addWidget(self.project_knowledge_group)
        global_knowledge_group_layout = QVBoxLayout(self.global_knowledge_group)
        global_knowledge_group_layout.setSpacing(5);
        global_knowledge_group_layout.addWidget(self.manage_global_knowledge_button)
        main_layout.addWidget(self.global_knowledge_group)
        tools_settings_group_layout = QVBoxLayout(self.tools_settings_group)
        tools_settings_group_layout.setSpacing(5)
        tools_settings_group_layout.addWidget(self.view_code_blocks_button)
        tools_settings_group_layout.addWidget(self.configure_ai_personality_button)
        tools_settings_group_layout.addWidget(self.model_label)
        tools_settings_group_layout.addWidget(self.model_selector)
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temperature_label)
        temp_layout.addWidget(self.temperature_slider, 1)
        temp_layout.addWidget(self.temperature_spinbox)
        tools_settings_group_layout.addLayout(temp_layout)
        main_layout.addWidget(self.tools_settings_group)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _connect_signals(self):
        self.create_project_context_button.clicked.connect(self.newProjectClicked)
        # --- CORRECTED LINE BELOW ---
        self.project_tree_view.selectionModel().currentChanged.connect(self._on_project_tree_item_changed)
        # --- END CORRECTION ---
        self.new_chat_button.clicked.connect(self.newSessionClicked)
        self.manage_chats_button.clicked.connect(self.manageSessionsClicked)
        self.add_files_button.clicked.connect(self.uploadFileClicked)
        self.add_folder_button.clicked.connect(self.uploadDirectoryClicked)
        self.view_project_rag_button.clicked.connect(self.viewRagContentClicked)
        self.manage_global_knowledge_button.clicked.connect(self.uploadGlobalClicked)
        self.view_code_blocks_button.clicked.connect(self.viewCodeBlocksClicked)
        self.configure_ai_personality_button.clicked.connect(self.editPersonalityClicked)
        self.model_selector.currentTextChanged.connect(self.modelSelected)
        self.temperature_slider.valueChanged.connect(self._on_slider_temp_changed)
        self.temperature_spinbox.valueChanged.connect(self._on_spinbox_temp_changed)

    # --- ADDED THE MISSING SLOT ---
    @pyqtSlot(QModelIndex, QModelIndex)
    def _on_project_tree_item_changed(self, current: QModelIndex, previous: QModelIndex):
        if self._is_programmatic_selection or not current.isValid():
            return
        item = self.project_tree_model.itemFromIndex(current)
        if item:
            project_id = item.data(self.PROJECT_ID_ROLE)
            if isinstance(project_id, str) and self._active_project_id_in_lcp != project_id:
                logger.debug(f"LeftPanel: User selected project ID '{project_id}' from TreeView.")
                self.projectSelected.emit(project_id)

    # --- END ADDED SLOT ---

    @pyqtSlot(int)
    def _on_slider_temp_changed(self, slider_value: int):
        if self._is_programmatic_temp_change: return
        float_value = slider_value / self.TEMP_PRECISION_FACTOR
        self._is_programmatic_temp_change = True
        self.temperature_spinbox.setValue(float_value)
        self._is_programmatic_temp_change = False
        self.temperatureChanged.emit(float_value)

    @pyqtSlot(float)
    def _on_spinbox_temp_changed(self, float_value: float):
        if self._is_programmatic_temp_change: return
        slider_value = int(float_value * self.TEMP_PRECISION_FACTOR)
        self._is_programmatic_temp_change = True
        self.temperature_slider.setValue(slider_value)
        self._is_programmatic_temp_change = False
        self.temperatureChanged.emit(float_value)

    def set_temperature_ui(self, temperature: float):
        if not (0.0 <= temperature <= 2.0):
            logger.warning(f"Attempted to set invalid temperature in UI: {temperature}")
            temperature = max(0.0, min(temperature, 2.0))
        self._is_programmatic_temp_change = True
        self.temperature_spinbox.setValue(temperature)
        self.temperature_slider.setValue(int(temperature * self.TEMP_PRECISION_FACTOR))
        self._is_programmatic_temp_change = False
        logger.debug(f"LeftPanel Temperature UI set to: {temperature}")

    def _get_project_display_name(self, project_id: str) -> str:
        if project_id == GLOBAL_COLLECTION_ID: return GLOBAL_CONTEXT_DISPLAY_NAME
        return self._projects_inventory.get(project_id, project_id)

    def _update_dynamic_group_titles(self):
        active_project_name = self._get_project_display_name(self._active_project_id_in_lcp)
        self.chat_sessions_group.setTitle(f"CHAT SESSIONS (for '{active_project_name}')")
        self.project_knowledge_group.setTitle(f"KNOWLEDGE FOR '{active_project_name}'")

    def _populate_project_tree_model(self, projects_dict: Dict[str, str]):
        self.project_tree_model.clear()
        global_item = QStandardItem(GLOBAL_CONTEXT_DISPLAY_NAME);
        if not self.global_context_tree_icon.isNull(): global_item.setIcon(self.global_context_tree_icon)
        global_item.setData(GLOBAL_COLLECTION_ID, self.PROJECT_ID_ROLE);
        global_item.setToolTip(f"Global foundational knowledge (ID: {GLOBAL_COLLECTION_ID})")
        self.project_tree_model.appendRow(global_item)
        sorted_projects = sorted(projects_dict.items(), key=lambda item_pair: item_pair[1].lower())
        for project_id, project_name in sorted_projects:
            if project_id == GLOBAL_COLLECTION_ID: continue
            project_item = QStandardItem(project_name)
            if not self.project_item_tree_icon.isNull(): project_item.setIcon(self.project_item_tree_icon)
            project_item.setData(project_id, self.PROJECT_ID_ROLE);
            project_item.setToolTip(f"Project: {project_name}\nID: {project_id}")
            self.project_tree_model.appendRow(project_item)

    @pyqtSlot(dict)
    def handle_project_inventory_update(self, projects_dict: Dict[str, str]):
        self._projects_inventory = projects_dict.copy();
        self._is_programmatic_selection = True
        self._populate_project_tree_model(projects_dict);
        self._select_project_item_in_tree(self._active_project_id_in_lcp)
        self._update_dynamic_group_titles();
        self._is_programmatic_selection = False

    @pyqtSlot(str)
    def handle_active_project_ui_update(self, active_project_id: str):
        if self._active_project_id_in_lcp == active_project_id and self.project_tree_view.currentIndex().isValid() and self.project_tree_model.itemFromIndex(
                self.project_tree_view.currentIndex()).data(self.PROJECT_ID_ROLE) == active_project_id:
            self._update_dynamic_group_titles();
            return
        self._active_project_id_in_lcp = active_project_id;
        self._is_programmatic_selection = True
        self._select_project_item_in_tree(active_project_id);
        self._update_dynamic_group_titles()
        self._is_programmatic_selection = False;
        self._update_rag_button_state()

    def _select_project_item_in_tree(self, project_id_to_select: str):
        target_id = project_id_to_select if project_id_to_select and project_id_to_select.strip() else GLOBAL_COLLECTION_ID
        found_item_index = QModelIndex()
        for i in range(self.project_tree_model.rowCount()):
            item = self.project_tree_model.item(i)
            if item and item.data(
                self.PROJECT_ID_ROLE) == target_id: found_item_index = self.project_tree_model.indexFromItem(
                item); break
        if found_item_index.isValid():
            if self.project_tree_view.currentIndex() != found_item_index: self.project_tree_view.setCurrentIndex(
                found_item_index)
        elif target_id != GLOBAL_COLLECTION_ID:
            self._select_project_item_in_tree(GLOBAL_COLLECTION_ID)

    def update_model_list(self, model_names: List[str], current_model_name: Optional[str]):
        self.model_selector.blockSignals(True);
        self.model_selector.clear()
        if not model_names:
            self.model_selector.addItem("No models available"); self.model_selector.setEnabled(False)
        else:
            self.model_selector.addItems(sorted(model_names, key=str.lower));
            self.model_selector.setEnabled(True)
            idx = self.model_selector.findText(current_model_name if current_model_name else "")
            if idx != -1: self.model_selector.setCurrentIndex(idx)
        self.model_selector.blockSignals(False)

    def update_model_selection(self, model_name: str):
        self.model_selector.blockSignals(True)
        idx = self.model_selector.findText(model_name)
        if idx != -1 and self.model_selector.currentIndex() != idx: self.model_selector.setCurrentIndex(idx)
        self.model_selector.blockSignals(False)

    def set_enabled_state(self, enabled: bool, is_busy: bool):
        effective_enabled = enabled and not is_busy
        self.create_project_context_button.setEnabled(enabled)
        self.new_chat_button.setEnabled(effective_enabled)
        self.manage_chats_button.setEnabled(effective_enabled)
        self.configure_ai_personality_button.setEnabled(effective_enabled)
        self.model_selector.setEnabled(enabled)
        self.model_label.setStyleSheet(f"QLabel {{ color: {'#CCCCCC' if enabled else '#777777'}; }}")
        self.project_tree_view.setEnabled(enabled)
        self.add_files_button.setEnabled(effective_enabled)
        self.add_folder_button.setEnabled(effective_enabled)
        self.manage_global_knowledge_button.setEnabled(effective_enabled)
        self.view_code_blocks_button.setEnabled(True)
        self.temperature_label.setEnabled(enabled)
        self.temperature_slider.setEnabled(effective_enabled)
        self.temperature_spinbox.setEnabled(effective_enabled)
        self.temperature_label.setStyleSheet(f"QLabel {{ color: {'#CCCCCC' if enabled else '#777777'}; }}")
        self._update_rag_button_state()

    def _update_rag_button_state(self):
        rag_context_ready = False
        try:
            main_window = self.parent()  # type: ignore
            if main_window and hasattr(main_window, 'chat_manager'):
                chat_manager = main_window.chat_manager  # type: ignore
                if chat_manager and hasattr(chat_manager, 'is_rag_context_initialized'):
                    rag_context_ready = chat_manager.is_rag_context_initialized(
                        self._active_project_id_in_lcp)  # type: ignore
        except Exception as e:
            logger.error(f"Error checking RAG context readiness in LeftPanel: {e}")
        self.view_project_rag_button.setEnabled(rag_context_ready)

    def update_personality_tooltip(self, active: bool):
        tooltip_base = "Customize the AI's personality and system prompt (Ctrl+P)"
        status = "(Active)" if active else "(Default)"
        self.configure_ai_personality_button.setToolTip(f"{tooltip_base}\nStatus: {status}")