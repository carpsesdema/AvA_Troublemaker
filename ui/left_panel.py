# ui/left_panel.py
import logging
from typing import List, Optional, Dict, Any # Added Any

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
    CHAT_FONT_FAMILY, CHAT_FONT_SIZE, GLOBAL_COLLECTION_ID, GLOBAL_CONTEXT_DISPLAY_NAME,
    GENERATOR_BACKEND_ID # For specialized LLM
)
from .widgets import load_icon
# from core.chat_manager import ChatManager # Placeholder for type hint

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
    projectSelected = pyqtSignal(str)
    newProjectClicked = pyqtSignal()
    temperatureChanged = pyqtSignal(float)

    PROJECT_ID_ROLE = Qt.ItemDataRole.UserRole + 1
    TEMP_SLIDER_MIN = 0
    TEMP_SLIDER_MAX = 200
    TEMP_PRECISION_FACTOR = 100.0

    # UserData role for the new comboboxes - KEEPING THIS DEFINITION
    # but we'll use the default UserRole for QComboBox.addItem's userData parameter
    MODEL_CONFIG_DATA_ROLE = Qt.ItemDataRole.UserRole + 2

    def __init__(self, chat_manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LeftControlPanel")
        self.chat_manager = chat_manager

        self._active_project_id_in_lcp: str = GLOBAL_COLLECTION_ID
        self._is_programmatic_selection: bool = False
        self._is_programmatic_temp_change: bool = False
        self._is_programmatic_model_change: bool = False
        self._projects_inventory: Dict[str, str] = {}
        self.project_item_tree_icon = load_icon("new_folder_icon.svg")
        self.global_context_tree_icon = QIcon()

        self._init_widgets()
        self._init_layout()
        self._connect_signals()
        self.set_temperature_ui(0.7)
        self._load_initial_model_settings()

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

        self.chat_llm_label = QLabel("Chat LLM:")
        self.chat_llm_label.setFont(self.button_font)
        self.chat_llm_combo_box = QComboBox()
        self.chat_llm_combo_box.setFont(self.button_font)
        self.chat_llm_combo_box.setObjectName("ChatLlmComboBox")
        self.chat_llm_combo_box.setToolTip("Select the main AI for chat conversations")
        self.chat_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.specialized_llm_label = QLabel("Specialized LLM:")
        self.specialized_llm_label.setFont(self.button_font)
        self.specialized_llm_combo_box = QComboBox()
        self.specialized_llm_combo_box.setFont(self.button_font)
        self.specialized_llm_combo_box.setObjectName("SpecializedLlmComboBox")
        self.specialized_llm_combo_box.setToolTip("Select the AI for specialized tasks (e.g., code generation)")
        self.specialized_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        if self.project_item_tree_icon.isNull(): self.project_item_tree_icon = self._get_qta_icon('fa5s.folder', color="#DAA520")

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
        tools_settings_group_layout.addWidget(self.chat_llm_label)
        tools_settings_group_layout.addWidget(self.chat_llm_combo_box)
        tools_settings_group_layout.addWidget(self.specialized_llm_label)
        tools_settings_group_layout.addWidget(self.specialized_llm_combo_box)
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
        self.project_tree_view.selectionModel().currentChanged.connect(self._on_project_tree_item_changed)
        self.new_chat_button.clicked.connect(self.newSessionClicked)
        self.manage_chats_button.clicked.connect(self.manageSessionsClicked)
        self.add_files_button.clicked.connect(self.uploadFileClicked)
        self.add_folder_button.clicked.connect(self.uploadDirectoryClicked)
        self.view_project_rag_button.clicked.connect(self.viewRagContentClicked)
        self.manage_global_knowledge_button.clicked.connect(self.uploadGlobalClicked)
        self.view_code_blocks_button.clicked.connect(self.viewCodeBlocksClicked)
        self.configure_ai_personality_button.clicked.connect(self.editPersonalityClicked)

        self.chat_llm_combo_box.currentIndexChanged.connect(self._on_chat_llm_selected)
        self.specialized_llm_combo_box.currentIndexChanged.connect(self._on_specialized_llm_selected)

        self.temperature_slider.valueChanged.connect(self._on_slider_temp_changed)
        self.temperature_spinbox.valueChanged.connect(self._on_spinbox_temp_changed)
        if self.chat_manager:
            self.chat_manager.backend_config_state_changed.connect(self._handle_chat_manager_config_change)
            self.chat_manager.available_models_changed_for_backend.connect(self._handle_chat_manager_models_update)

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

    def _load_initial_model_settings(self):
        logger.debug("LeftPanel: Loading initial model settings from ChatManager.")
        self._is_programmatic_model_change = True

        self.chat_llm_combo_box.blockSignals(True)
        self._populate_chat_llm_combo_box()
        self.chat_llm_combo_box.blockSignals(False)
        active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()
        active_chat_model_name = self.chat_manager.get_model_for_backend(active_chat_backend_id)
        self._set_combo_box_selection(self.chat_llm_combo_box, active_chat_backend_id, active_chat_model_name)

        self.specialized_llm_combo_box.blockSignals(True)
        self._populate_specialized_llm_combo_box()
        self.specialized_llm_combo_box.blockSignals(False)
        active_specialized_model_name = self.chat_manager.get_model_for_backend(GENERATOR_BACKEND_ID)
        self._set_combo_box_selection(self.specialized_llm_combo_box, GENERATOR_BACKEND_ID, active_specialized_model_name)

        self._is_programmatic_model_change = False
        logger.debug("LeftPanel: Initial model settings loaded.")

    def _set_combo_box_selection(self, combo_box: QComboBox, target_backend_id: str, target_model_name: Optional[str]):
        if target_model_name is None:
            if combo_box.count() > 0:
                combo_box.setCurrentIndex(0)
                # VVVV Use default Qt.ItemDataRole.UserRole VVVV
                new_selection_data = combo_box.currentData() # Or combo_box.currentData(Qt.ItemDataRole.UserRole)
                # ^^^^ End Change ^^^^
                if new_selection_data and isinstance(new_selection_data, dict):
                    if combo_box == self.chat_llm_combo_box:
                        self.chat_manager.set_active_chat_backend(new_selection_data["backend_id"])
                        self.chat_manager.set_model_for_backend(new_selection_data["backend_id"], new_selection_data["model_name"])
                    elif combo_box == self.specialized_llm_combo_box:
                        self.chat_manager.set_model_for_backend(GENERATOR_BACKEND_ID, new_selection_data["model_name"])
            return

        for i in range(combo_box.count()):
            # VVVV Use default Qt.ItemDataRole.UserRole VVVV
            item_data = combo_box.itemData(i) # Or combo_box.itemData(i, Qt.ItemDataRole.UserRole)
            # ^^^^ End Change ^^^^
            if isinstance(item_data, dict) and \
               item_data.get("backend_id") == target_backend_id and \
               item_data.get("model_name") == target_model_name:
                if combo_box.currentIndex() != i:
                    combo_box.setCurrentIndex(i)
                return
        if combo_box.count() > 0:
            combo_box.setCurrentIndex(0)
            logger.warning(f"Could not find '{target_backend_id}':'{target_model_name}' in {combo_box.objectName()}. Selecting first available.")
            # VVVV Use default Qt.ItemDataRole.UserRole VVVV
            new_selection_data = combo_box.currentData()
            # ^^^^ End Change ^^^^
            if new_selection_data and isinstance(new_selection_data, dict):
                if combo_box == self.chat_llm_combo_box:
                     if self.chat_manager.get_current_active_chat_backend_id() != new_selection_data["backend_id"] or \
                        self.chat_manager.get_model_for_backend(new_selection_data["backend_id"]) != new_selection_data["model_name"]:
                         self.chat_manager.set_active_chat_backend(new_selection_data["backend_id"])
                         self.chat_manager.set_model_for_backend(new_selection_data["backend_id"], new_selection_data["model_name"])
                elif combo_box == self.specialized_llm_combo_box:
                     if self.chat_manager.get_model_for_backend(GENERATOR_BACKEND_ID) != new_selection_data["model_name"]:
                         self.chat_manager.set_model_for_backend(GENERATOR_BACKEND_ID, new_selection_data["model_name"])


    def _populate_chat_llm_combo_box(self):
        self.chat_llm_combo_box.blockSignals(True)
        self.chat_llm_combo_box.clear()
        try:
            models_details = self.chat_manager.get_all_available_chat_models_with_details()
            if models_details:
                for detail in models_details:
                    self.chat_llm_combo_box.addItem(detail['display_name'], userData=detail)
                self.chat_llm_combo_box.setEnabled(True)
            else:
                self.chat_llm_combo_box.addItem("No Chat LLMs Available")
                self.chat_llm_combo_box.setEnabled(False)
        except Exception as e:
            logger.exception("Error populating Chat LLM combo box:")
            self.chat_llm_combo_box.addItem("Error Loading LLMs")
            self.chat_llm_combo_box.setEnabled(False)
        self.chat_llm_combo_box.blockSignals(False)

    def _populate_specialized_llm_combo_box(self):
        self.specialized_llm_combo_box.blockSignals(True)
        self.specialized_llm_combo_box.clear()
        try:
            models_details = self.chat_manager.get_all_available_specialized_models_with_details()
            if models_details:
                for detail in models_details:
                    self.specialized_llm_combo_box.addItem(detail['display_name'], userData=detail)
                self.specialized_llm_combo_box.setEnabled(True)
            else:
                self.specialized_llm_combo_box.addItem("No Specialized LLMs Available")
                self.specialized_llm_combo_box.setEnabled(False)
        except Exception as e:
            logger.exception("Error populating Specialized LLM combo box:")
            self.specialized_llm_combo_box.addItem("Error Loading LLMs")
            self.specialized_llm_combo_box.setEnabled(False)
        self.specialized_llm_combo_box.blockSignals(False)


    @pyqtSlot(int)
    def _on_chat_llm_selected(self, index: int):
        if self._is_programmatic_model_change or index < 0:
            return

        # VVVV Use default Qt.ItemDataRole.UserRole VVVV
        selected_data = self.chat_llm_combo_box.itemData(index) # Or itemData(index, Qt.ItemDataRole.UserRole)
        # ^^^^ End Change ^^^^

        if selected_data is None:
            logger.warning(f"Chat LLM selection changed to index {index}, but itemData is None. "
                           f"Text: '{self.chat_llm_combo_box.itemText(index)}'. Ignoring.")
            return

        if isinstance(selected_data, dict) and "backend_id" in selected_data and "model_name" in selected_data:
            backend_id = selected_data["backend_id"]
            model_name = selected_data["model_name"]
            display_name = self.chat_llm_combo_box.itemText(index)
            logger.info(f"LeftPanel: Chat LLM selected: '{display_name}' (Backend: {backend_id}, Model: {model_name})")

            self.chat_manager.set_active_chat_backend(backend_id)
            self.chat_manager.set_model_for_backend(backend_id, model_name)
        else:
            logger.warning(f"Chat LLM selection changed to an item with invalid data structure: {selected_data}")

    @pyqtSlot(int)
    def _on_specialized_llm_selected(self, index: int):
        if self._is_programmatic_model_change or index < 0:
            return

        # VVVV Use default Qt.ItemDataRole.UserRole VVVV
        selected_data = self.specialized_llm_combo_box.itemData(index) # Or itemData(index, Qt.ItemDataRole.UserRole)
        # ^^^^ End Change ^^^^

        if selected_data is None:
            logger.warning(f"Specialized LLM selection changed to index {index}, but itemData is None. "
                           f"Text: '{self.specialized_llm_combo_box.itemText(index)}'. Ignoring.")
            return

        if isinstance(selected_data, dict) and "backend_id" in selected_data and "model_name" in selected_data:
            model_name = selected_data["model_name"]
            display_name = self.specialized_llm_combo_box.itemText(index)
            logger.info(f"LeftPanel: Specialized LLM selected: '{display_name}' (Model: {model_name} for Generator Backend)")

            self.chat_manager.set_model_for_backend(GENERATOR_BACKEND_ID, model_name)
        else:
            logger.warning(f"Specialized LLM selection changed to an item with invalid data structure: {selected_data}")


    @pyqtSlot(str, str, bool, bool)
    def _handle_chat_manager_config_change(self, backend_id: str, model_name: str, is_configured: bool, personality_is_active: bool):
        logger.debug(f"LCP: Received config change from CM. Backend='{backend_id}', Model='{model_name}', ConfigOK={is_configured}, PersActive={personality_is_active}")
        self._is_programmatic_model_change = True

        active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()
        active_chat_model_name = self.chat_manager.get_model_for_backend(active_chat_backend_id)

        if backend_id == active_chat_backend_id:
            self.chat_llm_combo_box.blockSignals(True)
            self._populate_chat_llm_combo_box()
            self.chat_llm_combo_box.blockSignals(False)
            self._set_combo_box_selection(self.chat_llm_combo_box, active_chat_backend_id, active_chat_model_name)
            self.update_personality_tooltip(personality_is_active)

        if backend_id == GENERATOR_BACKEND_ID:
            active_generator_model_name = self.chat_manager.get_model_for_backend(GENERATOR_BACKEND_ID)
            self.specialized_llm_combo_box.blockSignals(True)
            self._populate_specialized_llm_combo_box()
            self.specialized_llm_combo_box.blockSignals(False)
            self._set_combo_box_selection(self.specialized_llm_combo_box, GENERATOR_BACKEND_ID, active_generator_model_name)

        # VVVV Use default Qt.ItemDataRole.UserRole VVVV
        current_chat_llm_data = self.chat_llm_combo_box.currentData()
        # ^^^^ End Change ^^^^
        if current_chat_llm_data and (current_chat_llm_data.get("backend_id") != active_chat_backend_id or current_chat_llm_data.get("model_name") != active_chat_model_name):
             self._set_combo_box_selection(self.chat_llm_combo_box, active_chat_backend_id, active_chat_model_name)

        # VVVV Use default Qt.ItemDataRole.UserRole VVVV
        current_spec_llm_data = self.specialized_llm_combo_box.currentData()
        # ^^^^ End Change ^^^^
        active_generator_model_name_check = self.chat_manager.get_model_for_backend(GENERATOR_BACKEND_ID)
        if current_spec_llm_data and (current_spec_llm_data.get("backend_id") != GENERATOR_BACKEND_ID or current_spec_llm_data.get("model_name") != active_generator_model_name_check):
            self._set_combo_box_selection(self.specialized_llm_combo_box, GENERATOR_BACKEND_ID, active_generator_model_name_check)

        self._is_programmatic_model_change = False


    @pyqtSlot(str, list)
    def _handle_chat_manager_models_update(self, backend_id: str, models: List[str]):
        logger.debug(f"LCP: Received model list update from CM for backend '{backend_id}'. Models: {models}")
        self._is_programmatic_model_change = True
        active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()

        if backend_id == active_chat_backend_id:
            logger.debug(f"Updating Chat LLM combo box due to model list change for active backend '{backend_id}'")
            current_model_name = self.chat_manager.get_model_for_backend(backend_id)
            self.chat_llm_combo_box.blockSignals(True)
            self._populate_chat_llm_combo_box()
            self.chat_llm_combo_box.blockSignals(False)
            self._set_combo_box_selection(self.chat_llm_combo_box, backend_id, current_model_name)

        if backend_id == GENERATOR_BACKEND_ID:
            logger.debug(f"Updating Specialized LLM combo box due to model list change for generator backend '{backend_id}'")
            current_model_name = self.chat_manager.get_model_for_backend(backend_id)
            self.specialized_llm_combo_box.blockSignals(True)
            self._populate_specialized_llm_combo_box()
            self.specialized_llm_combo_box.blockSignals(False)
            self._set_combo_box_selection(self.specialized_llm_combo_box, backend_id, current_model_name)
        self._is_programmatic_model_change = False

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

    def set_enabled_state(self, enabled: bool, is_busy: bool):
        effective_enabled = enabled and not is_busy
        self.create_project_context_button.setEnabled(enabled)
        self.new_chat_button.setEnabled(effective_enabled)
        self.manage_chats_button.setEnabled(effective_enabled)
        self.configure_ai_personality_button.setEnabled(effective_enabled)

        self.chat_llm_combo_box.setEnabled(enabled)
        self.specialized_llm_combo_box.setEnabled(enabled)

        self.chat_llm_label.setStyleSheet(f"QLabel {{ color: {'#CCCCCC' if enabled else '#777777'}; }}")
        self.specialized_llm_label.setStyleSheet(f"QLabel {{ color: {'#CCCCCC' if enabled else '#777777'}; }}")

        self.project_tree_view.setEnabled(enabled)
        self.add_files_button.setEnabled(effective_enabled)
        self.add_folder_button.setEnabled(effective_enabled)
        self.manage_global_knowledge_button.setEnabled(effective_enabled)
        self.view_code_blocks_button.setEnabled(True)
        self.temperature_label.setEnabled(enabled)
        self.temperature_slider.setEnabled(effective_enabled)
        self.temperature_spinbox.setEnabled(effective_enabled)
        self.temperature_label.setStyleSheet(f"QLabel {{ color: {'#CCCCCC' if enabled else '#777777'}; }}")

    def update_personality_tooltip(self, active: bool):
        tooltip_base = "Customize the AI's personality and system prompt (Ctrl+P)"
        status = "(Custom Persona Active)" if active else "(Default Persona)"
        self.configure_ai_personality_button.setToolTip(f"{tooltip_base}\nStatus: {status}")