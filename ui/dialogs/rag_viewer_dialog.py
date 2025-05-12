# SynAva_1.0/ui/dialogs/rag_viewer_dialog.py
# Contains the RAGViewerDialog class, extracted from the original dialogs.py
# Includes multi-select functionality.

import logging
import os
from collections import defaultdict
from typing import Optional, List, Dict, Any

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QSplitter,
    QApplication, QMenu, QLabel, QWidget, QAbstractItemView, QTreeWidget,
    QTreeWidgetItem, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot, QPoint
from PyQt6.QtGui import QFont, QClipboard, QIcon, QTextOption, QAction

# --- Local Imports ---
# Import necessary components from sibling modules or parent directories
# Adjust paths based on your final structure if dialogs are moved deeper
try:
    # --- MODIFIED IMPORTS ---
    from utils import constants # Was from ...utils
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE, GLOBAL_COLLECTION_ID # Was from ...utils.constants
    from services.vector_db_service import VectorDBService # Was from ...services.vector_db_service
    from ui.widgets import COPY_ICON, CHECK_ICON # Was from ..widgets
    # --- END MODIFIED IMPORTS ---
except ImportError as e:
    # Fallback for potential structure issues during refactoring
    logging.error(f"Error importing dependencies in rag_viewer_dialog.py: {e}. Check relative paths.")
    # Define dummy values if needed for the script to be syntactically valid
    class constants: CHAT_FONT_FAMILY="Arial"; CHAT_FONT_SIZE=10; GLOBAL_COLLECTION_ID="global_collection" # type: ignore
    class VectorDBService: pass # type: ignore
    COPY_ICON, CHECK_ICON = QIcon(), QIcon()


logger = logging.getLogger(__name__)

# --- RAG Viewer Dialog ---
class RAGViewerDialog(QDialog):
    """
    Dialog to view indexed RAG documents and chunks for a selected collection.
    Supports multi-selection of files for setting chat focus.
    """
    # Constants for data roles in the tree widget
    IS_DOCUMENT_ROLE = Qt.ItemDataRole.UserRole + 1
    FILEPATH_ROLE = Qt.ItemDataRole.UserRole + 2
    CHUNK_INDEX_ROLE = Qt.ItemDataRole.UserRole + 3
    COLLECTION_ID_ROLE = Qt.ItemDataRole.UserRole + 4

    # Signal emits a list of file paths when focus is requested
    focusRequested = pyqtSignal(list)

    def __init__(self, vector_db_service: VectorDBService, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if not vector_db_service or not isinstance(vector_db_service, VectorDBService):
            raise ValueError("RAGViewerDialog requires a valid VectorDBService instance.")
        self._vector_db_service = vector_db_service
        self.setWindowTitle("RAG Content Viewer")
        self.setObjectName("RAGViewerDialog")
        self.setMinimumSize(800, 600)
        self.setModal(True) # Keep it modal for now

        # Internal state
        self._current_collection_metadata: List[Dict[str, Any]] = []
        self._available_collections: List[str] = []

        # --- UI Setup ---
        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initialize the widgets for the dialog."""
        content_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
        label_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)

        # Top Controls
        self.info_label = QLabel("Indexed Documents and Chunks:")
        self.info_label.setFont(label_font)
        self.collection_label = QLabel("Collection:")
        self.collection_label.setFont(label_font)
        self.collection_selector = QComboBox()
        self.collection_selector.setFont(label_font)
        self.collection_selector.setObjectName("RAGCollectionSelector")
        self.collection_selector.setToolTip("Select RAG collection")
        self.collection_selector.setMinimumWidth(150)

        # Splitter and Panes
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setObjectName("RAGTreeWidget")
        self.tree_widget.setHeaderLabels(["Indexed Item", "Collection", "Chunks/Details"])
        self.tree_widget.setColumnWidth(0, 300)
        self.tree_widget.setColumnWidth(1, 120)
        self.tree_widget.setColumnWidth(2, 150)
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # Enable extended selection (Shift+Click, Ctrl+Click)
        self.tree_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.content_edit = QTextEdit()
        self.content_edit.setObjectName("RAGContentViewerEdit")
        self.content_edit.setReadOnly(True)
        self.content_edit.setFont(content_font)
        self.content_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        # Bottom Buttons
        self.copy_chunk_button = QPushButton(" Copy Chunk")
        self.copy_chunk_button.setToolTip("Copy selected chunk content")
        if not COPY_ICON.isNull(): self.copy_chunk_button.setIcon(COPY_ICON)
        self.copy_chunk_button.setEnabled(False)

        self.close_button = QPushButton("Close Viewer") # Changed from "Refresh"
        self.close_button.setToolTip("Close this viewer")

    def _init_layout(self):
        """Set up the layout for the dialog."""
        layout = QVBoxLayout(self)

        # Top Controls Layout
        top_controls_layout = QHBoxLayout()
        top_controls_layout.addWidget(self.info_label)
        top_controls_layout.addStretch()
        top_controls_layout.addWidget(self.collection_label)
        top_controls_layout.addWidget(self.collection_selector)
        layout.addLayout(top_controls_layout)

        # Splitter Layout
        self.splitter.addWidget(self.tree_widget)
        self.splitter.addWidget(self.content_edit)
        self.splitter.setSizes([450, 350]) # Initial sizes
        layout.addWidget(self.splitter, 1) # Allow splitter to stretch

        # Bottom Button Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.copy_chunk_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.collection_selector.currentIndexChanged.connect(self._load_selected_collection_data)
        self.tree_widget.itemSelectionChanged.connect(self._display_selected_content)
        self.tree_widget.customContextMenuRequested.connect(self._show_rag_item_context_menu)
        self.copy_chunk_button.clicked.connect(self._copy_selected_chunk_with_feedback)
        self.close_button.clicked.connect(self.accept) # Close dialog

    # --- Context Menu Logic ---
    @pyqtSlot(QPoint)
    def _show_rag_item_context_menu(self, pos: QPoint):
        """Shows context menu for RAG tree items, handles multiple selections for focus."""
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            return

        menu = QMenu(self)
        paths_to_focus = set() # Use a set to store unique file paths

        # Collect all unique parent document paths from the selection.
        for item in selected_items:
            is_doc = item.data(0, self.IS_DOCUMENT_ROLE)
            fp = item.data(0, self.FILEPATH_ROLE)
            if is_doc and fp: # If it's a document item itself
                paths_to_focus.add(fp)
            elif not is_doc: # If it's a chunk item
                parent = item.parent()
                if parent:
                    parent_fp = parent.data(0, self.FILEPATH_ROLE)
                    if parent_fp:
                        paths_to_focus.add(parent_fp)

        # Add "Set Focus" action if any valid paths were found
        if paths_to_focus:
            focus_label_text = ""
            if len(paths_to_focus) == 1:
                single_path = list(paths_to_focus)[0]
                is_file = '.' in os.path.basename(single_path) if single_path else False
                type_lbl = "File" if is_file else "Directory"
                disp_name = os.path.basename(single_path) or single_path
                focus_label_text = f"Set Chat Focus ({type_lbl}): {disp_name}"
            else:
                focus_label_text = f"Set Chat Focus on {len(paths_to_focus)} items"

            focus_action = menu.addAction(focus_label_text)
            # Connect to _emit_focus_request with the list of paths
            focus_action.triggered.connect(lambda checked=False, paths=list(paths_to_focus): self._emit_focus_request(paths))

        # Add "Copy Path" action only if exactly one item is selected (for clarity)
        item_at_pos = self.tree_widget.itemAt(pos) # Item directly under cursor
        if item_at_pos and len(selected_items) == 1 and item_at_pos == selected_items[0]:
            fp_at_pos = item_at_pos.data(0, self.FILEPATH_ROLE)
            # If it's a chunk, get the parent's path
            if not fp_at_pos and not item_at_pos.data(0, self.IS_DOCUMENT_ROLE):
                parent = item_at_pos.parent()
                fp_at_pos = parent.data(0, self.FILEPATH_ROLE) if parent else None

            if fp_at_pos:
                disp_name_at_pos = os.path.basename(fp_at_pos) or fp_at_pos
                copy_action = menu.addAction(f"Copy Path: {disp_name_at_pos}")
                copy_action.triggered.connect(lambda checked=False, p=fp_at_pos: self._copy_path(p))

        # Execute menu only if actions were added
        if menu.actions():
            menu.exec(self.tree_widget.mapToGlobal(pos))
        else:
            logger.debug("Context menu requested on item(s) without valid actions (e.g., no path data).")

    # --- Signal Emission & Copy Logic ---
    def _emit_focus_request(self, paths: List[str]):
        """Emits the focusRequested signal with a list of file paths."""
        if paths and isinstance(paths, list) and all(isinstance(p, str) for p in paths):
            unique_paths = sorted(list(set(paths))) # Ensure unique and sorted
            logger.info(f"RAG Viewer emitting focus request for paths: {unique_paths}")
            self.focusRequested.emit(unique_paths)
            self.accept() # Close dialog after setting focus
        else:
            logger.warning(f"Invalid paths provided for focus request: {paths}")

    def _copy_path(self, path: str):
        """Copies the provided file path to the clipboard."""
        if path and isinstance(path, str):
            try:
                cb = QApplication.clipboard()
                if cb:
                    cb.setText(path)
                    logger.info(f"Path copied to clipboard: {path}")
                else:
                    logger.error("Could not access clipboard to copy path.")
            except Exception as e:
                logger.exception(f"Error copying path to clipboard: {e}")
        else:
            logger.warning(f"Invalid path provided for copying: {path}")

    # --- Data Loading and Display Logic ---
    def populate_collection_selector(self):
        """Populates the collection dropdown."""
        logger.debug("Populating RAG collection selector...")
        try:
            self._available_collections = self._vector_db_service.get_available_collections()
            self._available_collections.sort(key=lambda x: (x != constants.GLOBAL_COLLECTION_ID, x.lower())) # Global first, then alpha
        except Exception as e:
            logger.exception("Error getting available RAG collections.")
            self._available_collections = []

        self.collection_selector.blockSignals(True)
        self.collection_selector.clear()

        if not self._available_collections:
            self.collection_selector.addItem("No collections available")
            self.collection_selector.setEnabled(False)
            logger.warning("No RAG collections available.")
        else:
            self.collection_selector.setEnabled(True)
            default_idx = 0
            for i, cid in enumerate(self._available_collections):
                # Display "Global" for the global collection ID, otherwise use the ID itself
                display_name = "Global" if cid == constants.GLOBAL_COLLECTION_ID else cid
                self.collection_selector.addItem(display_name, cid) # Store actual CID in UserRole data
                if cid == constants.GLOBAL_COLLECTION_ID: # Select Global by default
                    default_idx = i
            self.collection_selector.setCurrentIndex(default_idx)
            logger.info(f"Populated RAG selector. Selected: '{self.collection_selector.currentText()}' (CID: {self.collection_selector.itemData(default_idx)})")

        self.collection_selector.blockSignals(False)

        # Trigger data load for the default selected collection after selector is populated
        if self._available_collections:
            self._load_selected_collection_data(self.collection_selector.currentIndex())
        else:
            # Ensure tree is cleared if no collections exist
            self._populate_tree_widget("")


    @pyqtSlot(int)
    def _load_selected_collection_data(self, index: int):
        """Loads metadata for the selected collection and populates the tree."""
        # Clear previous state
        self.tree_widget.clear()
        self.content_edit.clear()
        self._current_collection_metadata = []
        self.copy_chunk_button.setEnabled(False)

        if index < 0 or index >= self.collection_selector.count():
            self.info_label.setText("Indexed Documents and Chunks:")
            logger.warning("Invalid index selected in collection selector.")
            return

        # Get the actual Collection ID stored in the item's UserRole data
        selected_cid = self.collection_selector.itemData(index, Qt.ItemDataRole.UserRole)

        # Validate the retrieved collection ID
        if not isinstance(selected_cid, str):
             # Handle cases like the "No collections available" item
            if selected_cid is None and self.collection_selector.itemText(index) == "No collections available":
                 logger.warning("No RAG collections available to load.")
                 self.info_label.setText("No RAG collections available.")
            else:
                 logger.error(f"Invalid collection ID type for index {index}. Data: '{selected_cid}' (Type: {type(selected_cid)})")
                 self.info_label.setText("Error: Invalid collection selected.")
            self._populate_tree_widget("") # Ensure tree is cleared
            return
        # Empty string represents the Global collection internally if needed, but should have GLOBAL_COLLECTION_ID
        if not selected_cid.strip() and selected_cid != constants.GLOBAL_COLLECTION_ID:
             logger.error(f"Empty or invalid collection ID string for index {index}.")
             self.info_label.setText("Error: Invalid collection ID.")
             self._populate_tree_widget("") # Ensure tree is cleared
             return


        logger.info(f"Loading data for RAG collection: '{selected_cid}'")
        display_name = self.collection_selector.itemText(index) # Get display name for UI

        try:
            if self._vector_db_service.is_ready(selected_cid):
                self._current_collection_metadata = self._vector_db_service.get_all_metadata(selected_cid)
                count = len(self._current_collection_metadata)
                self.info_label.setText(f"Collection '{display_name}' - Found {count} items:")
                self._populate_tree_widget(selected_cid)
            else:
                self.info_label.setText(f"Collection '{display_name}' - Not ready or empty.")
                logger.warning(f"Cannot load metadata for collection '{selected_cid}': Service reports not ready.")
                self._populate_tree_widget(selected_cid) # Populate (which will show empty)
        except Exception as e:
            logger.exception(f"Error loading metadata for collection '{selected_cid}': {e}")
            self.info_label.setText(f"Error loading collection '{display_name}'.")
            self._populate_tree_widget(selected_cid) # Populate (which will show empty)

    def _populate_tree_widget(self, current_cid: str):
        """Populates the tree widget with documents and chunks."""
        self.tree_widget.clear()
        self.content_edit.clear()
        self.copy_chunk_button.setEnabled(False)

        if not self._current_collection_metadata:
            logger.info(f"No metadata to populate tree for collection '{current_cid}'.")
            return

        # Group metadata indices by source document path
        docs = defaultdict(list)
        for i, meta in enumerate(self._current_collection_metadata):
            if isinstance(meta, dict):
                # Use 'source' key, default to 'Unknown_Source' if missing
                source = meta.get("source", "Unknown_Source")
                docs[source].append(i) # Store the index in the metadata list
            else:
                logger.warning(f"Skipping invalid metadata item at index {i} in collection '{current_cid}'. Type: {type(meta)}")

        # Populate the tree
        for source, chunk_indices in sorted(docs.items()):
            filename_display = os.path.basename(source) if source and source != "Unknown_Source" else "Unknown_Source"
            doc_item = QTreeWidgetItem(self.tree_widget)
            doc_item.setText(0, filename_display) # Column 0: Filename
            doc_item.setText(1, current_cid)      # Column 1: Collection ID
            doc_item.setText(2, f"{len(chunk_indices)} chunks") # Column 2: Chunk count
            doc_item.setToolTip(0, source)        # Tooltip shows full path

            # Store data roles for identification and path retrieval
            doc_item.setData(0, self.IS_DOCUMENT_ROLE, True)
            doc_item.setData(0, self.FILEPATH_ROLE, source)
            doc_item.setData(0, self.COLLECTION_ID_ROLE, current_cid)

            # Add child items for each chunk
            for metadata_list_index in sorted(chunk_indices):
                # Check if index is valid and metadata is a dictionary
                if 0 <= metadata_list_index < len(self._current_collection_metadata) and isinstance(self._current_collection_metadata[metadata_list_index], dict):
                    meta_chunk = self._current_collection_metadata[metadata_list_index]
                    chunk_display_index = meta_chunk.get("chunk_index", "?") # Index within its original file
                    start_char_index = meta_chunk.get("start_index", "?")

                    chunk_label = f"Chunk {chunk_display_index}"
                    chunk_details = f"Start Char: {start_char_index}"
                    # Add line numbers if available
                    if 'start_line' in meta_chunk and 'end_line' in meta_chunk:
                        chunk_details += f", Lines: {meta_chunk['start_line']}-{meta_chunk['end_line']}"

                    chunk_item = QTreeWidgetItem(doc_item)
                    chunk_item.setText(0, chunk_label) # Column 0: Chunk Label
                    chunk_item.setText(1, "")          # Column 1: Empty for chunk
                    chunk_item.setText(2, chunk_details) # Column 2: Chunk details
                    chunk_item.setToolTip(0, f"Chunk {chunk_display_index} from {filename_display}")

                    # Store data roles for identification and metadata index
                    chunk_item.setData(0, self.IS_DOCUMENT_ROLE, False)
                    chunk_item.setData(0, self.CHUNK_INDEX_ROLE, metadata_list_index) # Store index within the metadata list
                    chunk_item.setData(0, self.COLLECTION_ID_ROLE, current_cid) # Store collection ID
                else:
                    logger.warning(f"Invalid metadata or index ({metadata_list_index}) found for document '{source}' in collection '{current_cid}'. Skipping chunk item.")

            doc_item.setExpanded(False) # Collapse documents by default

        logger.info(f"RAGViewer tree populated for collection '{current_cid}' with {len(docs)} documents.")


    def _display_selected_content(self):
        """Displays content based on the first selected item in the tree."""
        self._reset_copy_button_icon() # Reset copy button state
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            self.content_edit.clear()
            self.copy_chunk_button.setEnabled(False)
            return

        # Display content based on the *first* selected item
        item_to_display = selected_items[0]

        is_doc = item_to_display.data(0, self.IS_DOCUMENT_ROLE)
        chunk_metadata_idx = item_to_display.data(0, self.CHUNK_INDEX_ROLE) # Index in _current_collection_metadata
        collection_id_of_item = item_to_display.data(0, self.COLLECTION_ID_ROLE)

        if is_doc:
            # Display document information if a document node is selected
            filepath = item_to_display.data(0, self.FILEPATH_ROLE)
            num_chunks = item_to_display.childCount()
            self.content_edit.setPlainText(
                f"Document: {item_to_display.text(0)}\n"
                f"Collection: {collection_id_of_item}\n"
                f"Full Path: {filepath}\n"
                f"Chunks: {num_chunks}\n\n"
                "(Select a specific chunk to view its content or right-click to set focus)"
            )
            self.copy_chunk_button.setEnabled(False) # Cannot copy document info
        elif chunk_metadata_idx is not None and 0 <= chunk_metadata_idx < len(self._current_collection_metadata):
            # Display chunk content if a chunk node is selected
            metadata = self._current_collection_metadata[chunk_metadata_idx]
            if isinstance(metadata, dict):
                content = metadata.get("content", "[Content not found in metadata]")
                self.content_edit.setPlainText(content)
                self.copy_chunk_button.setEnabled(bool(content)) # Enable copy only if content exists
            else:
                # Handle case where metadata at the index is invalid
                self.content_edit.setPlainText("[Error: Invalid metadata format for this chunk]")
                self.copy_chunk_button.setEnabled(False)
                logger.error(f"Invalid metadata format at index {chunk_metadata_idx} for collection '{collection_id_of_item}'")
        else:
            # Handle unexpected selection state
            self.content_edit.clear()
            self.copy_chunk_button.setEnabled(False)
            logger.warning(f"Invalid item selection state for display. IsDoc: {is_doc}, ChunkMetaIdx: {chunk_metadata_idx}")

    def _copy_selected_chunk_with_feedback(self):
        """Copies the content of the selected chunk (first selected if multiple)."""
        selected_items = self.tree_widget.selectedItems()
        if not selected_items: return

        item_to_copy = selected_items[0] # Base copy action on the first selected item
        is_doc = item_to_copy.data(0, self.IS_DOCUMENT_ROLE)
        chunk_metadata_idx = item_to_copy.data(0, self.CHUNK_INDEX_ROLE)

        # Only copy if it's a chunk item with a valid index
        if is_doc or chunk_metadata_idx is None: return

        if 0 <= chunk_metadata_idx < len(self._current_collection_metadata):
            metadata = self._current_collection_metadata[chunk_metadata_idx]
            if isinstance(metadata, dict):
                content = metadata.get("content", "")
                if not content:
                    logger.warning("Attempted to copy empty chunk content.")
                    return
                try:
                    cb = QApplication.clipboard()
                    if cb:
                        cb.setText(content)
                        logger.info("Copied chunk content from RAG viewer.")
                        # Provide visual feedback
                        if not CHECK_ICON.isNull(): self.copy_chunk_button.setIcon(CHECK_ICON)
                        self.copy_chunk_button.setEnabled(False)
                        QTimer.singleShot(1500, self._reset_copy_button_icon) # Reset after 1.5 seconds
                    else:
                        logger.error("Could not access clipboard.")
                        QMessageBox.warning(self, "Copy Error", "Could not access system clipboard.")
                except Exception as e:
                    logger.exception(f"Error copying chunk content: {e}")
                    QMessageBox.warning(self, "Copy Error", f"{e}")
            else:
                logger.error(f"Cannot copy: Invalid metadata format at index {chunk_metadata_idx}")
        else:
            logger.error(f"Invalid chunk metadata index ({chunk_metadata_idx}) for copy.")

    def _reset_copy_button_icon(self):
        """Resets the copy button icon and enabled state."""
        if not COPY_ICON.isNull(): self.copy_chunk_button.setIcon(COPY_ICON)
        # Re-enable copy button only if a single chunk is currently selected
        selected_items = self.tree_widget.selectedItems()
        is_single_chunk_selected = False
        if len(selected_items) == 1:
            item = selected_items[0]
            is_doc = item.data(0, self.IS_DOCUMENT_ROLE)
            chunk_idx = item.data(0, self.CHUNK_INDEX_ROLE)
            is_single_chunk_selected = not is_doc and chunk_idx is not None
        self.copy_chunk_button.setEnabled(is_single_chunk_selected)

    # --- Dialog Lifecycle ---
    def showEvent(self, event):
        """Called when the dialog is shown."""
        super().showEvent(event)
        logger.info("RAGViewerDialog shown.")
        # Populate collections when the dialog is shown for the first time or refreshed
        QTimer.singleShot(0, self.populate_collection_selector)
        self.activateWindow()
        self.raise_()