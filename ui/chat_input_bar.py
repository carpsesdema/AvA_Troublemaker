# Llama_Syn/ui/chat_input_bar.py
# UPDATED FILE - Added attach image button and logic, loads custom icon

import logging
import os
from typing import Optional, List, Dict, Any

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSizePolicy, QFileDialog,
    QStyle, QLabel
)
from PyQt6.QtGui import QFont, QIcon # Added QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QSize, pyqtSlot

# --- Local Imports ---
from utils import constants
from .multiline_input_widget import MultilineInputWidget
from .loading_indicator import LoadingIndicator
try:
    from services.image_handler_service import ImageHandlerService
    IMAGE_SERVICE_AVAILABLE = True
except ImportError:
    ImageHandlerService = None
    IMAGE_SERVICE_AVAILABLE = False
    logging.warning("ChatInputBar: ImageHandlerService not found. Image attachment disabled.")


logger = logging.getLogger(__name__)

class ChatInputBar(QWidget):
    """
    A composite widget containing the multi-line text input, attach button,
    loading indicator, and send button. Handles image attachment.
    """
    sendMessageRequested = pyqtSignal(str, list)

    INDICATOR_SIZE = QSize(24, 24)
    ATTACH_BUTTON_SIZE = QSize(26, 26)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatInputBar")

        self._multiline_input: Optional[MultilineInputWidget] = None
        self._send_button: Optional[QPushButton] = None
        self._loading_indicator: Optional[LoadingIndicator] = None
        self._attach_button: Optional[QPushButton] = None
        self._attachment_label: Optional[QLabel] = None

        self._attached_image_paths: List[str] = []
        self._image_service = ImageHandlerService() if IMAGE_SERVICE_AVAILABLE else None

        self._init_ui()
        self._connect_signals()

        self._is_busy = False
        self._is_enabled = True
        self._update_button_state()

    def _init_ui(self):
        """Set up the internal layout and widgets."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 0, 5, 0)
        main_layout.setSpacing(5)

        # --- Add Attach Button with Custom Icon ---
        self._attach_button = QPushButton(self)
        self._attach_button.setObjectName("AttachButton")
        self._attach_button.setToolTip("Attach Image(s)")

        # --- Load Custom Icon ---
        custom_icon_path = os.path.join(constants.ASSETS_PATH, "attach_icon.svg") # Assumes your icon is named this
        if os.path.exists(custom_icon_path):
            custom_icon = QIcon(custom_icon_path)
            if not custom_icon.isNull():
                self._attach_button.setIcon(custom_icon)
                logger.info(f"Loaded custom attach icon from: {custom_icon_path}")
            else:
                logger.warning(f"Custom attach icon loaded but is null: {custom_icon_path}")
                self._attach_button.setText("+") # Fallback text
        else:
            logger.warning(f"Custom attach icon not found: {custom_icon_path}. Using fallback text.")
            self._attach_button.setText("+") # Fallback text
        # --- End Custom Icon Loading ---

        self._attach_button.setFixedSize(self.ATTACH_BUTTON_SIZE)
        self._attach_button.setIconSize(self.ATTACH_BUTTON_SIZE - QSize(8, 8)) # Make icon slightly smaller than button
        self._attach_button.setEnabled(IMAGE_SERVICE_AVAILABLE)
        main_layout.addWidget(self._attach_button)
        # ------------------------

        # Multiline Input
        self._multiline_input = MultilineInputWidget(self)
        main_layout.addWidget(self._multiline_input, 1) # Input widget stretches

        # Loading Indicator
        self._loading_indicator = LoadingIndicator(self)
        self._loading_indicator.setObjectName("InputBarLoadingIndicator")
        self._loading_indicator.setFixedSize(self.INDICATOR_SIZE)
        self._loading_indicator.setVisible(False)
        main_layout.addWidget(self._loading_indicator)

        # Send button
        self._send_button = QPushButton("Send", self)
        self._send_button.setObjectName("SendButton")
        send_button_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)
        self._send_button.setFont(send_button_font)
        self._send_button.setToolTip("Send message (Enter) or add newline (Shift+Enter)")
        self._send_button.setDefault(True)
        self._send_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._send_button.adjustSize()
        main_layout.addWidget(self._send_button)

    def _connect_signals(self):
        """Connect internal widget signals."""
        if self._multiline_input:
            self._multiline_input.sendMessageRequested.connect(self._on_send)
            self._multiline_input.textChanged.connect(self._update_button_state)

        if self._send_button:
            self._send_button.clicked.connect(self._on_send)

        if self._attach_button:
            self._attach_button.clicked.connect(self._handle_attach_image)

    @pyqtSlot()
    def _handle_attach_image(self):
        """Opens file dialog to select images."""
        if not self._image_service:
            logger.warning("Image service not available, cannot attach images.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach Image(s)",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)"
        )

        if file_paths:
            new_paths = [fp for fp in file_paths if fp not in self._attached_image_paths]
            self._attached_image_paths.extend(new_paths)
            logger.info(f"Attached {len(new_paths)} new image(s). Total: {len(self._attached_image_paths)}")
            self._update_attachment_display()
            self._update_button_state()

    def _update_attachment_display(self):
        """Updates UI element to show attached files (e.g., tooltip or label)."""
        if self._attach_button:
            if self._attached_image_paths:
                filenames = [os.path.basename(p) for p in self._attached_image_paths]
                tooltip_text = f"Attached ({len(filenames)}):\n- " + "\n- ".join(filenames)
                if len(tooltip_text) > 500: tooltip_text = tooltip_text[:497]+"..."
                self._attach_button.setToolTip(tooltip_text)
            else:
                self._attach_button.setToolTip("Attach Image(s)")

    @pyqtSlot()
    def _on_send(self):
        """Handles send action, processes images, and emits signal."""
        if not self._is_enabled or self._is_busy:
            return

        text_to_send = self.get_text()
        image_data_list = []

        if not text_to_send and not self._attached_image_paths:
            logger.warning("Send requested with no text or attachments.")
            return

        if self._attached_image_paths and self._image_service:
            logger.info(f"Processing {len(self._attached_image_paths)} attached images for sending...")
            processed_count = 0
            failed_files = []
            for img_path in self._attached_image_paths:
                processed_data = self._image_service.process_image_to_base64(img_path)
                if processed_data:
                    base64_str, mime_type = processed_data
                    image_data_list.append({
                        "type": "image",
                        "mime_type": mime_type,
                        "data": base64_str
                    })
                    processed_count += 1
                else:
                    failed_files.append(os.path.basename(img_path))

            if failed_files:
                logger.error(f"Failed to process {len(failed_files)} images: {', '.join(failed_files)}")
            logger.info(f"Successfully processed {processed_count} images.")

        elif self._attached_image_paths and not self._image_service:
             logger.error("Cannot send attached images: Image service is not available.")

        logger.debug(f"ChatInputBar emitting sendMessageRequested (Text: {bool(text_to_send)}, Images: {len(image_data_list)})")
        self.sendMessageRequested.emit(text_to_send, image_data_list)
        self.clear_text()

    @pyqtSlot(bool)
    def handle_busy_state(self, is_busy: bool):
        """Handles busy state for enabling/disabling input/button AND indicator."""
        logger.debug(f"ChatInputBar handling busy state: {is_busy}")
        if self._is_busy == is_busy:
             return
        self._is_busy = is_busy
        effective_enabled = self._is_enabled and not self._is_busy

        if self._multiline_input:
            self._multiline_input.set_enabled(effective_enabled)
        if self._attach_button:
            self._attach_button.setEnabled(IMAGE_SERVICE_AVAILABLE and effective_enabled)

        self._update_button_state()

        if self._loading_indicator:
            self._loading_indicator.setVisible(is_busy)
            if is_busy: self._loading_indicator.start()
            else: self._loading_indicator.stop()

    @pyqtSlot()
    def _update_button_state(self):
        """Update send button based on text, attachments, and busy state."""
        if self._send_button:
             has_text = bool(self.get_text())
             has_attachments = bool(self._attached_image_paths)
             can_send = self._is_enabled and not self._is_busy and (has_text or has_attachments)
             self._send_button.setEnabled(can_send)

    # --- Public Methods ---
    def get_text(self) -> str:
        return self._multiline_input.get_text() if self._multiline_input else ""

    def clear_text(self):
        """Clears text and attachments."""
        if self._multiline_input:
            self._multiline_input.clear_text()
        self._attached_image_paths = []
        self._update_attachment_display()
        self._update_button_state()

    def set_focus(self):
        if self._multiline_input:
            self._multiline_input.set_focus()

    def set_enabled(self, enabled: bool):
        self._is_enabled = enabled
        self.handle_busy_state(self._is_busy)