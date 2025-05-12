# ui/chat_item_delegate.py
# UPDATED - Implemented loading indicator within AI message bubbles
#           Uses loading.gif and loading_complete.png from assets.

import logging
import base64
import html
import hashlib
import os # Added for path joining
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import QStyledItemDelegate, QStyle, QApplication, QStyleOptionViewItem, QWidget
from PyQt6.QtGui import (
    QPainter, QColor, QFontMetrics, QTextDocument, QPixmap, QImage, QFont,
    QImageReader, QPen, QMovie # Added QMovie
)
from PyQt6.QtCore import QModelIndex, QRect, QPoint, QSize, Qt, QObject, QByteArray, QUrl, QPersistentModelIndex, \
    pyqtSlot  # Added QPersistentModelIndex

# --- Core Model & Enum Imports ---
try:
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
    from ui.chat_list_model import ChatListModel, ChatMessageRole, LoadingStatusRole # Added LoadingStatusRole
    from utils import constants
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE, ASSETS_PATH # Added ASSETS_PATH
except ImportError: # Fallback for different environment setups
    try:
        from ..core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE # type: ignore
        from ..core.message_enums import MessageLoadingState # type: ignore
        from ..ui.chat_list_model import ChatListModel, ChatMessageRole, LoadingStatusRole # type: ignore
        from ..utils import constants # type: ignore
        from ..utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE, ASSETS_PATH # type: ignore
    except ImportError as e_imp:
        logging.critical(f"ChatItemDelegate: Failed to import dependencies: {e_imp}")
        # Define dummy classes/values if import fails
        class ChatMessage: pass # type: ignore
        from enum import Enum, auto
        class MessageLoadingState(Enum): IDLE=auto(); LOADING=auto(); COMPLETED=auto(); ERROR=auto() # type: ignore
        class ChatListModel: pass # type: ignore
        ChatMessageRole, LoadingStatusRole = 0, 1 # type: ignore
        class constants: CHAT_FONT_FAMILY="Arial"; CHAT_FONT_SIZE=10; ASSETS_PATH="assets" # type: ignore


# --- Dependency for Markdown ---
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Constants for Delegate ---
BUBBLE_PADDING_V = 8
BUBBLE_PADDING_H = 12
BUBBLE_MARGIN_V = 4
BUBBLE_MARGIN_H = 10
BUBBLE_RADIUS = 12
IMAGE_PADDING = 5
MAX_IMAGE_WIDTH = 250
MAX_IMAGE_HEIGHT = 250
MIN_BUBBLE_WIDTH = 50
USER_BUBBLE_INDENT = 40
TIMESTAMP_PADDING_TOP = 3
TIMESTAMP_HEIGHT = 15
BUBBLE_MAX_WIDTH_PERCENTAGE = 0.75

# --- Loading Indicator Specifics ---
INDICATOR_SIZE = QSize(22, 22)  # Desired display size of the loading/completed icon
INDICATOR_PADDING_X = 5 # Padding from the right edge of the bubble
INDICATOR_PADDING_Y = 5 # Padding from the bottom edge of the bubble

# --- Colors ---
# (Colors remain the same as your existing file)
USER_BUBBLE_COLOR = QColor("#0b93f6")
USER_TEXT_COLOR = QColor(Qt.GlobalColor.white)
AI_BUBBLE_COLOR = QColor("#3c3f41")
AI_TEXT_COLOR = QColor("#dcdcdc")
SYSTEM_BUBBLE_COLOR = QColor("#4a4e51")
SYSTEM_TEXT_COLOR = QColor("#aabbcc")
ERROR_BUBBLE_COLOR = QColor("#6e3b3b")
ERROR_TEXT_COLOR = QColor("#ffcccc")
CODE_BG_COLOR = QColor("#282c34")
BUBBLE_BORDER_COLOR = QColor("#4f5356")
TIMESTAMP_COLOR = QColor("#888888")


class ChatItemDelegate(QStyledItemDelegate):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._font = QFont(CHAT_FONT_FAMILY, CHAT_FONT_SIZE)
        self._font_metrics = QFontMetrics(self._font)
        self._timestamp_font = QFont(CHAT_FONT_FAMILY, CHAT_FONT_SIZE - 2)
        self._timestamp_font_metrics = QFontMetrics(self._timestamp_font)
        self._text_doc_cache: Dict[Tuple[str, int, bool, str], QTextDocument] = {}
        self._image_pixmap_cache: Dict[str, QPixmap] = {}

        # --- For Loading Indicator ---
        self._loading_animation_movie: Optional[QMovie] = None
        self._completed_icon_pixmap: Optional[QPixmap] = None
        self._active_loading_movies: Dict[QPersistentModelIndex, QMovie] = {} # Track QMovie per item
        self._view_ref: Optional[QWidget] = None # Reference to the view for updates

        self._init_loading_indicator_assets()
        logger.info("ChatItemDelegate initialized with loading indicator assets.")

    def _init_loading_indicator_assets(self):
        """Load the GIF for animation and PNG for completed state."""
        try:
            loading_gif_path = os.path.join(ASSETS_PATH, "loading.gif")
            if os.path.exists(loading_gif_path):
                self._loading_animation_movie = QMovie(loading_gif_path)
                if self._loading_animation_movie.isValid():
                    self._loading_animation_movie.setScaledSize(INDICATOR_SIZE) # Scale the QMovie itself
                    logger.info(f"Loading GIF '{loading_gif_path}' loaded and scaled to {INDICATOR_SIZE.width()}x{INDICATOR_SIZE.height()}.")
                else:
                    logger.error(f"Failed to load QMovie from '{loading_gif_path}'. Invalid GIF.")
                    self._loading_animation_movie = None
            else:
                logger.error(f"Loading GIF not found at '{loading_gif_path}'.")

            completed_png_path = os.path.join(ASSETS_PATH, "loading_complete.png")
            if os.path.exists(completed_png_path):
                self._completed_icon_pixmap = QPixmap(completed_png_path)
                if not self._completed_icon_pixmap.isNull():
                    self._completed_icon_pixmap = self._completed_icon_pixmap.scaled(
                        INDICATOR_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    logger.info(f"Completed PNG '{completed_png_path}' loaded and scaled.")
                else:
                    logger.error(f"Failed to load QPixmap from '{completed_png_path}'.")
                    self._completed_icon_pixmap = None
            else:
                logger.error(f"Completed PNG not found at '{completed_png_path}'.")

        except Exception as e:
            logger.exception(f"Error initializing loading indicator assets: {e}")
            self._loading_animation_movie = None
            self._completed_icon_pixmap = None

    def setView(self, view: QWidget):
        """Stores a reference to the view for triggering updates from QMovie."""
        self._view_ref = view
        if self._loading_animation_movie: # Connect frameChanged if movie is already loaded
            # Disconnect previous connection if any to avoid multiple connections on view reset
            try: self._loading_animation_movie.frameChanged.disconnect()
            except TypeError: pass # No connection to disconnect
            self._loading_animation_movie.frameChanged.connect(self._on_movie_frame_changed)

    @pyqtSlot(int)
    def _on_movie_frame_changed(self, frame_number: int):
        """Slot to handle QMovie frame changes and trigger view updates for relevant items."""
        if self._view_ref and self._active_loading_movies:
            movie_sender = self.sender() # Get the QMovie instance that emitted the signal
            for p_index, active_movie in list(self._active_loading_movies.items()): # Iterate over a copy for safe removal
                if active_movie == movie_sender:
                    if p_index.isValid() and self._view_ref.model().data(p_index, LoadingStatusRole) == MessageLoadingState.LOADING:
                        logger.debug(f"Delegate: Movie frame changed for item at row {p_index.row()}. Updating view.")
                        self._view_ref.update(p_index) # Trigger repaint for the specific item
                    else:
                        # Stale entry or state changed, remove it
                        logger.debug(f"Delegate: Removing stale/invalid QMovie from active_movies for row {p_index.row()}.")
                        active_movie.stop()
                        active_movie.deleteLater() # Clean up QMovie
                        del self._active_loading_movies[p_index]
                    break # Found the sender


    def clearCache(self):
        logger.debug("Clearing ChatItemDelegate cache.")
        self._text_doc_cache.clear()
        self._image_pixmap_cache.clear()
        # Also clear active QMovie instances
        for movie in self._active_loading_movies.values():
            movie.stop()
            movie.deleteLater()
        self._active_loading_movies.clear()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        message = index.data(ChatMessageRole)
        if not isinstance(message, ChatMessage):
            super().paint(painter, option, index)
            painter.restore()
            return

        # --- Get Loading State ---
        loading_status = index.model().data(index, LoadingStatusRole)
        if not isinstance(loading_status, MessageLoadingState):
            loading_status = MessageLoadingState.IDLE # Default if role data is missing

        is_user = (message.role == USER_ROLE)
        bubble_color, _ = self._get_colors(message.role)
        available_item_width = option.rect.width()
        required_content_size = self._calculate_content_size(message, available_item_width, is_user)
        bubble_rect = self._get_bubble_rect(option.rect, required_content_size, is_user)

        painter.setPen(QPen(BUBBLE_BORDER_COLOR, 1))
        painter.setBrush(bubble_color)
        painter.drawRoundedRect(bubble_rect, BUBBLE_RADIUS, BUBBLE_RADIUS)

        content_placement_rect = bubble_rect.adjusted(BUBBLE_PADDING_H, BUBBLE_PADDING_V,
                                                      -BUBBLE_PADDING_H, -BUBBLE_PADDING_V)
        current_y = content_placement_rect.top()
        content_width_constraint = content_placement_rect.width()

        text_height = 0
        if message.text and content_width_constraint > 0:
            text_doc = self._get_prepared_text_document(message, content_width_constraint)
            text_doc.setTextWidth(content_width_constraint)
            text_height = int(text_doc.size().height())
            if text_height > 0 and current_y + text_height <= content_placement_rect.bottom() + 1:
                painter.save()
                painter.translate(content_placement_rect.left(), current_y)
                text_doc.drawContents(painter)
                painter.restore()
                current_y += text_height

        if message.has_images and content_width_constraint > 0:
            if message.text and text_height > 0: current_y += IMAGE_PADDING
            image_count = 0
            for img_part in message.image_parts:
                pixmap = self._get_image_pixmap(img_part)
                if pixmap and not pixmap.isNull():
                    target_width = min(pixmap.width(), content_width_constraint, MAX_IMAGE_WIDTH)
                    scaled_pixmap = pixmap.scaledToWidth(target_width, Qt.TransformationMode.SmoothTransformation)
                    if scaled_pixmap.height() > MAX_IMAGE_HEIGHT:
                        scaled_pixmap = scaled_pixmap.scaledToHeight(MAX_IMAGE_HEIGHT, Qt.TransformationMode.SmoothTransformation)
                    if current_y + scaled_pixmap.height() <= content_placement_rect.bottom() + 1:
                        if image_count > 0: current_y += IMAGE_PADDING
                        img_x = content_placement_rect.left() + (content_width_constraint - scaled_pixmap.width()) // 2
                        img_rect = QRect(QPoint(img_x, current_y), scaled_pixmap.size())
                        painter.drawPixmap(img_rect.topLeft(), scaled_pixmap)
                        current_y += scaled_pixmap.height()
                        image_count += 1
                    else: break

        # --- Draw Loading/Completed Indicator for AI Messages ---
        if message.role == MODEL_ROLE and content_width_constraint > 0: # Only for AI messages
            persistent_index = QPersistentModelIndex(index)
            indicator_x = bubble_rect.right() - INDICATOR_SIZE.width() - INDICATOR_PADDING_X
            indicator_y = bubble_rect.bottom() - INDICATOR_SIZE.height() - INDICATOR_PADDING_Y
            indicator_rect = QRect(QPoint(indicator_x, indicator_y), INDICATOR_SIZE)

            if loading_status == MessageLoadingState.LOADING:
                active_movie = self._active_loading_movies.get(persistent_index)
                if not active_movie:
                    if self._loading_animation_movie: # Use the pre-loaded one as a template
                        # Create a new QMovie instance for this item to manage its state independently
                        active_movie = QMovie(self._loading_animation_movie.fileName(), QByteArray(), self) # Parent to delegate
                        if active_movie.isValid():
                            active_movie.setScaledSize(INDICATOR_SIZE) # Ensure new instance is scaled
                            # Connect its frameChanged to the delegate's slot for updates
                            active_movie.frameChanged.connect(self._on_movie_frame_changed)
                            active_movie.start()
                            self._active_loading_movies[persistent_index] = active_movie
                            logger.debug(f"Delegate: Started new QMovie for LOADING AI message at row {index.row()}")
                        else:
                            logger.error(f"Delegate: Failed to create new QMovie instance for row {index.row()}.")
                            active_movie = None # Ensure it's None if creation failed
                    else: # Fallback if the main _loading_animation_movie wasn't loaded
                        logger.warning(f"Delegate: Main loading animation movie not available for row {index.row()}.")

                if active_movie and active_movie.isValid():
                    current_frame_pixmap = active_movie.currentPixmap()
                    if not current_frame_pixmap.isNull():
                        painter.drawPixmap(indicator_rect, current_frame_pixmap)
                    else:
                         logger.warning(f"Delegate: QMovie for row {index.row()} returned null pixmap for current frame.")

            elif loading_status == MessageLoadingState.COMPLETED:
                # Stop and remove movie if it was playing for this index
                if persistent_index in self._active_loading_movies:
                    old_movie = self._active_loading_movies.pop(persistent_index)
                    old_movie.stop()
                    old_movie.deleteLater() # Clean up QMovie
                    logger.debug(f"Delegate: Stopped and removed QMovie for COMPLETED AI message at row {index.row()}.")

                if self._completed_icon_pixmap and not self._completed_icon_pixmap.isNull():
                    painter.drawPixmap(indicator_rect, self._completed_icon_pixmap)

            elif loading_status == MessageLoadingState.IDLE or loading_status == MessageLoadingState.ERROR:
                 # Ensure any active movie for this index is stopped and removed if state changed unexpectedly
                if persistent_index in self._active_loading_movies:
                    old_movie = self._active_loading_movies.pop(persistent_index)
                    old_movie.stop()
                    old_movie.deleteLater() # Clean up QMovie
                    logger.debug(f"Delegate: Stopped/removed QMovie for AI message at row {index.row()} (state became IDLE/ERROR).")
                # Optionally draw an error icon for MessageLoadingState.ERROR here

        # --- Draw Timestamp ---
        formatted_timestamp = self._format_timestamp(message.timestamp)
        if formatted_timestamp:
            timestamp_width = self._timestamp_font_metrics.horizontalAdvance(formatted_timestamp)
            timestamp_x = bubble_rect.right() - timestamp_width
            timestamp_x = max(timestamp_x, bubble_rect.left())
            timestamp_y = bubble_rect.bottom() + TIMESTAMP_PADDING_TOP + self._timestamp_font_metrics.ascent()
            if timestamp_y < option.rect.bottom() - BUBBLE_MARGIN_V:
                painter.setFont(self._timestamp_font)
                painter.setPen(TIMESTAMP_COLOR)
                painter.drawText(QPoint(timestamp_x, timestamp_y), formatted_timestamp)

        if option.state & QStyle.StateFlag.State_Selected:
            highlight_color = option.palette.highlight().color()
            highlight_color.setAlpha(80)
            painter.fillRect(option.rect, highlight_color)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        message = index.data(ChatMessageRole)
        if not isinstance(message, ChatMessage):
            return super().sizeHint(option, index)

        is_user = (message.role == USER_ROLE)
        available_view_width = option.rect.width()
        content_size = self._calculate_content_size(message, available_view_width, is_user)

        if content_size is None or not isinstance(content_size, QSize):
            min_bubble_base_height = self._font_metrics.height() + 2 * BUBBLE_PADDING_V
            return QSize(available_view_width, max(min_bubble_base_height, TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT + 2 * BUBBLE_MARGIN_V))

        total_required_height = content_size.height()
        formatted_timestamp = self._format_timestamp(message.timestamp)
        if formatted_timestamp:
            total_required_height += TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT
        final_height = total_required_height + 2 * BUBBLE_MARGIN_V
        final_width = available_view_width
        min_bubble_base_height = self._font_metrics.height() + 2 * BUBBLE_PADDING_V
        min_total_item_height = min_bubble_base_height + 2 * BUBBLE_MARGIN_V + TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT
        final_height = max(final_height, min_total_item_height)
        return QSize(final_width, final_height)

    # --- Helper Methods (largely unchanged, but _calculate_content_size is important) ---
    def _get_colors(self, role: str) -> tuple[QColor, QColor]:
        if role == USER_ROLE: return USER_BUBBLE_COLOR, USER_TEXT_COLOR
        if role == SYSTEM_ROLE: return SYSTEM_BUBBLE_COLOR, SYSTEM_TEXT_COLOR
        if role == ERROR_ROLE: return ERROR_BUBBLE_COLOR, ERROR_TEXT_COLOR
        return AI_BUBBLE_COLOR, AI_TEXT_COLOR

    def _get_bubble_rect(self, item_rect: QRect, content_size: QSize, is_user: bool) -> QRect:
        bubble_width = content_size.width()
        bubble_height = content_size.height()
        base_x = item_rect.left() + BUBBLE_MARGIN_H
        if is_user:
            bubble_x = item_rect.right() - BUBBLE_MARGIN_H - bubble_width
            bubble_x = max(bubble_x, item_rect.left() + BUBBLE_MARGIN_H + USER_BUBBLE_INDENT)
        else:
            bubble_x = item_rect.left() + BUBBLE_MARGIN_H
        bubble_y = item_rect.top() + BUBBLE_MARGIN_V
        return QRect(bubble_x, bubble_y, bubble_width, bubble_height)

    def _calculate_content_size(self, message: ChatMessage, total_item_width: int, is_user: bool) -> QSize:
        total_height = 0
        actual_content_width = 0
        text_render_height = 0

        try:
            max_bubble_width = int(total_item_width * BUBBLE_MAX_WIDTH_PERCENTAGE)
            if is_user:
                 max_user_bubble_width_considering_indent = total_item_width - (2 * BUBBLE_MARGIN_H) - USER_BUBBLE_INDENT
                 max_bubble_width = min(max_bubble_width, max_user_bubble_width_considering_indent)
            max_bubble_width = max(max_bubble_width, MIN_BUBBLE_WIDTH + 2 * BUBBLE_PADDING_H)
            effective_inner_content_width = max(1, max_bubble_width - 2 * BUBBLE_PADDING_H)
            inner_width_constraint = effective_inner_content_width

            if message.text:
                text_doc = self._get_prepared_text_document(message, inner_width_constraint)
                text_doc.setTextWidth(-1)
                ideal_size = text_doc.size()
                render_text_width = min(int(ideal_size.width()), inner_width_constraint)
                text_doc.setTextWidth(max(1, render_text_width))
                text_render_height = max(0, int(text_doc.size().height()))
                total_height += text_render_height
                actual_content_width = max(actual_content_width, render_text_width)

            if message.has_images:
                if message.text and text_render_height > 0: total_height += IMAGE_PADDING
                image_count = 0
                for img_part in message.image_parts:
                    pixmap = self._get_image_pixmap(img_part)
                    if pixmap and not pixmap.isNull():
                        if image_count > 0: total_height += IMAGE_PADDING
                        target_width = min(pixmap.width(), inner_width_constraint, MAX_IMAGE_WIDTH)
                        scaled_pixmap = pixmap.scaledToWidth(target_width, Qt.TransformationMode.SmoothTransformation)
                        if scaled_pixmap.height() > MAX_IMAGE_HEIGHT:
                            scaled_pixmap = scaled_pixmap.scaledToHeight(MAX_IMAGE_HEIGHT, Qt.TransformationMode.SmoothTransformation)
                        img_render_height = max(0, scaled_pixmap.height())
                        img_render_width = max(0, scaled_pixmap.width())
                        total_height += img_render_height
                        actual_content_width = max(actual_content_width, img_render_width)
                        image_count += 1
                    else:
                        if image_count > 0: total_height += IMAGE_PADDING
                        total_height += self._font_metrics.height()
                        actual_content_width = max(actual_content_width, self._font_metrics.horizontalAdvance("[X]"))
                        image_count += 1

            final_height = total_height + 2 * BUBBLE_PADDING_V
            final_width = actual_content_width + 2 * BUBBLE_PADDING_H
            final_width = max(final_width, MIN_BUBBLE_WIDTH + 2 * BUBBLE_PADDING_H)
            final_width = min(final_width, max_bubble_width)
            return QSize(max(1, final_width), max(1, final_height))
        except Exception as e:
            logger.exception(f"Error calculating content size for message role {message.role}: {e}.")
            min_bubble_base_height = self._font_metrics.height() + 2 * BUBBLE_PADDING_V
            return QSize(MIN_BUBBLE_WIDTH + 2 * BUBBLE_PADDING_H, min_bubble_base_height)


    def _get_prepared_text_document(self, message: ChatMessage, width_constraint: int) -> QTextDocument:
        is_streaming = (message.metadata is not None) and message.metadata.get("is_streaming", False)
        text_content = message.text if message.text else ""
        content_hash = hashlib.sha1(text_content.encode('utf-8')).hexdigest()[:16]
        cache_key = (content_hash, width_constraint, is_streaming, message.role)

        cached_doc = self._text_doc_cache.get(cache_key)
        if cached_doc:
            constrained_width = max(width_constraint, 1)
            if abs(cached_doc.textWidth() - constrained_width) > 1:
                cached_doc.setTextWidth(constrained_width)
            return cached_doc

        doc = QTextDocument()
        doc.setDefaultFont(self._font)
        doc.setDocumentMargin(0)
        _, text_color = self._get_colors(message.role)
        html_content = self._prepare_html(text_content, text_color, is_streaming)
        # Stylesheet content remains the same as your existing file
        doc.setDefaultStyleSheet(f"""
            p {{ margin: 0 0 8px 0; padding: 0; line-height: 130%; }}
            ul, ol {{ margin: 3px 0 8px 20px; padding: 0; }}
            li {{ margin-bottom: 4px; }}
            pre {{ background-color: {CODE_BG_COLOR.name()}; border: 1px solid {BUBBLE_BORDER_COLOR.name()}; padding: 8px; margin: 6px 0; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; font-family: '{self._font.family()}', monospace; font-size: {self._font.pointSize()}pt; color: {AI_TEXT_COLOR.name()}; line-height: 120%; }}
            code {{ background-color: {CODE_BG_COLOR.lighter(110).name()}; padding: 1px 3px; border-radius: 3px; font-family: '{self._font.family()}', monospace; font-size: {int(self._font.pointSize() * 0.95)}pt; color: {AI_TEXT_COLOR.lighter(120).name()}; }}
            table {{ border-collapse: collapse; margin: 8px 0; color: {text_color.name()}; background-color: {CODE_BG_COLOR.lighter(105).name()}; }}
            th, td {{ border: 1px solid {BUBBLE_BORDER_COLOR.name()}; padding: 4px 6px; }}
            th {{ background-color: {CODE_BG_COLOR.lighter(120).name()}; font-weight: bold; }}
            td {{ color: #a9b7c6; }}
            a {{ color: #61afef; text-decoration: underline; }}
            a:hover {{ color: #82c0ff; }}
            blockquote {{ border-left: 3px solid {text_color.darker(120).name()}; margin: 8px 0px 8px 5px; padding-left: 10px; color: {text_color.darker(110).name()}; font-style: italic; }}
            h1, h2, h3, h4, h5, h6 {{ margin-top: 10px; margin-bottom: 5px; font-weight: bold; color: {text_color.lighter(110).name()}; }}
            h1 {{ font-size: 1.4em; border-bottom: 1px solid {BUBBLE_BORDER_COLOR.name()}; }}
            h2 {{ font-size: 1.2em; border-bottom: 1px solid {BUBBLE_BORDER_COLOR.name()}; }}
            h3 {{ font-size: 1.1em; }} h4 {{ font-size: 1.0em; }} h5 {{ font-size: 0.9em; }}
            h6 {{ font-size: 0.9em; font-style: italic; color: {text_color.darker(110).name()}; }}
            hr {{ border: 0; height: 1px; background-color: {BUBBLE_BORDER_COLOR.name()}; margin: 12px 0; }}
        """)
        doc.setHtml(html_content)
        doc.setTextWidth(max(width_constraint, 1))
        self._text_doc_cache[cache_key] = doc
        return doc

    def _prepare_html(self, text: str, text_color: QColor, is_streaming: bool) -> str:
        if not text: return ""
        escaped_text = html.escape(text)
        html_content = escaped_text.replace('\n', '<br/>')
        if not is_streaming and MARKDOWN_AVAILABLE:
            try:
                md_content = markdown.markdown(text, extensions=['fenced_code', 'nl2br', 'tables', 'sane_lists', 'extra'])
                html_content = md_content
            except Exception as e:
                logger.error(f"Markdown conversion failed: {e}. Using escaped text.")
                html_content = escaped_text.replace('\n', '<br/>')
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body style="color:{text_color.name()};">{html_content}</body></html>"""

    def _get_image_pixmap(self, image_part: Dict[str, Any]) -> Optional[QPixmap]:
        base64_data = image_part.get("data")
        if not base64_data or not isinstance(base64_data, str): return None
        data_hash = hashlib.sha1(base64_data.encode()).hexdigest()[:16]
        cached_pixmap = self._image_pixmap_cache.get(data_hash)
        if cached_pixmap: return cached_pixmap
        try:
            missing_padding = len(base64_data) % 4
            if missing_padding: base64_data += '=' * (4 - missing_padding)
            image_bytes = base64.b64decode(base64_data)
            qimage = QImage()
            if qimage.loadFromData(image_bytes):
                 pixmap = QPixmap.fromImage(qimage)
                 if not pixmap.isNull(): self._image_pixmap_cache[data_hash] = pixmap; return pixmap
        except Exception as e: logger.error(f"Error decoding/loading image in delegate: {e}")
        return None

    def _format_timestamp(self, iso_timestamp: Optional[str]) -> Optional[str]:
        if not iso_timestamp: return None
        try:
            dt_object = datetime.fromisoformat(iso_timestamp)
            return dt_object.strftime("%H:%M")
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timestamp: {iso_timestamp}")
            return None