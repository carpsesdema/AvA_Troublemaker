# SynaChat/ui/loading_indicator.py
# UPDATED FILE - Reverted to using QLabel.setScaledContents
import os
import logging
from typing import Optional

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QMovie, QPixmap
from PyQt6.QtCore import QSize, QTimer, Qt

# --- Local Imports ---
from utils import constants

logger = logging.getLogger(__name__)

class LoadingIndicator(QLabel):
    """
    A QLabel widget dedicated to displaying and controlling an animated loading GIF.
    Expects its size to be set explicitly (e.g., using setFixedSize).
    Uses QLabel.setScaledContents to fit the GIF animation to the label bounds.
    """
    DEFAULT_MIN_SIZE = QSize(16, 16)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LoadingIndicatorLabel")

        self._movie: Optional[QMovie] = None
        self._is_setup_attempted = False
        self._is_setup_successful = False

        if parent:
            logger.debug(f"{self.__class__.__name__} initialized with parent: {parent.objectName()} ({type(parent)})")
        else:
            logger.warning(f"{self.__class__.__name__} initialized WITHOUT parent!")

        QTimer.singleShot(0, self._setup_movie)

        self.setMinimumSize(self.DEFAULT_MIN_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True) # *** USE QLabel SCALING ***
        self.setVisible(False)

    def _setup_movie(self):
        """Loads the QMovie from the GIF file and validates dimensions."""
        if self._is_setup_attempted: return
        self._is_setup_attempted = True
        logger.info(f"{self.__class__.__name__}: Attempting setup...")

        try:
            gif_path = os.path.join(constants.ASSETS_PATH, constants.LOADING_GIF_FILENAME)
            logger.info(f"{self.__class__.__name__}: Attempting to load GIF from: {gif_path}")

            if not os.path.exists(gif_path):
                logger.error(f"{self.__class__.__name__}: GIF file NOT FOUND at: {gif_path}")
                raise FileNotFoundError(f"Loading GIF not found at: {gif_path}")

            self._movie = QMovie(gif_path)
            if not self._movie.isValid():
                logger.error(f"{self.__class__.__name__}: QMovie reported INVALID GIF format: {gif_path}")
                raise ValueError(f"Invalid GIF file format: {gif_path}")

            # *** REMOVED QMovie.setScaledSize() ***
            # current_label_size = self.size()
            # if not current_label_size.isEmpty() and current_label_size.isValid():
            #     logger.info(f"{self.__class__.__name__}: Setting movie scaled size during setup to: {current_label_size}")
            #     self._movie.setScaledSize(current_label_size)
            # else:
            #     logger.warning(f"{self.__class__.__name__}: Label size invalid/empty during setup ({current_label_size}), cannot set movie scaled size yet.")
            # *** END REMOVAL ***

            self.setMovie(self._movie) # Assign movie to the label

            # Check frame validity (keep this check)
            if self._movie.frameCount() > 0:
                logger.debug(f"{self.__class__.__name__}: Jumping to frame 0 to check dimensions.")
                if self._movie.jumpToFrame(0):
                    first_pixmap = self._movie.currentPixmap()
                    if first_pixmap.isNull() or first_pixmap.size().isEmpty():
                        logger.error(f"{self.__class__.__name__}: First frame pixmap is NULL or EMPTY size after jumpToFrame(0). GIF likely unloadable by Qt.")
                        # Don't raise error, just log
                    else:
                        loaded_size = first_pixmap.size()
                        logger.info(f"{self.__class__.__name__}: First frame pixmap seems valid. Size: {loaded_size}")
                else:
                     logger.warning(f"{self.__class__.__name__}: jumpToFrame(0) failed.")
            else:
                 logger.warning(f"{self.__class__.__name__}: Movie has 0 frames according to QMovie.")

            self._is_setup_successful = True
            logger.info(f"{self.__class__.__name__}: GIF '{constants.LOADING_GIF_FILENAME}' setup marked successful (frame count: {self._movie.frameCount()}).")

        except Exception as e:
            self._is_setup_successful = False
            logger.exception(f"{self.__class__.__name__}: CRITICAL FAILURE during _setup_movie: {e}")
            self.setText("[X]")
            self.setStyleSheet("QLabel { color: red; font-weight: bold; background-color: pink; }")
            self.setFixedSize(self.DEFAULT_MIN_SIZE)
            self.setScaledContents(False) # Don't scale error text
            self.setVisible(True)

    def start(self) -> None:
        logger.debug(f"{self.__class__.__name__}: Received start() request.")
        if not self._is_setup_successful or not self._movie:
            logger.warning(f"{self.__class__.__name__}: Cannot start, setup failed or movie not available.")
            self.setVisible(False)
            return

        if self.isVisible() and self._movie.state() == QMovie.MovieState.Running:
            logger.debug(f"{self.__class__.__name__}: Already visible and running, ignoring start().")
            return

        # *** REMOVED ensuring QMovie.setScaledSize before start ***
        # current_label_size = self.size()
        # if not current_label_size.isEmpty() and current_label_size.isValid():
        #      logger.debug(f"{self.__class__.__name__}: Ensuring movie scaled size before start: {current_label_size}")
        #      self._movie.setScaledSize(current_label_size)
        # else:
        #     logger.warning(f"{self.__class__.__name__}: Label size invalid/empty before start ({current_label_size}), movie might not scale correctly.")
        # *** END REMOVAL ***

        logger.info(f"{self.__class__.__name__}: Starting animation and setting visible.")
        self.setVisible(True)
        self.raise_()
        self._movie.start()

    def stop(self) -> None:
        logger.debug(f"{self.__class__.__name__}: Received stop() request.")
        if not self._is_setup_successful or not self._movie:
             if self.isVisible():
                 logger.debug(f"{self.__class__.__name__}: Hiding indicator on stop() (setup failed).")
                 self.setVisible(False)
             return

        if self._movie.state() == QMovie.MovieState.Running:
            logger.debug(f"{self.__class__.__name__}: Stopping animation.")
            self._movie.stop()

        if self.isVisible():
             logger.debug(f"{self.__class__.__name__}: Hiding indicator on stop().")
             self.setVisible(False)