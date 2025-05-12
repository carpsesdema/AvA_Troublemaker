# Syn_LLM/services/image_handler_service.py
# NEW FILE - Handles image loading, validation, and encoding

import os
import base64
import io
import logging
from typing import Optional, Tuple

try:
    from PIL import Image, ExifTags
    PILLOW_AVAILABLE = True
except ImportError:
    Image = None
    ExifTags = None
    PILLOW_AVAILABLE = False
    logging.error("ImageHandlerService: Pillow library not found. Install: pip install Pillow")

logger = logging.getLogger(__name__)

# Default limits (can be moved to constants)
DEFAULT_MAX_IMAGE_SIZE_MB = 15
DEFAULT_MAX_DIMENSION = 2048 # Max width or height

# Mapping from PIL format to common MIME types
MIME_TYPE_MAP = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
}

class ImageHandlerService:
    """Provides methods for handling image files for LLM input."""

    def __init__(self,
                 max_size_mb: int = DEFAULT_MAX_IMAGE_SIZE_MB,
                 max_dimension: int = DEFAULT_MAX_DIMENSION):
        if not PILLOW_AVAILABLE:
            logger.critical("Pillow library not available. Image handling is disabled.")
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_dimension = max_dimension
        logger.info(f"ImageHandlerService initialized (Max Size: {max_size_mb}MB, Max Dim: {max_dimension}px)")

    def process_image_to_base64(self, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Loads, validates, resizes (if needed), and converts an image to a base64 string.

        Args:
            file_path: Path to the image file.

        Returns:
            A tuple (base64_string, mime_type) or None if processing fails.
        """
        if not PILLOW_AVAILABLE: return None
        if not os.path.exists(file_path):
            logger.error(f"Image file not found: {file_path}")
            return None

        try:
            # 1. Validate file size before loading fully
            file_size = os.path.getsize(file_path)
            if file_size > self.max_size_bytes:
                logger.warning(f"Image skipped: '{os.path.basename(file_path)}' exceeds max size ({file_size / (1024*1024):.1f}MB > {self.max_size_bytes / (1024*1024):.1f}MB)")
                return None # Indicate failure due to size

            # 2. Load Image using Pillow
            img = Image.open(file_path)

            # 3. Handle Orientation using EXIF data (important for mobile photos)
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(img.getexif().items())

                if exif.get(orientation) == 3: img = img.rotate(180, expand=True)
                elif exif.get(orientation) == 6: img = img.rotate(270, expand=True)
                elif exif.get(orientation) == 8: img = img.rotate(90, expand=True)
                logger.debug(f"Applied EXIF orientation {exif.get(orientation)} to image.")
            except (AttributeError, KeyError, IndexError, Exception) as e_exif:
                # cases: image don't have getexif method, don't have exif data or orientation tag
                logger.debug(f"Could not get/apply EXIF orientation for {os.path.basename(file_path)}: {e_exif}")
                pass # Ignore errors if EXIF data is missing/corrupt

            # 4. Convert to RGB if it has transparency (needed for JPEG saving)
            original_format = img.format
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                logger.debug("Converted image mode to RGB.")

            # 5. Resize if necessary
            if img.width > self.max_dimension or img.height > self.max_dimension:
                img.thumbnail((self.max_dimension, self.max_dimension))
                logger.info(f"Resized image '{os.path.basename(file_path)}' to fit within {self.max_dimension}px.")

            # 6. Encode to Base64
            buffered = io.BytesIO()
            # Save in original format if supported and common, else default to JPEG
            save_format = original_format if original_format in ["JPEG", "PNG", "GIF", "WEBP"] else "JPEG"
            img.save(buffered, format=save_format)
            base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # 7. Determine MIME type
            mime_type = MIME_TYPE_MAP.get(save_format, "image/jpeg") # Default to jpeg

            logger.info(f"Successfully processed image '{os.path.basename(file_path)}' (Format: {save_format}, Size: {len(base64_string)} chars)")
            return base64_string, mime_type

        except Exception as e:
            logger.exception(f"Error processing image file '{file_path}': {e}")
            return None
        finally:
             if 'img' in locals() and hasattr(img, 'close'):
                  img.close()