# Llama_Syn/services/file_handler_service.py
# NEW FILE - Handles reading content from different file types

import os
import logging
from typing import Tuple, Optional

# --- Dependency Imports ---
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PyPDF2 = None; PYPDF2_AVAILABLE = False
    logging.warning("FileHandlerService: PyPDF2 library not found. Install: pip install pypdf2")

try:
    import docx # From python-docx library
    DOCX_AVAILABLE = True
except ImportError:
     docx = None; DOCX_AVAILABLE = False;
     logging.warning("FileHandlerService: python-docx library not found. Install: pip install python-docx")

# --- Local Imports ---
from utils import constants

logger = logging.getLogger(__name__)

class FileHandlerService:
    """Provides methods to read and extract text content from various file types."""

    def __init__(self):
        logger.info("FileHandlerService initialized.")
        if not PYPDF2_AVAILABLE:
            logger.warning("PDF handling will be unavailable.")
        if not DOCX_AVAILABLE:
            logger.warning("DOCX handling will be unavailable.")

    def read_file_content(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Safely reads file content, identifies type (text, binary, error), handles PDF and DOCX.

        Args:
            file_path: The absolute path to the file.

        Returns:
            A tuple containing:
            - content (Optional[str]): The extracted text content, or None on error/binary.
            - file_type (str): "text", "binary", or "error".
            - error_msg (Optional[str]): An error message if file_type is "error", otherwise None.
        """
        display_name = os.path.basename(file_path)
        file_ext_lower = os.path.splitext(file_path)[1].lower()

        # --- File Size Check ---
        try:
            file_size = os.path.getsize(file_path)
            max_size_mb = getattr(constants, 'RAG_MAX_FILE_SIZE_MB', 50)
            max_size_bytes = max_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                err_msg = f"File > {max_size_mb}MB"
                logger.warning(f"{err_msg}: '{display_name}' ({file_size / (1024*1024):.2f}MB)")
                return None, "error", err_msg
            if file_size == 0:
                logger.warning(f"File empty: '{display_name}'"); return None, "error", "File empty"
        except OSError as e:
            err_msg = f"OS Error (size): {e}"; logger.error(f"Error getting size for '{display_name}': {e}"); return None, "error", err_msg

        # --- PDF Handling ---
        if file_ext_lower == '.pdf':
             if not PYPDF2_AVAILABLE:
                 logger.error(f"Cannot read PDF '{display_name}': PyPDF2 not installed.")
                 return None, "error", "PyPDF2 not installed"
             try:
                  content_list = []
                  with open(file_path, 'rb') as pdf_file:
                      reader = PyPDF2.PdfReader(pdf_file)
                      num_pages = len(reader.pages)
                      logger.info(f"Reading {num_pages} pages from PDF: '{display_name}'")
                      for page_num in range(num_pages):
                           try: content_list.append(reader.pages[page_num].extract_text() or "")
                           except Exception as e_page: logger.warning(f"Error extracting text from page {page_num+1} of '{display_name}': {e_page}")
                  pdf_content = "\n\n--- Page Break ---\n\n".join(content_list).strip()
                  if not pdf_content:
                      logger.warning(f"No text extracted from PDF: '{display_name}'")
                      return None, "error", "No text extracted from PDF"
                  logger.info(f"Successfully extracted text from PDF: '{display_name}' (Length: {len(pdf_content)})")
                  return pdf_content, "text", None
             except Exception as e_pdf:
                  logger.exception(f"Error reading PDF '{display_name}': {e_pdf}")
                  return None, "error", f"PDF Read Error: {e_pdf}"

        # --- DOCX Handling ---
        elif file_ext_lower == '.docx':
             if not DOCX_AVAILABLE:
                 logger.error(f"Cannot read DOCX '{display_name}': python-docx not installed.")
                 return None, "error", "python-docx not installed"
             try:
                  logger.info(f"Reading DOCX file: '{display_name}'")
                  document = docx.Document(file_path)
                  content_list = [p.text for p in document.paragraphs if p.text]
                  docx_content = "\n\n".join(content_list).strip() # Join paragraphs with double newline
                  if not docx_content:
                       logger.warning(f"No text extracted from DOCX: '{display_name}'")
                       return None, "error", "No text extracted from DOCX"
                  logger.info(f"Successfully extracted text from DOCX: '{display_name}' (Length: {len(docx_content)})")
                  return docx_content, "text", None
             except Exception as e_docx:
                  logger.exception(f"Error reading DOCX '{display_name}': {e_docx}")
                  return None, "error", f"DOCX Read Error: {e_docx}"

        # --- Generic Text Reading ---
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors='strict') as f:
                    content = f.read()
                    # Basic check for null bytes to guess if binary
                    if '\x00' in content[:1024]:
                        logger.warning(f"File likely binary (null bytes found): '{display_name}'"); return None, "binary", None
                    return content, "text", None
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for '{display_name}'. Trying latin-1...")
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                        if '\x00' in content[:1024]:
                            logger.warning(f"File (read as latin-1) likely binary: '{display_name}'"); return None, "binary", None
                        logger.info(f"Read '{display_name}' using latin-1."); return content, "text", None
                except Exception as e_fallback:
                     err_msg = f"Fallback read failed: {e_fallback}"
                     logger.warning(f"Failed read '{display_name}' with fallback. Treating as binary/unreadable. Error: {err_msg}")
                     return None, "binary", err_msg # Treat as binary if fallback fails
            except FileNotFoundError:
                logger.error(f"File not found during read: '{display_name}'"); return None, "error", "File not found"
            except OSError as e_os:
                err_msg = f"OS Error (read): {e_os}"; logger.error(f"OS error reading '{display_name}': {e_os}"); return None, "error", err_msg
            except Exception as e_gen:
                err_msg = f"Unexpected read error: {e_gen}"; logger.exception(f"Unexpected read error '{display_name}': {e_gen}"); return None, "error", err_msg

    # Potential future methods:
    # def read_image_content_ocr(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
    #     """ Reads image content using Tesseract OCR. """
    #     # Implementation would go here, requiring pytesseract and Pillow
    #     pass