"""Agent 1: Format & Language Detector."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from langdetect import detect, LangDetectException
import mimetypes

try:  # pragma: no cover - exercised in environments with libmagic installed
    import magic  # type: ignore
except ImportError:  # pragma: no cover - python-magic is optional
    magic = None

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from magic import Magic  # type: ignore

import config
from models import FileMetadata

logger = logging.getLogger(__name__)


_magic_instance = None
_magic_error = False


def _normalise_mime(mime_type: Optional[str]) -> Optional[str]:
    """Map MIME type strings to the detector's format labels."""
    if not mime_type:
        return None
    mime_type = mime_type.lower()
    if "pdf" in mime_type:
        return "pdf"
    if "word" in mime_type or "document" in mime_type:
        return "docx"
    if "csv" in mime_type or "text/csv" in mime_type:
        return "csv"
    if "json" in mime_type:
        return "json"
    if "html" in mime_type:
        return "html"
    if "text" in mime_type:
        return "txt"
    return None


def _get_magic() -> Optional["Magic"]:
    """Return a cached python-magic detector if the dependency is available."""
    global _magic_instance, _magic_error
    if magic is None or _magic_error:
        return None
    if _magic_instance is None:
        try:
            _magic_instance = magic.Magic(mime=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialise libmagic: %s", exc)
            _magic_error = True
            return None
    return _magic_instance


def detect_file_type(file_path: Path) -> str:
    """Detect file type from file path and content."""
    suffix = file_path.suffix.lower()
    suffix_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".csv": "csv",
        ".txt": "txt",
        ".json": "json",
        ".html": "html",
    }
    if suffix in suffix_map:
        return suffix_map[suffix]

    magic_detector = _get_magic()
    if magic_detector is not None:
        try:
            detected = _normalise_mime(magic_detector.from_file(str(file_path)))
            if detected:
                return detected
        except Exception as exc:
            logger.warning("Could not detect MIME type for %s: %s", file_path, exc)

    guessed, _ = mimetypes.guess_type(file_path.name)
    detected = _normalise_mime(guessed)
    if detected:
        return detected

    return "unknown"


def detect_language(text: str) -> str:
    """Detect language of text content."""
    if not text or len(text.strip()) < 10:
        return "unknown"
    try:
        lang = detect(text)
        return lang
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        return "unknown"


def extract_metadata(file_path: Path) -> dict:
    """Extract file metadata."""
    stat = file_path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
    modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
    file_size = stat.st_size
    encoding = None
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(1024)
            if raw_data.startswith(b"\xef\xbb\xbf"):
                encoding = "utf-8-sig"
            elif raw_data.startswith(b"\xff\xfe"):
                encoding = "utf-16-le"
            elif raw_data.startswith(b"\xfe\xff"):
                encoding = "utf-16-be"
            else:
                encoding = "utf-8"
    except Exception as e:
        logger.warning(f"Could not detect encoding for {file_path}: {e}")
        encoding = "unknown"
    return {
        "file_size": file_size,
        "encoding": encoding,
        "created_at": created_at,
        "modified_at": modified_at,
    }


def format_detection_result(file_path: Path, file_type: str, language: str, metadata: dict) -> FileMetadata:
    """Format detection results into FileMetadata object."""
    return FileMetadata(
        file_path=file_path,
        file_type=file_type,
        language=language,
        file_size=metadata["file_size"],
        encoding=metadata.get("encoding"),
        created_at=metadata.get("created_at"),
        modified_at=metadata.get("modified_at"),
    )


def extract_sample_text(file_path: Path, file_type: str, encoding: str = "utf-8") -> str:
    """Extract sample text from file for language detection."""
    sample_text = ""
    try:
        if file_type == "txt" or file_type == "csv":
            with open(file_path, "r", encoding=encoding) as f:
                sample_text = f.read(1000)
        elif file_type == "pdf":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) > 0:
                    page = pdf.pages[0]
                    page_text = page.extract_text()
                    if page_text:
                        sample_text = page_text[:1000]
        elif file_type == "docx":
            import docx
            doc = docx.Document(file_path)
            para_index = 0
            while para_index < len(doc.paragraphs) and len(sample_text) < 1000:
                para = doc.paragraphs[para_index]
                if para.text:
                    sample_text += para.text + " "
                para_index += 1
            sample_text = sample_text[:1000]
    except Exception as e:
        logger.warning(f"Could not extract sample text from {file_path}: {e}")
    return sample_text


def detect_format_and_language(file_path: Path, sample_text: str = None) -> FileMetadata:
    """Main detector agent: detects file format and language."""
    logger.info(f"Detecting format and language for {file_path}")
    file_type = detect_file_type(file_path)
    metadata_dict = extract_metadata(file_path)
    if sample_text is None:
        sample_text = extract_sample_text(file_path, file_type, metadata_dict.get("encoding", "utf-8"))
    language = detect_language(sample_text)
    result = format_detection_result(file_path, file_type, language, metadata_dict)
    logger.info(f"Detected: type={file_type}, language={language}")
    return result

