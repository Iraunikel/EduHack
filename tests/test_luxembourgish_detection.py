"""Test Luxembourgish language detection with Polyglot."""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.detector import detect_language

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_luxembourgish_samples():
    """Test language detection with Luxembourgish text samples."""
    luxembourgish_texts = [
        "Ze vill Theorie, ze wéineg Praxis.",
        "D'Kurse sinn interessant, mee et feelen praktesch Übungen.",
        "Ech géif gär méi praktesch Aktivitéiten a manner Coursen hunn.",
        "De theoreteschen Inhalt ass gutt, mee mir brauchen méi Praxis.",
        "D'Konzepter sinn gutt erkläert, mee d'praktesch Uwendung ass limitéiert.",
        "Moien, wéi geet et dir?",
        "Dëst ass e Beispilltext op Lëtzebuergesch.",
        "Ech si frou, Iech ze treffen.",
    ]
    logger.info("Testing Luxembourgish language detection")
    passed = 0
    failed = 0
    index = 0
    while index < len(luxembourgish_texts):
        text = luxembourgish_texts[index]
        detected = detect_language(text)
        if detected == "lb":
            logger.info(f"✓ PASS: Detected 'lb' for: {text[:50]}...")
            passed += 1
        else:
            logger.warning(f"✗ FAIL: Expected 'lb', got '{detected}' for: {text[:50]}...")
            failed += 1
        index += 1
    return passed, failed


def test_other_languages():
    """Test language detection with other languages for compatibility."""
    test_cases = [
        ("fr", "Trop de théorie, pas assez de pratique."),
        ("de", "Zu viel Theorie, zu wenig Praxis."),
        ("en", "Too much theory, not enough practice."),
    ]
    logger.info("Testing other languages for backward compatibility")
    passed = 0
    failed = 0
    index = 0
    while index < len(test_cases):
        expected_lang, text = test_cases[index]
        detected = detect_language(text)
        if detected == expected_lang:
            logger.info(f"✓ PASS: Detected '{expected_lang}' correctly")
            passed += 1
        else:
            logger.warning(f"✗ FAIL: Expected '{expected_lang}', got '{detected}'")
            failed += 1
        index += 1
    return passed, failed


def test_file_detection():
    """Test language detection with actual test files."""
    logger.info("Testing language detection with test files")
    test_data_dir = Path(__file__).parent / "test_data"
    test_files = {
        "lb": test_data_dir / "feedback_lb_comprehensive.txt",
        "fr": test_data_dir / "feedback_fr_comprehensive.txt",
        "de": test_data_dir / "feedback_de_comprehensive.txt",
        "en": test_data_dir / "feedback_en_comprehensive.txt",
    }
    passed = 0
    failed = 0
    for expected_lang, file_path in test_files.items():
        if not file_path.exists():
            logger.warning(f"Test file not found: {file_path}")
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(500)
            detected = detect_language(content)
            if detected == expected_lang:
                logger.info(f"✓ PASS: File {file_path.name} detected as '{expected_lang}'")
                passed += 1
            else:
                logger.warning(f"✗ FAIL: File {file_path.name} - Expected '{expected_lang}', got '{detected}'")
                failed += 1
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            failed += 1
    return passed, failed


def run_all_tests():
    """Run all language detection tests."""
    logger.info("=" * 60)
    logger.info("Luxembourgish Language Detection Tests")
    logger.info("=" * 60)
    total_passed = 0
    total_failed = 0
    logger.info("\n1. Testing Luxembourgish text samples:")
    logger.info("-" * 60)
    passed, failed = test_luxembourgish_samples()
    total_passed += passed
    total_failed += failed
    logger.info(f"\nLuxembourgish samples: {passed} passed, {failed} failed")
    logger.info("\n2. Testing other languages (backward compatibility):")
    logger.info("-" * 60)
    passed, failed = test_other_languages()
    total_passed += passed
    total_failed += failed
    logger.info(f"\nOther languages: {passed} passed, {failed} failed")
    logger.info("\n3. Testing with actual test files:")
    logger.info("-" * 60)
    passed, failed = test_file_detection()
    total_passed += passed
    total_failed += failed
    logger.info(f"\nFile detection: {passed} passed, {failed} failed")
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY: {total_passed} passed, {total_failed} failed")
    logger.info("=" * 60)
    if total_failed > 0:
        logger.error("Some tests failed!")
        logger.info("Make sure Polyglot is installed and language models are downloaded:")
        logger.info("  pip install polyglot pycld2")
        logger.info("  polyglot download LANG:lb")
        logger.info("  polyglot download LANG:fr")
        logger.info("  polyglot download LANG:de")
        logger.info("  polyglot download LANG:en")
        return 1
    logger.info("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())

