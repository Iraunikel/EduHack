from pathlib import Path

import pytest

import agents.detector as detector


@pytest.fixture(autouse=True)
def reset_detector_state(monkeypatch):
    """Ensure detector caches are reset between tests."""
    monkeypatch.setattr(detector, "_magic_instance", None, raising=False)
    monkeypatch.setattr(detector, "_magic_error", False, raising=False)
    yield
    monkeypatch.setattr(detector, "_magic_instance", None, raising=False)
    monkeypatch.setattr(detector, "_magic_error", False, raising=False)


def test_detect_file_type_prefers_suffix(tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    assert detector.detect_file_type(file_path) == "pdf"


def test_detect_file_type_uses_mimetypes_when_magic_unavailable(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "report.data"
    file_path.write_text("id,name\n1,Alice\n")

    monkeypatch.setattr(detector, "_magic_error", True, raising=False)
    monkeypatch.setattr(detector.mimetypes, "guess_type", lambda name: ("text/csv", None))

    assert detector.detect_file_type(file_path) == "csv"


def test_detect_file_type_handles_magic_failure(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "unknown.bin"
    file_path.write_bytes(b"binary data")

    class BrokenMagic:
        def from_file(self, path: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(detector, "_magic_instance", BrokenMagic(), raising=False)

    assert detector.detect_file_type(file_path) == "unknown"
