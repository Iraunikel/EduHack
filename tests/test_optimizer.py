from pathlib import Path

from agents import optimizer


def test_extract_text_json_handles_nested_structures(tmp_path: Path):
    json_path = tmp_path / "data.json"
    json_path.write_text(
        """
{
  "course": {
    "title": "Physics 101",
    "feedback": [
      {"lang": "fr", "comment": "Trop de théorie"},
      {"lang": "de", "comment": "Mehr Praxis bitte"}
    ],
    "score": 4.2
  }
}
        """.strip()
    )

    extracted = optimizer.extract_text(json_path, "json")

    assert "Physics 101" in extracted
    assert "Trop de théorie" in extracted
    assert "Mehr Praxis bitte" in extracted
    assert "4.2" in extracted
