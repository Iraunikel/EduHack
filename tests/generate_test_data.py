"""Generate test data for Edalo pipeline."""

import csv
import logging
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import docx

import config

logger = logging.getLogger(__name__)


def generate_multilingual_content():
    """Generate multilingual educational content with common themes."""
    content = {
        "fr": [
            "Trop de théorie, pas assez de pratique.",
            "Les cours sont intéressants mais manquent d'exercices pratiques.",
            "J'aimerais plus d'activités pratiques et moins de cours magistraux.",
            "Le contenu théorique est bon mais nous avons besoin de plus de pratique.",
            "Les concepts sont bien expliqués mais l'application pratique est limitée.",
        ],
        "de": [
            "Zu viel Theorie, zu wenig Praxis.",
            "Die Kurse sind interessant, aber es fehlen praktische Übungen.",
            "Ich würde gerne mehr praktische Aktivitäten und weniger Vorlesungen haben.",
            "Der theoretische Inhalt ist gut, aber wir brauchen mehr Praxis.",
            "Die Konzepte sind gut erklärt, aber die praktische Anwendung ist begrenzt.",
        ],
        "en": [
            "Too much theory, not enough practice.",
            "The courses are interesting but lack practical exercises.",
            "I would like more practical activities and fewer lectures.",
            "The theoretical content is good but we need more practice.",
            "The concepts are well explained but practical application is limited.",
        ],
        "lb": [
            "Ze vill Theorie, ze wéineg Praxis.",
            "D'Kurse sinn interessant, mee et feelen praktesch Übungen.",
            "Ech géif gär méi praktesch Aktivitéiten a manner Coursen hunn.",
            "De theoreteschen Inhalt ass gutt, mee mir brauchen méi Praxis.",
            "D'Konzepter sinn gutt erkläert, mee d'praktesch Uwendung ass limitéiert.",
        ],
    }
    return content


def generate_txt_file(output_path: Path, language: str, content: list):
    """Generate a text file with multilingual content."""
    logger.info(f"Generating TXT file: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        line_index = 0
        while line_index < len(content):
            f.write(content[line_index] + "\n")
            line_index += 1
        f.write("\n")
        f.write("Student Feedback - Semester 2024\n")
        f.write(f"Language: {language}\n")


def generate_csv_file(output_path: Path, language: str, content: list):
    """Generate a CSV file with multilingual content."""
    logger.info(f"Generating CSV file: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "feedback", "language", "rating"])
        content_index = 0
        student_id = 1
        while content_index < len(content):
            writer.writerow([student_id, content[content_index], language, 3])
            content_index += 1
            student_id += 1


def generate_docx_file(output_path: Path, language: str, content: list):
    """Generate a DOCX file with multilingual content."""
    logger.info(f"Generating DOCX file: {output_path}")
    doc = docx.Document()
    doc.add_heading(f"Student Feedback - {language.upper()}", 0)
    doc.add_paragraph("Educational Data Collection")
    content_index = 0
    while content_index < len(content):
        doc.add_paragraph(content[content_index])
        content_index += 1
    doc.save(output_path)


def generate_pdf_file(output_path: Path, language: str, content: list):
    """Generate a PDF file with multilingual content."""
    logger.info(f"Generating PDF file: {output_path}")
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    y_position = height - 50
    c.setFont("Helvetica", 16)
    c.drawString(50, y_position, f"Student Feedback - {language.upper()}")
    y_position -= 30
    c.setFont("Helvetica", 12)
    content_index = 0
    while content_index < len(content):
        if y_position < 50:
            c.showPage()
            y_position = height - 50
        c.drawString(50, y_position, content[content_index])
        y_position -= 20
        content_index += 1
    c.save()


def generate_test_data():
    """Generate all test data files."""
    logger.info("Generating test data")
    test_data_dir = config.TEST_DATA_DIR
    test_data_dir.mkdir(parents=True, exist_ok=True)
    content = generate_multilingual_content()
    languages = ["fr", "de", "en", "lb"]
    file_index = 0
    while file_index < len(languages):
        language = languages[file_index]
        lang_content = content.get(language, [])
        txt_path = test_data_dir / f"feedback_{language}.txt"
        csv_path = test_data_dir / f"feedback_{language}.csv"
        docx_path = test_data_dir / f"feedback_{language}.docx"
        pdf_path = test_data_dir / f"feedback_{language}.pdf"
        generate_txt_file(txt_path, language, lang_content)
        generate_csv_file(csv_path, language, lang_content)
        generate_docx_file(docx_path, language, lang_content)
        generate_pdf_file(pdf_path, language, lang_content)
        file_index += 1
    logger.info(f"Test data generated in {test_data_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_test_data()

