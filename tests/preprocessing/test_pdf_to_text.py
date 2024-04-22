"""Test cases for the PdfToText class."""

import os
from pathlib import Path
from unittest import TestCase

import pandas as pd
from reportlab.pdfgen import canvas

from src.preprocessing.pdf_to_text import PdfToText


class PdfToTextTest(TestCase):
    """Test the PdfToText class."""

    @classmethod
    def setup_class(cls) -> None:
        """Create a sample PDF file."""
        # create the file
        with open("sample.pdf", "wb") as f:
            c = canvas.Canvas(f)
            text = "This is a sample PDF file."
            c.drawString(100, 750, text)
            c.showPage()
            c.save()

    def test__pdf_to_text(self) -> None:
        """Test the extract_text_for_doc method."""
        pdf_path = "sample.pdf"
        pdf_to_text = PdfToText()
        extracted_dict = pdf_to_text.extract_text_for_doc(Path("sample.pdf"))
        assert extracted_dict["full_text"] == "This is a sample PDF file."

    def test__custom_transform(self) -> None:
        """Test the custom_transform method."""
        pdf_to_text = PdfToText()
        data = pdf_to_text.transform(pd.DataFrame(columns=["title", "date", "publication", "url", "full_text"]), files=[Path("sample.pdf")])
        assert data.iloc[0]["title"] == "sample.pdf"
        assert data.iloc[0]["full_text"] == "This is a sample PDF file."

    @classmethod
    def teardown_class(cls) -> None:
        """Remove the sample PDF file."""
        os.remove("sample.pdf")
