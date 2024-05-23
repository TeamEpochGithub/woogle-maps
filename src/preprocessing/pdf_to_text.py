"""Extract text from PDFs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from pypdf import PdfReader

from src.logging.logger import Logger


@dataclass
class PdfToText(TransformationBlock, Logger):
    """Extract the text from a list of PDF files.

    :param files: The list of PDF files to extract the text from.
    """

    files: list[Path]

    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Extract text from a list of PDF files.

        :param data: The data to transform.
        :param transform_args: [UNUSED] Additional keyword arguments.
        :return: a DataFrame with the extracted text.
        """
        if not self.files:
            self.log_to_warning("The files are not provided.")
            return data

        self.log_to_terminal(f"Extracting text from {len(self.files)} PDF files...")

        docs = [self.extract_text_for_doc(file) for file in self.files]

        self.log_to_terminal(f"Extracted text from {len(docs)} PDF files.")
        return pd.DataFrame(docs).drop("index", errors="ignore")

    def extract_text_for_doc(self, filepath: Path) -> dict[str, Any]:
        """Extract the text from a PDF file.

        :param filepath: The path to the PDF file.
        :return: The extracted text.
        """
        reader = PdfReader(filepath)
        if reader.metadata and reader.metadata.title and str(reader.metadata.title) != "untitled":
            title = str(reader.metadata.title)
        else:
            title = filepath.name

        self.log_to_terminal(f"Extracting text from {title}...")

        date = reader.metadata.creation_date if reader.metadata else None
        full_text = "".join([page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) for page in reader.pages])
        return {"title": title, "date": date, "full_text": full_text}
