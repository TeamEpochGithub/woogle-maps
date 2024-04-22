"""Load data from CSV files and PDF files using Polars and OCR."""

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.preprocessing.pdf_to_text import PdfToText

logger = logging.getLogger(__name__)

BODYTEXT_DTYPES = {
    "foi_documentId": str,
    "foi_pageNumber": int,
    "foi_bodyText": str,
    "foi_bodyTextOCR": str,
    "foi_hasOCR": bool,
    "foi_redacted": float,
    "foi_contourArea": float,
    "foi_textArea": float,
    "foi_charArea": float,
    "foi_percentageTextAreaRedacted": float,
    "foi_percentageCharAreaRedacted": float,
}

DOSSIER_DTYPES = {
    "dc_identifier": str,
    "dc_title": str,
    "dc_description": str,
    "dc_type": str,
    "dc_publisher_name": str,
    "dc_publisher": str,
    "dc_source": str,
    "foi_valuation": str,
    "foi_requestText": str,
    "foi_decisionText": str,
    "foi_isAdjourned": str,
    "foi_requester": str,
}

DOCUMENT_DTYPES = {
    "dc_identifier": str,
    "foi_dossierId": str,
    "dc_title": str,
    "foi_fileName": str,
    "dc_format": str,
    "dc_source": str,
    "dc_type": str,
    "foi_nrPages": float,
}


def load_dossier(raw_data_path: Path, dossier_id: str) -> pl.DataFrame:
    """Load and merge data from CSV files using Polars, with improvements for efficiency.

    :param raw_data_path: The path to the raw data files.
    :param dossier_id: The dossier ID to filter the data.
    :return: A merged dataframe with all data filtered by the selected dossier ID.
    """
    path = Path(raw_data_path)

    logger.info("Reading dossiers...")
    dossier_df = pl.read_csv(path / "woo_dossiers.csv", ignore_errors=True).with_columns([pl.col(column).cast(dtype) for column, dtype in DOSSIER_DTYPES.items()])

    logger.info("Reading documents...")
    document_df = pl.read_csv(path / "woo_documents.csv").with_columns([pl.col(column).cast(dtype) for column, dtype in DOCUMENT_DTYPES.items()])

    logger.info("Reading bodytext...")
    bodytext_df = pl.read_csv(path / "woo_bodytext.csv").with_columns([pl.col(column).cast(dtype) for column, dtype in BODYTEXT_DTYPES.items()])

    logger.info("Joining dataframes...")

    # Applying filters and transformations lazily, and collecting the results at the end
    filtered_dossier_df = dossier_df.filter(pl.col("dc_identifier") == dossier_id)
    filtered_document_df = document_df.filter(pl.col("foi_dossierId") == dossier_id)

    # Handling missing values
    filtered_document_df = filtered_document_df.with_columns(
        [
            pl.col("dc_source").fill_null(pl.lit("")),
            pl.col("dc_title").fill_null(pl.col("foi_fileName")).fill_null(pl.lit("")),
        ],
    )

    # Extracting publisher name assuming at least one record exists after filtering
    publisher_df = filtered_dossier_df.select(pl.col("dc_publisher_name"))
    publisher = publisher_df.get_column("dc_publisher_name")[0] if not publisher_df.is_empty() else ""
    filtered_document_df = filtered_document_df.with_columns(pl.lit(publisher).alias("dc_publisher_name"))

    # Aggregate body text
    bodytext_df = bodytext_df.with_columns(pl.col("foi_bodyTextOCR").fill_null(pl.lit("")))
    bodytext_df = bodytext_df.filter(pl.col("foi_documentId").is_in(document_df["dc_identifier"]))
    bodytext_df = bodytext_df.group_by("foi_documentId").agg(
        full_text=pl.col("foi_bodyTextOCR"),
    )
    bodytext_df = bodytext_df.with_columns(
        pl.col("full_text").map_elements(lambda x: "/n".join(x), return_dtype=pl.String).alias("full_text"),
    )

    merged_df = filtered_document_df.join(bodytext_df, left_on="dc_identifier", right_on="foi_documentId", how="left")

    logger.info("Reading complete.")
    return merged_df.rename({"dc_title": "title", "dc_source": "url", "dc_publisher_name": "publication"})


def load_pdf_dossier(upload_data_path: Path) -> pd.DataFrame:
    """Load and transform the data from PDF files using OCR.

    :param upload_data_path: The path to the raw upload data files.
    :return: A dataframe with the extracted text from the PDF files.
    """
    upload_dir_path = Path(upload_data_path)
    files = list(upload_dir_path.glob("*.pdf"))
    if not files:
        return pd.DataFrame()
    ocr_step = PdfToText(files=files)
    return ocr_step.transform(data=pd.DataFrame())


if __name__ == "__main__":
    data = load_dossier(raw_data_path=Path("../data/extracted"), dossier_id="nl.gm0867.2i.2023.11")
    data.to_pandas().to_pickle("../data/raw/nl.gm0867.2i.2023.11.pkl")
