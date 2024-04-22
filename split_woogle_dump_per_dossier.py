"""Split the Woogle dump into separate files per dossier."""

import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

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


@hydra.main(version_base=None, config_path="conf", config_name="split_woogle_dump_per_dossier")
def split_woogle_dump_per_dossier(cfg: DictConfig) -> None:
    """Split the Woogle dump into separate files per dossier.

    :param cfg: The configuration.
    """
    source = Path(cfg.source_path)
    target = Path(cfg.target_path)

    logger.info("Reading dossiers...")
    dossier_df = pl.read_csv(source / "woo_dossiers.csv").with_columns([pl.col(column).cast(dtype) for column, dtype in DOSSIER_DTYPES.items()])

    logger.info("Reading documents...")
    document_df = pl.read_csv(source / "woo_documents.csv").with_columns([pl.col(column).cast(dtype) for column, dtype in DOCUMENT_DTYPES.items()])

    logger.info("Reading bodytext...")
    bodytext_df = pl.read_csv(source / "woo_bodytext.csv").with_columns([pl.col(column).cast(dtype) for column, dtype in BODYTEXT_DTYPES.items()])

    bodytext_df = bodytext_df.group_by("foi_documentId").agg(
        full_text=pl.col("foi_bodyTextOCR").fill_null(""),
    )
    bodytext_df = bodytext_df.with_columns(
        full_text=pl.col("full_text").map_elements(lambda x: "".join(x), return_dtype=pl.String),
    )

    dossier_ids = dossier_df["dc_identifier"].unique().to_list()
    for dossier_id in dossier_ids:
        logger.info("Processing dossier %s", dossier_id)
        dossier_df_filtered = dossier_df.filter(pl.col("dc_identifier") == dossier_id)
        document_df_filtered = document_df.filter(pl.col("foi_dossierId") == dossier_id)

        document_df_filtered = document_df_filtered.with_columns(
            title=pl.col("dc_title"),
            publication=pl.lit(dossier_df_filtered.select(pl.first("dc_publisher_name"))),
            url=pl.lit(dossier_df_filtered.select(pl.first("dc_publisher_name"))),
        )

        merged_df = document_df_filtered.join(bodytext_df, left_on="dc_identifier", right_on="foi_documentId", how="left")
        if merged_df.height <= cfg.min_docs_per_dossier:
            logger.info("%s only has %s documents. Skipping...", dossier_id, merged_df.height)
            continue
        merged_df.write_csv(target / "csv" / f"{dossier_id}.csv")
        merged_df.to_pandas().to_pickle(target / f"{dossier_id}.pkl")


if __name__ == "__main__":
    split_woogle_dump_per_dossier()
