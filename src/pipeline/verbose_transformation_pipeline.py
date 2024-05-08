"""Module containing the VerboseTransformationPipeline class."""

from pathlib import Path
from typing import Any

import pandas as pd
from epochalyst.pipeline.model.transformation.transformation import TransformationPipeline

from src.logging.logger import Logger
from src.utils.fix_dtypes import fix_dtypes


class VerboseTransformationPipeline(TransformationPipeline, Logger):
    """A verbose transformation pipeline that logs to the terminal."""

    def get_cache_exists(self, cache_args: dict[str, Any]) -> bool:
        """Check if some cache exists.

        :param cache_args: The cache arguments.
        :return: Whether the cache exists.
        """
        if not self.steps:
            return False
        if self.cache_exists(name=self.get_hash(), cache_args=cache_args):
            return True
        return any(step.cache_exists(name=step.get_hash(), cache_args=cache_args) for step in self.steps)

    def run_dossier_pipeline(self, dossier_id: str, raw_data_path: Path, processed_data_path: Path, final_data_path: Path) -> pd.DataFrame:
        """Run the pipeline on the data.

        It also creates a cache after each step in the pipeline.

        :param dossier_id: The dossier ID.
        :param raw_data_path: The path to the raw data.
        :param processed_data_path: The path to the processed data.
        :param final_data_path: The path to the final data.
        :return: The processed data.
        """
        processed_data_path = Path(processed_data_path) / dossier_id
        processed_data_path.mkdir(parents=True, exist_ok=True)
        final_data_path = Path(final_data_path) / dossier_id
        final_data_path.mkdir(parents=True, exist_ok=True)

        cache_args = {
            "output_data_type": "pandas_dataframe",
            "storage_type": ".pkl",
            "storage_path": processed_data_path.as_posix(),
        }

        cache_exists = self.get_cache_exists(cache_args)
        dossier_raw: pd.DataFrame | None = None
        if cache_exists:
            self.log_to_terminal(f"Cache exists for {dossier_id}. Using the cache.")
        else:
            self.log_to_terminal(f"Cache does not exist for {dossier_id}. Loading data...")
            dossier_raw = pd.read_pickle(raw_data_path / f"{dossier_id}.pkl")  # noqa: S301

        transform_args = {step.__class__.__name__: {"cache_args": cache_args} for step in self.get_steps()}

        data = self.transform(dossier_raw, cache_args=cache_args, **transform_args)
        return fix_dtypes(data)
