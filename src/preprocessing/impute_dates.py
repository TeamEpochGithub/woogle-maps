"""Impute missing dates by filling them with the most similar embedding."""

from dataclasses import dataclass
from typing import Any, override
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from sklearn.metrics.pairwise import cosine_similarity

from src.logging.logger import Logger


@dataclass
class ImputeDates(TransformationBlock, Logger):
    """Impute missing dates.

    Replaces NaN values in the date column with one from the closest document embedding.
    """

    @override
    def custom_transform(self, data: pd.DataFrame, **transform_args: Any) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Set the missing dates to the fill_date.

        :param data: The data to transform.
        :param transform_args: Additional keyword arguments (UNUSED).
        :return: The transformed data.
        """
        if "date" not in data.columns:
            self.log_to_warning("The input data does not have a 'date' column. Unable to impute dates.")
            return data
        if "embed" not in data.columns:
            self.log_to_warning("The input data does not have an 'embed' column. Unable to impute dates.")
            return data

        self.log_to_terminal(f"Imputing {data["date"].isna().sum()} missing dates...")

        all_embeddings = np.stack(data["embed"].to_numpy())
        not_na_rows = data[data["date"].notna()]
        if len(not_na_rows) <= 0:
            self.log_to_warning("None of the documents have a date. Unable to impute dates.")
            return data

        not_na_embeddings = np.stack(not_na_rows["embed"].to_numpy())

        similarities = cosine_similarity(all_embeddings, not_na_embeddings)
        most_similar_indices = np.argmax(similarities, axis=1)
        most_similar_dates = not_na_rows["date"].iloc[most_similar_indices]
        most_similar_dates.index = data.index

        data = data.fillna({"date": most_similar_dates})

        data["date"] = data["date"].apply(lambda x: x.replace(tzinfo=ZoneInfo("Europe/Amsterdam")) if pd.notna(x) else x)

        if data["date"].isna().sum() != 0:  # Sanity check
            self.log_to_warning(f"There are still {data["date"].isna().sum()} missing dates in the data! This may cause problems later on. Continuing anyway...")

        return data.sort_values("date").reset_index().drop("index", errors="ignore")
