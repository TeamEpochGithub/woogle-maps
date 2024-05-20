"""Generate a summary of the clusters."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Never, override

import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from lexrank import LexRank

from src.logging.logger import Logger


@dataclass
class ClusterExplainer(TransformationBlock, Logger):
    """Generate a summary and extracts the entities from the cluster.

    :param threshold: the number of important sentences in the cluster.
    """

    threshold: int = 10

    def rank_sentences(self, sentences: Iterable[str]) -> list[str]:
        """Rank the sentences based on the LexRank model.

        :param sentences: the filtered sentences in a cluster.
        :return: the most important sentences in the cluster.
        """
        # Initialize the model and rank the sentences
        model = LexRank(sentences)
        scores = model.rank_sentences(sentences)
        ranked = sorted(zip(scores, sentences, strict=False), reverse=True)

        # Extract the most important sentences from the cluster
        important: list[str] = [sentence for _, sentence in ranked]

        return important[: self.threshold]

    @override  # type: ignore[misc]
    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # noqa: DOC103
        """Generate a summary and extract the entities from the cluster.

        :param data: the input data.
        :param transform_args: [UNUSED] the transformation arguments.
        :return: the transformed data.
        """
        if "clusters" not in data.columns or "summary" not in data.columns:
            self.log_to_warning("No clusters or summary columns found in the data. Not generating an explanation.")
            return data

        try:
            # Create a dataframe for the clusters and group the sentences
            filtered = data.groupby("clusters").agg(sentences=("summary", "sum")).reset_index()

            # Extract and translate the most important sentences in each cluster
            filtered["abstract"] = filtered["sentences"].apply(self.rank_sentences)

            # Remove the unnecessary columns and merge the dataframe
            filtered = filtered[["clusters", "abstract"]]
            data = data.merge(filtered, on="clusters", how="left", validate=None)
        except ValueError as e:  # ValueError: documents are not informative
            self.log_to_warning(f"Error in extracting important sentences: {e}")

        return data
