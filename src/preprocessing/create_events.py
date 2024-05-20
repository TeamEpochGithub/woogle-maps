"""Cluster the documents based on time and event similarity."""

from dataclasses import dataclass
from typing import Never, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from sklearn.metrics.pairwise import cosine_similarity

from src.logging.logger import Logger


@dataclass
class CreateEvents(TransformationBlock, Logger):
    """Create a narrative graph from the clusters and the memberships.

    :param period: period around the discard document
    """

    period: int = 4

    def find_most_similar(self, candidates: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> int:
        """Find the most similar candidate around a time period of target.

        :param candidates: the embeddings of the candidates.
        :param target: the embedding of the discarded document.
        :return: the index of the most similar candidate.
        """
        # Compute the cosine similarity between candidates and target
        similarities = cosine_similarity([target], candidates)[0]

        # Return the index of the most similar
        return int(np.argmax(similarities))

    def modify_adjacency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the adjacency list with the clusters.

        :param data: pandas dataframe with an "adj_list" column.
        :return: The data with the modified adjacency list.
        """
        # Extract the non-discarded documents
        data["index"] = data.index
        filtered = data.loc[~data["discarded"]].copy()

        # Convert the adjacency list with the clusters
        cluster_map = dict(zip(filtered["index"], filtered["clusters"], strict=False))
        data["adj_list"] = data["adj_list"].apply(lambda lst: [cluster_map[x] for x in lst])

        return data.drop("index", axis=1)

    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # noqa: DOC103
        """Cluster the documents based on the event and the date similarity.

        :param data: The data to transform.
        :param transform_args: [UNUSED] Additional keyword arguments.
        :return: The transformed data.
        """
        if "embed" not in data.columns or "discarded" not in data.columns or "adj_list" not in data.columns or "adj_weights" not in data.columns:
            self.log_to_warning("The input data does not have 'embed', 'discarded', 'adj_list' or 'adj_weights' columns. Unable to cluster the documents...")
            return data

        # Create a column for referencing the clusters
        data["clusters"] = range(data.shape[0])

        for idx, row in data.iterrows():
            idx = cast(int, idx)
            if row["discarded"]:
                # Extract the closest not-discarded documents
                data["distance"] = (data["clusters"] - idx).abs()
                candidates = data[(~data["discarded"]) & (data["clusters"] != idx)].nsmallest(self.period, "distance")

                # Find the most similar document in the candidates
                if not candidates.empty:
                    # Extract the embeddings from the candidates
                    candidate_embeddings = np.vstack(candidates["embed"].to_list())

                    # Find the index of the most similar document
                    similar_idx = self.find_most_similar(candidate_embeddings, row["embed"])

                    # Extract adjacency list, weights and the cluster index
                    adj_list: list[int] = candidates.iloc[similar_idx]["adj_list"]
                    adj_weight: list[float] = candidates.iloc[similar_idx]["adj_weights"]
                    cluster: int = candidates.iloc[similar_idx]["clusters"]

                    # Replace these values of the discarded document
                    data.at[idx, "adj_list"] = adj_list  # noqa: PD008
                    data.at[idx, "adj_weights"] = adj_weight  # noqa: PD008
                    data.at[idx, "clusters"] = cluster  # noqa: PD008

        # Cleanup the temporary column distance
        data = data.drop(["distance"], axis=1, errors="ignore")

        # Compute the unique clusters reference
        unique_clusters = data["clusters"].unique()
        unique_clusters.sort()

        # Create a mapping from old cluster labels to new cluster labels
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
        data["clusters"] = data["clusters"].map(mapping)

        # Convert the adjacency list to the cluster index
        data = self.modify_adjacency(data)

        return data.drop("index", errors="ignore")
