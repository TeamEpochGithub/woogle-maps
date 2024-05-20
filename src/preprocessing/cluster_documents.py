"""Compute the membership vectors for each cluster."""

from dataclasses import dataclass
from typing import Never

import hdbscan
import numpy as np
import numpy.typing as npt
import pandas as pd
import umap
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock

from src.logging.logger import Logger


@dataclass
class ClusterDocuments(TransformationBlock, Logger):
    """Cluster the documents based on the event and the date similarity.

    :param: periods: how many time periods to consider.
    """

    periods: int = 4

    def _compute_variable(self, num_docs: int) -> tuple[int, int, str]:
        """Compute the number of neighbors and the minimum cluster size.

        :param num_docs: the number of documents to be clustered.
        :return: the number of neighbors and the minimum cluster size.
        """
        # Compute the minimum cluster size
        min_cluster = max(5 * round(np.sqrt(num_docs) / 10), 2)

        # Compute the number of neighbors and init
        if num_docs <= 40:
            neighbors = 2
            init = "random"
        elif num_docs <= 120:
            neighbors = 10
            init = "spectral"
        else:
            neighbors = min_cluster
            init = "spectral"

        return min_cluster, neighbors, init

    def _compute_embeddings(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert the embeddings to two dimensions using UMAP.

        :param embeddings: numpy array containing the embeddings.
        :return new_embeddings: numpy array containing the 2-dimensional embeddings.
        """
        # Compute the number of neighbors and init
        _, neighbors, init = self._compute_variable(embeddings.shape[0])

        # Train and perform UMAP on the embeddings
        encoder = umap.UMAP(n_neighbors=neighbors, min_dist=0.01, init=init)

        return encoder.fit_transform(embeddings)

    def _compute_memberships(self, embeddings: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the clusters of the embeddings using HDBSCAN.

        :param embeddings: numpy array containing the embeddings.
        :return: the cluster labels and the membership vectors.
        """
        # Compute the minimum clusters size for the embeddings
        min_cluster, _, _ = self._compute_variable(embeddings.shape[0])

        # Convert the embeddings to two dimensions using UMAP
        clusterable = self._compute_embeddings(embeddings)

        # Train and compute the clusters based on HDBSCAN
        model = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=min_cluster, prediction_data=True, cluster_selection_method="leaf")
        labels = model.fit_predict(clusterable)

        # Compute and normalize the membership vectors
        memberships = hdbscan.prediction.all_points_membership_vectors(model)

        if len(memberships.shape) > 1:
            # Remove any low probability
            ratio = 1 / memberships.shape[1]
            num_clust = memberships.shape[1]
            memberships[memberships < ratio] = 0

            # If any row is full of zeros replace with uniform distribution
            memberships[np.all(memberships == 0, axis=1)] = ratio * np.ones(num_clust)
            memberships = memberships / memberships.sum(axis=1)[:, np.newaxis]

        else:
            memberships = np.ones((memberships.shape[0], 1))

        return labels + 1, memberships

    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # noqa: DOC103
        """Cluster the documents based on the event and the date similarity.

        :param data: The data to transform.
        :param transform_args: [UNUSED] Additional keyword arguments.
        :return: The transformed data.
        """
        if "embed" not in data.columns:
            self.log_to_warning("The input data does not have an 'embed' column. Unable to cluster documents...")
            return data

        # Compute the memberships of the documents in the dossier
        embeddings = np.array(data["embed"].tolist())

        # Compute the membership vectors of the documents
        try:
            _, memberships = self._compute_memberships(embeddings)
            data["memberships"] = list(memberships)
        except Exception as e:  # noqa: BLE001
            self.log_to_warning(f"Unable to compute the memberships of the documents in the dossier: {e}")

        return data.drop("index", errors="ignore")
