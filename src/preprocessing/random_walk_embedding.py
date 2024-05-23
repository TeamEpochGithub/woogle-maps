"""Extract the embeddings from the topical and date similarities."""

from dataclasses import dataclass, field
from typing import Never

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from gensim.models import Word2Vec
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from external.random_walks import RandomWalk
from src.logging.logger import Logger


@dataclass
class RandomWalkEmbedding(TransformationBlock, Logger):
    """Create the embeddings of documents based on the topics and the dates.

    :param threshold: sparsifies the similarity matrix.
    :param num_walks: the number of random walks per node.
    :param walk_length: the length of a random walk.
    :param dimension: the final embedding dimension.
    """

    threshold: int = 15
    num_walks: int = 50
    walk_length: int = 120
    dimension: int = 120

    _model: Word2Vec = field(init=False, repr=False)

    def sparse_matrix(self, similarity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert the similarity matrix to a sparse matrix.

        :param similarity: the similarity matrix of the fully connected graph.
        :return: the sparse similarity matrix.
        """
        # Compute the index of the highest elements
        top_index = np.argsort(similarity, axis=1)[:, -self.threshold :]

        # Create a mask to keep only the top n elements
        rows = np.arange(similarity.shape[0]).reshape(-1, 1)
        mask = np.zeros_like(similarity, dtype=bool)
        mask[rows, top_index] = True

        return np.where(mask, similarity, 0)

    def topic_graph(self, data: pd.DataFrame) -> nx.Graph:  # type: ignore[type-arg]
        """Compute the network graph from the topical similarity.

        :param data: A pandas dataframe with "topic_dist" column.
        :return: The network graph of the topical similarity.
        """
        topics = data["topical_dist"]

        # Initialize the date similarity matrix
        similarity = np.zeros((data.shape[0], data.shape[0]))

        for i, _ in data.iterrows():
            for j, _ in data.iterrows():
                divergence = jensenshannon(topics[i], topics[j])  # type: ignore[call-overload]
                similarity[i][j] = similarity[j][i] = 1 - divergence  # type: ignore[call-overload]

        # Compute the sparse similarity matrix
        similarity = self.sparse_matrix(similarity)

        return nx.from_numpy_array(similarity)

    def event_graph(self, data: pd.DataFrame) -> nx.Graph | None:  # type: ignore[type-arg]
        """Compute the network graph from the summaries.

        :param data: A pandas dataframe with "summary" column.
        :return: The network graph of the event similarity.
        """
        summary = data["summary"].tolist()
        summary = [" ".join(sentences) for sentences in summary if sentences]

        if not summary:
            self.log_to_warning("No summaries found in the data.")
            return None

        # Compute the similarity matrix with cosine
        encoder = TfidfVectorizer()
        embeddings = encoder.fit_transform(summary)
        similarity = cosine_similarity(embeddings, embeddings)

        # Compute the sparse similarity matrix
        similarity = self.sparse_matrix(similarity)

        return nx.from_numpy_array(similarity)

    def generate_walks(self, data: pd.DataFrame) -> list[list[str]]:
        """Generate the random walks based on the topic and date similarities.

        :param data: A pandas dataframe with "topics_dist" and "summary" columns.
        :return: the list containing the random walks.
        """
        if "topical_dist" not in data.columns:
            self.log_to_warning("The data does not contain a 'topical_dist' column.")
            topical_graph = None
        else:
            topical_graph = self.topic_graph(data)

        if "summary" not in data.columns:
            self.log_to_warning("The data does not contain a 'summary' column.")
            evental_graph = None
        else:
            evental_graph = self.event_graph(data)

        # Compute the random walks for the summary and the title
        topical_walk = RandomWalk(topical_graph, walk_length=self.walk_length, num_walks=self.num_walks, workers=6).walks if topical_graph else []
        event_walk = RandomWalk(evental_graph, walk_length=self.walk_length, num_walks=self.num_walks, workers=6).walks if evental_graph else []

        return topical_walk + event_walk

    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Compute embeddings after random walk.

        :param data: The input dataframe.
        :param transform_args: [UNUSED] Additional keyword arguments.
        :return: The transformed data.
        """
        # Compute the date similarity and generate the random walks
        random_walks = self.generate_walks(data)

        if not random_walks:
            self.log_to_warning("No random walks generated. Not computing embeddings...")
            return data

        # Train the deep walk algorithm on the random walks
        self._model = Word2Vec(sentences=random_walks, vector_size=self.dimension, window=5, min_count=1, sg=1)

        # Compute the embedding of the documents
        data["embed"] = [self._model.wv[node].tolist() for node in self._model.wv.index_to_key]

        return data.drop("index", errors="ignore")
