"""Perform the linear programming on the clusters."""

from dataclasses import dataclass
from typing import Never, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

from src.logging.logger import Logger


@dataclass
class LinearProgramming(TransformationBlock, Logger):
    """Create a narrative graph from the clusters and the memberships.

    :param min_cover: the minimum coverage of the memberships.
    :param K: the expected length of the main story.
    :param threshold: remove low edges and nodes.
    """

    min_cover: float = 0.6
    K: int = 10
    threshold: float = 0.01

    def _create_variables(self, n_nodes: int, n_clusters: int) -> None:
        """Create the names of the variables for LP.

        :param n_nodes: the number of nodes in LP.
        :param n_clusters: the number of clusters in LP.
        """
        # Create the windows based on the clusters
        self._window_ij = {}
        self._window_ji = {}

        for i in range(n_nodes):
            self._window_ij[i] = list(range(i + 1, n_nodes))

        for j in range(n_nodes):
            self._window_ji[j] = list(range(j))

        # Create the names of the nodes and the edges
        self._var_i = []
        self._var_ij = []

        for i in range(n_nodes):
            self._var_i.append(str(i))

            for j in self._window_ij[i]:
                self._var_ij.append(f"{i}_{j}")

        # Create the names of the clusters
        self._var_k = [str(k) for k in range(n_clusters)]

    def _create_problem(self, similarity: npt.NDArray[np.float64], memberships: npt.NDArray[np.float64]) -> LpProblem:
        """Create the linear programming problem.

        :param similarity: the similarity matrix of the nodes.
        :param memberships: the memberships of the nodes.
        :return: the linear programming problem.
        """
        # Compute the number of nodes and clusters
        n_nodes = similarity.shape[0]
        n_clusters = memberships.shape[1]

        # Create the linear programming variables
        self._create_variables(n_nodes, n_clusters)

        # The minimum edge to be maximized
        minedge = LpVariable("minedge", lowBound=0, upBound=1)

        # The weights of the edges , nodes and clusters
        node_active = LpVariable.dicts("node_active", self._var_i, lowBound=0, upBound=1)
        edge_active = LpVariable.dicts("node_next", self._var_ij, lowBound=0, upBound=1)
        cluster_active = LpVariable.dicts("clust_active", self._var_k, lowBound=0, upBound=1)

        # Create the problem variable to contain the date
        prob = LpProblem("StoryChainProblem", LpMaximize)
        prob += minedge, "WeakestLink"

        # Create the constraint of the start and end
        prob += node_active["0"] == 1, "InitialNode"
        prob += node_active[str(n_nodes - 1)] == 1, "FinalNode"

        # Create constraint of the bipolar orientation
        prob += lpSum([node_active[i] for i in self._var_i]) == self.K, "KNodes"

        # Chain constraint of the start and the other nodes
        for j in range(1, n_nodes):
            prob += lpSum([edge_active[f"{i}_{j}"] for i in self._window_ji[j]]) == node_active[str(j)], "InEdgeReq" + str(j)

        # Chain constraint of the end and the other nodes
        for i in range(n_nodes - 1):
            prob += lpSum([edge_active[f"{i}_{j}"] for j in self._window_ij[i]]) == node_active[str(i)], "OutEdgeReq" + str(i)

        # Coverage constraint for the clusters
        prob += lpSum([cluster_active[str(k)] for k in self._var_k]) >= n_clusters * self.min_cover, "MinCover"
        for k in range(n_clusters):
            prob += (
                cluster_active[str(k)]
                == lpSum([edge_active[f"{i}_{j}"] * np.sqrt(memberships[i, k] * memberships[j, k]) for i in range(n_nodes - 1) for j in self._window_ij[i]]),
                "CoverDef" + str(k),
            )

        # The objective function consisting of min_edge
        for i in range(n_nodes):
            for j in self._window_ij[i]:
                prob += minedge <= 1 - edge_active[f"{i}_{j}"] + similarity[i, j]

        return prob

    def _solving_problem(self, similarity: npt.NDArray[np.float64], memberships: npt.NDArray[np.float64]) -> dict[str, float]:
        """Create the linear programming problem.

        :param similarity: the similarity matrix of the nodes.
        :param memberships: the memberships of the nodes.
        :return: the values of the nodes and the edges.
        """
        # Create and solve the linear programming problem
        problem = self._create_problem(similarity, memberships)
        problem.solve(PULP_CBC_CMD(mip=False, warmStart=True))

        # Extract the nodes, the edges and the clusters
        variable_dict = {}

        for v in problem.variables():
            if "node_next" in v.name or "node_active" in v.name:
                variable_dict[v.name] = min(max(v.varValue, 0), 1)

        return variable_dict

    def _create_similarity(self, data: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Create the similarity matrix from the event and topic similarities.

        :param data: the pandas dataframe with embed and memberships columns.
        :return similarity: the similarity matrix of the nodes.
        """
        # Extract the membership and embeddings from the data
        memberships = np.array(data["memberships"].tolist())
        embeddings = np.array(data["embed"].tolist())

        # Compute the cluster similarity matrix
        cluster = distance.cdist(memberships, memberships, lambda u, v: distance.jensenshannon(u, v, base=2.0))

        # Compute and normalize the event similarity matrix
        event = np.clip(cosine_similarity(embeddings), -1, 1)
        event = 1 - np.arccos(event) / np.pi

        # Select the maximum not on the diagonal
        mask = np.ones(event.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        # Compute the minimum and this maximum
        max_value = event[mask].max()
        min_value = event[mask].min()

        # Normalize the event similarity between 0 and 1
        event = (event - min_value) / (max_value - min_value)
        event = np.clip(event, 0, 1)

        return np.sqrt(event * cluster)

    @override
    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # type: ignore[misc]  # noqa: DOC103
        """Create the adjacency list and weights based on the solution.

        :param data: the pandas dataframe containing the embeddings.
        :param transform_args: [UNUSED] Additional keyword arguments.
        :return data: the pandas dataframe containing adjacency list and weights.
        """
        if "memberships" not in data.columns or "embed" not in data.columns:
            self.log_to_warning("The data does not contain 'memberships' and/or 'embed' columns. Unable to compute the narrative graph...")
            return data

        # Compute the similarity matrix and the memberships
        similarity = self._create_similarity(data)
        memberships = np.array(data["memberships"].tolist())

        # Perform linear programming and compute the dictionary
        variable_dict = self._solving_problem(similarity, memberships)

        # Compute the adjacency list and weights of the documents
        adj_list: list[list[int]] = [[]] * len(data)
        adj_weights: list[list[float]] = [[]] * len(data)
        discarded: list[bool] = [False] * len(data)

        for i in range(len(data)):
            # Check whether the node is active or not
            coherence = variable_dict[f"node_active_{i}"]
            if coherence > self.threshold:
                # Extract the neighboring nodes and their values
                coherence_list = []
                probability = []

                for j in self._window_ij[i]:
                    name = f"node_next_{i}_{j}"
                    probability.append(variable_dict[name])
                    coherence_list.append(variable_dict[f"node_active_{j}"])

                # Compute the adjacency weight with normalized probabilities
                idx_list = [self._window_ij[i][idx] for idx, e in enumerate(probability) if round(e, 8) != 0 and e > self.threshold and coherence_list[idx] > self.threshold]

                nz_prob = [e for idx, e in enumerate(probability) if round(e, 8) != 0 and e > self.threshold and coherence_list[idx] > self.threshold]

                # Update the adjacency list and weights
                discarded[i] = False
                adj_list[i] = idx_list
                adj_weights[i] = [nz_prob[j] / sum(nz_prob) for j in range(len(nz_prob))]
            else:
                discarded[i] = True
                adj_list[i] = []
                adj_weights[i] = []

        data["discarded"] = discarded
        data["adj_list"] = adj_list
        data["adj_weights"] = adj_weights

        return data.drop("index", errors="ignore")
