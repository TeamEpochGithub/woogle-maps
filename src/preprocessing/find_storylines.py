"""Find the storylines in the data."""

from dataclasses import dataclass
from typing import Any, override

import pandas as pd
import polars as pl
import rustworkx as rx
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock

from src.logging.logger import Logger
from src.utils.construct_graph import build_graph_from_adj_list


@dataclass
class FindStorylines(TransformationBlock, Logger):
    """Find the storylines in the data.

    The data is expected to have an "adj_list" and "adj_weights" column that contains the adjacency list of the graph.
    It will return the data with a "storyline" column that contains the storyline index for each row.
    """

    @override
    def custom_transform(self, data: pd.DataFrame | pl.DataFrame, **transform_args: Any) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Find the storylines in the data.

        :param data: The data to find the storylines in.
        :param transform_args: Additional keyword arguments (UNUSED).
        :return: The data with the storylines.
        """
        if "adj_list" not in data.columns or "adj_weights" not in data.columns:
            self.log_to_warning("The input data does not have an 'adj_list' and 'adj_weights' column. Unable to find storylines...")
            return data

        data.drop("index", errors="ignore")

        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

        full_data = None
        if "clusters" in data.columns:
            full_data = data.clone().drop(["adj_list", "adj_weights"])
            data = (
                data.group_by("clusters")
                .agg(
                    [
                        pl.col("adj_list").first().alias("adj_list"),
                        pl.col("adj_weights").first().alias("adj_weights"),
                    ],
                )
                .sort("clusters")
            )

        self.log_to_terminal(f"Finding storylines in {data.height} nodes...")

        graph = build_graph_from_adj_list(data["adj_list"].to_list(), data["adj_weights"].to_list())
        storylines: list[list[int]] = find_storylines(graph)
        storyline_index_map = {row: idx for idx, rows in enumerate(storylines) for row in rows}

        data = data.with_row_index(name="idx").with_columns(storyline=pl.col("idx").replace(storyline_index_map)).drop("idx")

        self.log_to_terminal(f"Found storylines {storylines} in the data.")

        if full_data is not None:
            data = data.join(full_data, on="clusters")

        return data.to_pandas().drop("index", errors="ignore")


def find_storylines(graph: rx.PyDiGraph) -> list[list[int]]:  # type: ignore[type-arg]
    """Find the storylines in the graph.

    The storylines are found by repeatedly checking the shortest path from the source node to every other node and taking the longest path.

    :param graph: The graph to find the storylines in.
    :return: The storylines as list of lists of node IDs.
    """
    if len(graph.nodes()) <= 0:  # Edge case: no nodes, so no storylines
        return []
    if len(graph.nodes()) == 1:  # Edge case: only one node, so it is its own storyline
        return [graph.nodes()]
    if len(graph.edges()) <= 0:  # Edge case: no edges left, so each node is its own storyline
        return [[node] for node in graph.nodes()]

    temp_graph = graph.copy()
    storylines = []

    while len(temp_graph.nodes()) > 0:
        # Find the shortest path from the source node to every other node. We use Bellman-Ford instead of Dijkstra because the weights can be negative.
        shortest_paths = rx.bellman_ford_shortest_paths(temp_graph, temp_graph.nodes()[0], weight_fn=(lambda x: x), default_weight=0.0)

        if len(shortest_paths) <= 0:  # No paths, so this node is its own storyline
            storylines.append([temp_graph.nodes()[0]])
            temp_graph.remove_node(temp_graph.nodes()[0])
            continue

        # Take the longest shortest path as the storyline
        shortest_path = list(shortest_paths[list(shortest_paths)[-1]])
        storylines.append(shortest_path)
        temp_graph.remove_nodes_from(shortest_path)

    return storylines
