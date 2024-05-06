"""Filter redundant edges from the data."""

from dataclasses import dataclass
from math import exp, log
from typing import Any

import pandas as pd
import polars as pl
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock

from src.logging.logger import Logger
from src.utils.construct_graph import build_graph_from_adj_list, filter_interstory_connections, transitive_reduction


@dataclass
class FilterRedundantEdges(TransformationBlock, Logger):
    """Filter redundant edges from the data.

    The data is expected to have an "adj_list" and "adj_weights" column that contains the adjacency list of the graph.
    It will return the data with the redundant edges filtered out.
    """

    def custom_transform(self, data: pd.DataFrame, **transform_args: Any) -> pd.DataFrame:  # noqa: ARG002, ANN401, DOC103  # type: ignore[misc]
        """Filter out redundant edges from the data.

        :param data: The data to filter the redundant edges from.
        :param transform_args: Additional keyword arguments (UNUSED).
        :return: The data with the redundant edges filtered out.
        """
        if "adj_list" not in data.columns or "adj_weights" not in data.columns:
            self.log_to_warning("The input data does not have an 'adj_list' and 'adj_weights' column. Filtering redundant edges not necessary.")
            return data

        pl_data = pl.from_pandas(data)

        full_data = None
        if "clusters" in pl_data.columns:
            full_data = pl_data.clone().drop(["adj_list", "adj_weights", "storyline"])
            pl_data = (
                pl_data.group_by("clusters")
                .agg(
                    [
                        pl.col("adj_list").first().alias("adj_list"),
                        pl.col("adj_weights").first().alias("adj_weights"),
                        pl.col("storyline").first().alias("storyline"),
                    ],
                )
                .sort("clusters")
            )

        storylines = pl_data.select("storyline").with_row_index().group_by("storyline").agg(pl.col("index")).sort("storyline").get_column("index").to_list()
        graph = build_graph_from_adj_list(pl_data.get_column("adj_list").to_list(), pl_data.get_column("adj_weights").to_list())

        # Apply Transitive Reduction
        graph = transitive_reduction(graph, storylines)

        # Filter Interstory Connections
        graph = filter_interstory_connections(graph, storylines)

        # Update the graph to reflect the changes
        new_adj_list: list[list[int]] = [[0]] * pl_data.height
        new_adj_weights: list[list[float]] = [[0]] * pl_data.height

        for i in range(pl_data.height):
            edges: list[tuple[int, int, float]] = list(graph.out_edges(i))
            if not edges:
                continue

            sum_weights = sum([exp(-weight) for (_, _, weight) in edges])
            for u, v, weight in edges:
                graph.update_edge(u, v, -log(exp(-weight) / sum_weights))

            new_edges: list[tuple[int, int, float]] = list(graph.out_edges(i))
            new_adj_list[i] = [v for (_, v, _) in new_edges]
            new_adj_weights[i] = [exp(-weight) for (_, _, weight) in new_edges]

        # Update the data
        pl_data = pl_data.with_columns(
            [
                pl.Series("adj_list", new_adj_list),
                pl.Series("adj_weights", new_adj_weights),
            ],
        ).drop("index")

        if full_data is not None:
            pl_data = pl_data.join(full_data, on="clusters")

        return pl_data.to_pandas().drop("index", errors="ignore")
