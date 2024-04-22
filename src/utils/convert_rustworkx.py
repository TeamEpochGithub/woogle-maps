"""Utility functions for converting rustworkx graphs to networkx and pygraphviz graphs."""

import networkx as nx
import rustworkx as rx


def to_networkx(graph: rx.PyGraph | rx.PyDiGraph) -> nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph:  # type: ignore[type-arg]
    """Convert a rustworkx PyGraph or PyDiGraph to a networkx graph.

    :param graph: The rustworkx graph to convert.
    :raises TypeError: If the graph is not a rustworkx PyGraph or PyDiGraph.
    :return: The networkx graph.
    """
    if not isinstance(graph, rx.PyGraph | rx.PyDiGraph):
        raise TypeError("graph must be of type rustworkx.PyGraph or rustworkx.PyDiGraph.")

    edge_list = [(graph[x[0]], graph[x[1]], {"weight": x[2]}) for x in graph.weighted_edge_list()]

    if isinstance(graph, rx.PyGraph):
        if graph.multigraph:
            return nx.MultiGraph(edge_list)
        return nx.Graph(edge_list)
    if graph.multigraph:
        return nx.MultiDiGraph(edge_list)
    return nx.DiGraph(edge_list)
