"""Construct a graph from a DataFrame and find storylines in it."""

import itertools
from collections.abc import Iterable, Sequence
from math import log

import rustworkx as rx


def build_graph_from_adj_list(adj_list: Sequence[Sequence[int]], adj_weights: Sequence[Sequence[float]]) -> rx.PyDiGraph:  # type: ignore[type-arg]
    """Build a graph from an adjacency list.

    :param adj_list: The adjacency list.
    :param adj_weights: The weights of the edges.
    :return: The graph as a directed graph.
    """
    graph = rx.PyDiGraph()  # type: ignore[var-annotated]
    graph.add_nodes_from(range(len(adj_list)))
    for i, (nodes, weights) in enumerate(zip(adj_list, adj_weights, strict=False)):
        for j in range(len(nodes)):
            graph.add_edge(i, nodes[j], max(-log(weights[j]), 0))
    return graph


def transitive_reduction(graph: rx.PyDiGraph, storylines: list[list[int]]) -> rx.PyDiGraph:  # type: ignore[type-arg]
    """Perform transitive reduction on the graph within storylines.

    :param graph: The graph to perform transitive reduction on.
    :param storylines: The story lines.
    :return: The graph after transitive reduction.
    """
    remove_list_transitive = [
        (story[i], story[j]) for story in storylines for i in range(len(story) - 1) for j in range(i + 2, len(story)) if graph.has_edge(story[i], story[j])
    ]
    graph.remove_edges_from(remove_list_transitive)
    return graph


def edge_boundary(graph: rx.PyDiGraph, nbunch1: Iterable[int], nbunch2: Iterable[int] | None = None) -> list[tuple[int, int]]:  # type: ignore[type-arg]
    """Return the edge boundary of two sets of nodes ina  rustworkx graph. Should be functionally equivalent to networkx.edge_boundary.

    :param graph: The graph to get the edge boundary from.
    :param nbunch1: The first set of nodes.
    :param nbunch2: The second set of nodes.
    :return: The edge boundary of the two sets of nodes.
    """
    nset1 = {n for n in nbunch1 if n in graph.nodes()}

    outgoing_edges = []
    for node in nset1:
        outgoing_edges.extend([(node, target) for _, target, _ in graph.out_edges(node)])

    if nbunch2 is None:
        return [e for e in outgoing_edges if (e[0] in nset1) ^ (e[1] in nset1)]

    nset2 = set(nbunch2)
    return [e for e in outgoing_edges if (e[0] in nset1 and e[1] in nset2) or (e[1] in nset1 and e[0] in nset2)]


def filter_interstory_connections(graph: rx.PyDiGraph, storylines: list[list[int]]) -> rx.PyDiGraph:  # type: ignore[type-arg]  # noqa: C901, PLR0912
    """Filter interstory connections from the graph.

    This function filters out interstory connections from the graph.

    :param graph: The graph to filter interstory connections from.
    :param storylines: The story lines.
    :return: The graph after filtering interstory connections.
    """
    remove_list_interstory = []
    for story_i, story_j in list(itertools.combinations(storylines, 2)):
        keep_set = set()
        edge_boundary_story_i = set(edge_boundary(graph, story_i, story_j))
        edge_boundary_story_j = set(edge_boundary(graph, story_j, story_i))

        earliest_i = None
        latest_i = None
        earliest_j = None
        latest_j = None

        if edge_boundary_story_i:
            earliest_i = min(edge_boundary_story_i)
            latest_i = max(edge_boundary_story_i)

        if edge_boundary_story_j:
            earliest_j = min(edge_boundary_story_j, key=lambda t: (t[1], t[0]))
            latest_j = max(edge_boundary_story_j, key=lambda t: (t[1], t[0]))

        if not edge_boundary_story_i and not edge_boundary_story_j:
            # No connections
            continue
        if edge_boundary_story_i and not edge_boundary_story_j:
            # One-sided.
            if earliest_i == latest_i:  # There's only one connection, keep it.
                continue
            if earliest_i[0] == latest_i[0]:  # type: ignore[index]
                # This means that the second values are different, but the first one is the same.
                # For this special case we keep the earliest one.
                keep_set.add(earliest_i)
            else:
                # Keep both
                keep_set.add(earliest_i)
                keep_set.add(latest_i)
        elif not edge_boundary_story_i and edge_boundary_story_j:
            # One-sided and reversed.
            if earliest_j == latest_j:  # There's only one connection, keep it.
                continue
            if earliest_j[1] == latest_j[1]:  # type: ignore[index]
                # This means that the first values are different, but the second one is the same.
                # For this special case we keep the earliest one.
                keep_set.add(earliest_j)
            else:
                # Keep both
                keep_set.add(earliest_j)
                keep_set.add(latest_j)
        elif earliest_i[0] < earliest_j[1]:  # type: ignore[index]
            # E_i happened first. Keep it
            keep_set.add(earliest_i)
            if latest_i[0] > latest_j[1]:  # type: ignore[index]
                # L_i happened last. Keep it.
                keep_set.add(latest_i)
            else:
                keep_set.add(latest_j)
        else:
            # E_j happened first.
            keep_set.add(earliest_j)
            if latest_i[0] > latest_j[1]:  # type: ignore[index]
                # E_i happened last. Keep it.
                keep_set.add(latest_i)
            else:
                keep_set.add(latest_j)
        redundant_edge_set = set(edge_boundary_story_i) | set(edge_boundary_story_j)
        remove_list_interstory += list(redundant_edge_set - keep_set)

    graph.remove_edges_from(remove_list_interstory)
    return graph


def normalize_adj_weights(graph: rx.PyDiGraph) -> rx.PyDiGraph:  # type: ignore[type-arg]
    """Normalize the weights of the edges in the graph.

    :param graph: The graph to normalize the weights of.
    :return: The graph with normalized weights.
    """
    for node in graph.nodes():
        total_weight = sum([graph.get_edge_data(node, neighbor)["weight"] for neighbor in graph.neighbors(node)])
        for neighbor in graph.neighbors(node):
            graph.get_edge_data(node, neighbor)["weight"] /= total_weight
    return graph
