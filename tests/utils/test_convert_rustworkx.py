import math

import networkx as nx
import pytest
import rustworkx as rx

from src.utils import convert_rustworkx


def create_rustworkx_graph(multigraph=False, directed=False):
    if directed:
        graph = rx.PyDiGraph(multigraph=multigraph)
    else:
        graph = rx.PyGraph(multigraph=multigraph)
    node_a = graph.add_node("A")
    node_b = graph.add_node("B")
    graph.add_edge(node_a, node_b, 1.0)
    return graph


@pytest.mark.parametrize("multigraph,directed,expected_type", [
    (False, False, nx.Graph),
    (False, True, nx.DiGraph),
    (True, False, nx.MultiGraph),
    (True, True, nx.MultiDiGraph),
])
def test_convert_rustworkx_to_networkx(multigraph, directed, expected_type):
    rx_graph = create_rustworkx_graph(multigraph=multigraph, directed=directed)
    nx_graph = convert_rustworkx.to_networkx(rx_graph)
    assert isinstance(nx_graph, expected_type)
    assert len(nx_graph.edges) == 1
    edge = list(nx_graph.edges(data=True))[0]
    assert math.isclose(edge[2]['weight'], 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_convert_rustworkx_to_networkx_invalid_type():
    with pytest.raises(TypeError):
        convert_rustworkx.to_networkx("not a graph")
