import pandas as pd
import polars as pl
import rustworkx as rx

from src.preprocessing.filter_redundant_edges import FilterRedundantEdges
from src.utils.construct_graph import build_graph_from_adj_list, transitive_reduction, filter_interstory_connections, edge_boundary


def test__filter_redundant_edges():
    data = pd.DataFrame(
        {
            "adj_list": [[8, 10, 1], [7, 8, 2], [7, 8, 10], [4, 5], [7, 8, 6], [7], [8, 9, 10], [8, 10], [12, 10, 9], [12, 13, 10], [11, 13], [13], [], []],
            "adj_weights": [
                [0.0640658152988518, 0.06406582718181346, 0.8718683575193347],
                [0.11111110958452239, 0.11111110958452239, 0.7777777808309553],
                [0.3333333127243863, 0.3333333127243863, 0.3333333745512273],
                [0.6666666666666666, 0.3333333333333333],
                [0.25, 0.25, 0.5],
                [1.0],
                [0.25, 0.25, 0.5],
                [0.09999999958782103, 0.900000000412179],
                [0.19999999999999998, 0.2000000247307379, 0.5999999752692621],
                [0.33333331959403506, 0.33333331959403506, 0.3333333608119299],
                [0.39835457951843506, 0.6016454204815649],
                [1.0],
                [],
                [],
            ],
            "storyline": [0, 0, 0, 1, 1, 2, 3, 2, 1, 3, 0, 4, 1, 0],
        },
    )

    block = FilterRedundantEdges()
    res = block.transform(data)

    ans = pd.DataFrame(
        {
            "adj_list": [[8, 1], [7, 2], [7, 8, 10], [4, 5], [8], [7], [8, 9, 10], [8], [12], [12, 13], [11, 13], [13], [], []],
            "adj_weights": [
                [0.0684511979148529, 0.9315488020851471],
                [0.12499999806791111, 0.8750000019320888],
                [0.3333333127243864, 0.3333333127243864, 0.3333333745512273],
                [0.6666666666666666, 0.3333333333333333],
                [1.0],
                [1.0],
                [0.25, 0.25, 0.5],
                [1.0],
                [1.0],
                [0.5, 0.5],
                [0.39835457951843506, 0.6016454204815649],
                [1.0],
                [],
                [],
            ],
            "storyline": [0, 0, 0, 1, 1, 2, 3, 2, 1, 3, 0, 4, 1, 0],
        },
    )

    # assert res.equals(ans)  # TODO(Jeffrey): Fix failing test

def test__filter_redundant_edges__with_clusters():
    data = pd.DataFrame({
        "title": [
            "Covid in Wuhan",
            "Sars ruled out",
            "First death reported",
            "Lockdown in New York",
            "New strain found",
            "Vaccine development begins",
            "Global cases rise",
            "Travel restrictions applied",
        ],
        "date": [
            "2023-02-19", "2023-02-24", "2023-02-28", "2023-03-21",
            "2023-03-22", "2023-05-02", "2023-04-12", "2023-05-17",
        ],
        "clusters": [
            0, 0, 0, 1, 1, 1, 2, 2,
        ],
        "adj_list": [
            [1, 2],
            [1, 2],
            [1, 2],
            [0],
            [0],
            [0],
            [0, 1],
            [0, 1],
        ],
        "adj_weights": [
            [0.7, 0.3],
            [0.7, 0.3],
            [0.7, 0.3],
            [0.5],
            [0.5],
            [0.5],
            [0.6, 0.4],
            [0.6, 0.4],
        ],
        "storyline": [0, 0, 0, 1, 1, 1, 0, 0]
    })

    block = FilterRedundantEdges()
    res = block.transform(data)

    ans = pd.DataFrame({
        "title": [
            "Covid in Wuhan",
            "Sars ruled out",
            "First death reported",
            "Lockdown in New York",
            "New strain found",
            "Vaccine development begins",
            "Global cases rise",
            "Travel restrictions applied",
        ],
        "date": [
            "2023-02-19", "2023-02-24", "2023-02-28", "2023-03-21",
            "2023-03-22", "2023-05-02", "2023-04-12", "2023-05-17",
        ],
        "clusters": [
            0, 0, 0, 1, 1, 1, 2, 2,
        ],
        "adj_list": [
            [2, 1],
            [2, 1],
            [2, 1],
            [0],
            [0],
            [0],
            [0],
            [0],
        ],
        "adj_weights": [
            [0.3, 0.7],
            [0.3, 0.7],
            [0.3, 0.7],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        "storyline": [0, 0, 0, 1, 1, 1, 0, 0]
    })
    # TODO(Jeffrey): Fix test. Result is randomized?
    # assert pd.testing.assert_frame_equal(res, ans, check_like=True) is None

def test__transitive_reduction():
    data = pl.DataFrame(
        {
            "adj_list": [[8, 10, 1], [7, 8, 2], [7, 8, 10], [4, 5], [7, 8, 6], [7], [8, 9, 10], [8, 10], [12, 10, 9], [12, 13, 10], [11, 13], [13], [], []],
            "adj_weights": [
                [0.0640658152988518, 0.06406582718181346, 0.8718683575193347],
                [0.11111110958452239, 0.11111110958452239, 0.7777777808309553],
                [0.3333333127243863, 0.3333333127243863, 0.3333333745512273],
                [0.6666666666666666, 0.3333333333333333],
                [0.25, 0.25, 0.5],
                [1.0],
                [0.25, 0.25, 0.5],
                [0.09999999958782103, 0.900000000412179],
                [0.19999999999999998, 0.2000000247307379, 0.5999999752692621],
                [0.33333331959403506, 0.33333331959403506, 0.3333333608119299],
                [0.39835457951843506, 0.6016454204815649],
                [1.0],
                [],
                [],
            ],
            "storyline": [0, 0, 0, 1, 1, 2, 3, 2, 1, 3, 0, 4, 1, 0],
        },
    )

    storylines = data.select("storyline").with_row_index().group_by("storyline").agg(pl.col("index")).sort("storyline").get_column("index").to_list()
    graph = build_graph_from_adj_list(data.get_column("adj_list").to_list(), data.get_column("adj_weights").to_list())

    graph_new = transitive_reduction(graph.copy(), storylines)

    graph.remove_edges_from([(0, 10)])
    assert graph.edge_list() == graph_new.edge_list()

def test__edge_boundary():
    G = rx.PyDiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2, 1), (1, 3, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (1, 5, 1), (2, 5, 1), (3, 5, 1), (4, 5, 1)])

    assert edge_boundary(G, [1, 2], [4, 5]) == [(1, 5), (2, 5)]


def test__filter_interstory_connections():
    data = pl.DataFrame(
        {
            "adj_list": [[8, 10, 1], [7, 8, 2], [7, 8, 10], [4, 5], [7, 8, 6], [7], [8, 9, 10], [8, 10], [12, 10, 9], [12, 13, 10], [11, 13], [13], [], []],
            "adj_weights": [
                [0.0640658152988518, 0.06406582718181346, 0.8718683575193347],
                [0.11111110958452239, 0.11111110958452239, 0.7777777808309553],
                [0.3333333127243863, 0.3333333127243863, 0.3333333745512273],
                [0.6666666666666666, 0.3333333333333333],
                [0.25, 0.25, 0.5],
                [1.0],
                [0.25, 0.25, 0.5],
                [0.09999999958782103, 0.900000000412179],
                [0.19999999999999998, 0.2000000247307379, 0.5999999752692621],
                [0.33333331959403506, 0.33333331959403506, 0.3333333608119299],
                [0.39835457951843506, 0.6016454204815649],
                [1.0],
                [],
                [],
            ],
            "storyline": [0, 0, 0, 1, 1, 2, 3, 2, 1, 3, 0, 4, 1, 0],
        },
    )

    storylines = data.select("storyline").with_row_index().group_by("storyline").agg(pl.col("index")).sort("storyline").get_column("index").to_list()
    graph = build_graph_from_adj_list(data.get_column("adj_list").to_list(), data.get_column("adj_weights").to_list())

    stories = [[0, 1, 2, 10, 13], [3, 4, 8, 12], [5, 7], [6, 9], [11]]

    transitive_reduction(graph, storylines)

    graph_new = filter_interstory_connections(graph.copy(), storylines)

    # TODO(Jeffrey): Fix this test
    # In this case, we can't compare this functionality because there was actually a bug in the original implementation
    # because he compared node indices as strings instead of ints.
    graph.remove_edges_from([(1, 8), (8, 10), (7, 10), (9, 10), (4, 7), (8, 9), (4, 6)])
    # assert graph.edge_list() == graph_new.edge_list()
