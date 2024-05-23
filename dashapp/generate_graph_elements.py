"""Helper functions for the dashapp app."""

import logging
from typing import Literal, TypedDict

import dash_cytoscape as cyto
import pandas as pd
from dash_extensions.enrich import html
from scipy.stats import qmc

STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "content": "data(label)",
            "text-valign": "center",
            "color": "#000",  # Black font color for the text
            "background-color": "rgb(200, 200, 200)",  # Light grey color for all nodes
            "shape": "ellipse",
            "width": "40px",
            "height": "40px",
            "font-size": "12px",
        },
    },
    {  # Override style for nodes with class 'story'
        "selector": ".story",
        "style": {
            "background-color": "rgb(246, 246, 246)",  # Lighter grey color for story nodes
        },
    },
    {  # Default style for all edges
        "selector": "edge",
        "style": {
            "label": "data(label)",
            "font-size": "10px",
            "color": "#000",  # Black font color for the text
            "text-background-opacity": 1,  # Full opacity for the text background
            "text-background-color": "#ffffff",  # White background to block the edge
            "text-background-padding": "3px",  # Padding around the text
            "text-background-shape": "roundrectangle",  # Shape of the background
            "text-border-color": "#000",  # Optional: border color for the text background
            "text-border-width": 1,  # Optional: border width for the text background
            "text-border-opacity": 1,  # Optional: border opacity for the text background
            "target-arrow-shape": "triangle",
            "curve-style": "straight",
        },
    },
    {  # Specific style for edges with class 'sp'
        "selector": "edge.sp",
        "style": {
            "line-style": "dashed",  # Dashed line style
            "line-color": "rgb(35, 28, 173)",  # Blue color for the storyline
            "target-arrow-color": "rgb(35, 28, 173)",  # Same color for the arrow
            "width": 2,
        },
    },
    {  # Style for labels on nodes and edges
        "selector": "node, edge",
        "style": {
            "label": "data(label)",
            "text-wrap": "wrap",
            "font-family": '"Plus Jakarta Sans", sans-serif',
            "text-max-width": "80px",
            "text-valign": "top",
            "text-halign": "center",
            "text-margin-x": "-5px",
            "font-size": "10px",
            "color": "#000",  # Black font color for the text
            "text-outline-width": 0,  # No outline for better readability on light backgrounds
        },
    },
]

logger = logging.getLogger(__name__)


class StoryElementData(TypedDict):
    """The data for a story element.

    :param id: The story string ID.
    :param number: The number of the story.
    :param story: Whether the element is a story.
    """

    id: str
    number: int
    story: Literal[True]


class StoryElement(TypedDict):
    """A story element for the graph.

    :param data: The is, number, and story number for the element.
    :param classes: The classes for the element.
    :param selectable: Whether the element is selectable.
    """

    data: StoryElementData
    classes: str
    selectable: Literal[False]


class ClusterNodeElementData(TypedDict, total=False):
    """The data for a cluster node element.

    :param id: The cluster node ID.
    :param parent: The parent node ID.
    :param label: The label for the cluster node.
    :param cluster_members: The cluster members.
    """

    id: str
    parent: str | None
    label: str
    cluster_members: list[str]
    summary: list[str]


class Point(TypedDict):
    """A point for the graph.

    :param x: The x position for the point.
    :param y: The y position for the point.
    """

    x: float
    y: float


class NodeElement(TypedDict, total=False):
    """A cluster node element for the graph.

    :param data: The id, parent, label, and cluster members for the element.
    :param classes: The classes for the element.
    :param position: The x and y position for the element.
    """

    data: ClusterNodeElementData
    classes: str
    position: Point


class EdgeElementData(TypedDict, total=False):
    """The data for an edge element.

    :param id: The edge ID.
    :param source: The source node ID.
    :param target: The target node ID.
    :param label: The label for the edge.
    :param weight: The weight for the edge.
    :param width: The width for the edge.
    """

    id: str
    source: str
    target: str
    label: str
    weight: float
    width: float
    directed: bool


class EdgeElement(TypedDict):
    """An edge element for the graph.

    :param data: The id, source, target, label, weight, and width for the element.
    :param classes: The classes for the element.
    """

    data: EdgeElementData
    classes: str


def create_story_element(story: int) -> StoryElement:
    """Create a story element for the graph.

    :param story: The story number.
    :return: The story element.
    """
    return {
        "data": {
            "id": f"story_{story}_1I",
            "number": story,
            "story": True,
        },
        "classes": "story",
        "selectable": False,
    }


def generate_node_elements(data: pd.DataFrame) -> list[NodeElement]:
    """Generate elements for each cluster, represented by a node.

    :param data : The dataframe containing the graph data.
    :return: The elements corresponding to the different clusters of documents.
    """
    node_elements: list[NodeElement] = [{}] * (data["clusters"].max() + 1)

    if "x" not in data.columns or "y" not in data.columns:
        logger.warning("The data does not contain valid 'x' and 'y' columns. Scattering nodes instead.")
        coords = qmc.Halton(d=2).random(data.shape[0]) * 1000

        data["x"] = coords[:, 0]
        data["y"] = coords[:, 1]

    for i in range(len(node_elements)):
        cluster_data = data.query(f"clusters == {i}")
        if "storyline" in cluster_data.columns:
            parent_id = f"story_{cluster_data.iloc[0]["storyline"]}_1I"
        else:
            parent_id = None

        dates: list[str] = ["n.d." for _ in cluster_data.index]
        if "date" in cluster_data.columns:
            cluster_data = cluster_data.sort_values(by="date")
            if pd.api.types.is_datetime64_any_dtype(cluster_data["date"]):
                dates = [date.strftime("%Y-%m-%d") if not pd.isna(date) else "n.d." for date in cluster_data["date"]]  # Ensure dates are strings

        if "title" in cluster_data.columns:
            titles = cluster_data["title"].tolist()
        else:
            titles = ["No Title" for _ in cluster_data.index]

        if "abstract" in cluster_data.columns:
            summary = cluster_data.iloc[0]["abstract"].tolist()
        else:
            summary = ["No Summary" for _ in cluster_data.index]

        cluster_members = [f"{date} - {title}" for date, title in zip(dates, titles, strict=False)]

        node_data: NodeElement = {
            "data": {"id": f"{i}_1I", "parent": parent_id, "label": titles[0] if cluster_members else "No title", "cluster_members": cluster_members, "summary": summary},
            "classes": "cluster_data ac",
            "position": {
                "x": cluster_data.iloc[0]["x"],
                "y": cluster_data.iloc[0]["y"],
            },
        }
        node_elements[i] = node_data

    return node_elements


def generate_edges(data: pd.DataFrame) -> list[EdgeElement]:
    """Generate elements for each edge between clusters.

    :param data : The dataframe containing the graph data.
    :return: The elements corresponding to the different clusters of documents.
    """
    edges: list[EdgeElement] = []

    for i in data["clusters"].unique():
        adj_list: list[int] = data.query(f"clusters == {i}").iloc[0]["adj_list"]
        adj_weights: list[float] = data.query(f"clusters == {i}").iloc[0]["adj_weights"]

        source_id = f"{i}_1I"
        story_i = data.query(f"clusters == {i}").iloc[0]["storyline"]

        for j in range(len(adj_list)):
            if adj_weights[j] <= 0:
                continue
            target_id = f"{adj_list[j]}_1I"
            story_j = data.query(f"clusters == {adj_list[j]}").iloc[0]["storyline"]
            edge_data: EdgeElement = {
                "data": {
                    "id": f"{source_id}-{target_id}",
                    "source": source_id,
                    "target": target_id,
                    "weight": adj_weights[j],
                    "width": adj_weights[j] * 5,
                    "directed": True,
                },
                "classes": "sp" if story_i == 0 and story_j == 0 else "",
            }
            edges.append(edge_data)

    return edges


def generate_graph_elements(data: pd.DataFrame) -> list[StoryElement | NodeElement | EdgeElement]:
    """Generate elements for each edge between clusters.

    :param data : The dataframe containing the graph data.
    :return: The elements for the graph.
    """
    if "storyline" in data.columns:
        storyline_elements = [create_story_element(storyline) for storyline in range(data["storyline"].max() + 1)]
    else:
        storyline_elements = []

    if "clusters" not in data.columns:
        data["clusters"] = data.index
    node_elements = generate_node_elements(data)

    if "adj_list" in data.columns and "adj_weights" in data.columns:
        edge_elements = generate_edges(data)
    else:
        edge_elements = []

    return storyline_elements + node_elements + edge_elements


def create_graph(data: pd.DataFrame) -> cyto.Cytoscape:
    """Create a Cytoscape graph from the given data.

    :param data: The data to create the graph from.
    :return: The Cytoscape graph.
    """
    elements = generate_graph_elements(data)
    pop_up_div = html.Div(id="pop-up-content", style={"display": None})  # noqa: F841; Cannot be removed.

    return cyto.Cytoscape(
        id="cytoscape",
        mouseoverNodeData={},
        zoom=1,
        layout={"name": "preset"},
        style={"height": "92vh", "width": "100vw"},
        stylesheet=STYLESHEET,
        elements=elements,
    )
