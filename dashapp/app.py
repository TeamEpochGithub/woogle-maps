"""Base application for the narrative maps visualization tool."""

from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, Input, Output, State, callback, callback_context, dash, html

app = DashProxy(__name__, use_pages=True, prevent_initial_callbacks=True, suppress_callback_exceptions=True, assets_folder=(Path() / "assets").absolute().as_posix())
app.title = "Woogle Maps"
app.layout = dmc.MantineProvider(
    theme={
        "primaryColor": "indigo",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        },
    },
    inherit=True,
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
        dmc.LoadingOverlay(
            style={"height": "100vh", "width": "100vw"},
            children=[
                dmc.Header(
                    height="10vh",
                    style={"backgroundColor": "#3d3c3c"},
                    top=0,
                    children=[
                        dmc.Center(
                            style={"height": "100%", "width": "100%"},
                            children=[
                                html.Img(
                                    src="https://images.squarespace-cdn.com/content/v1/5ef63c9ed5738e5562697e28/537e4287-ab40-45c1-8f8d-5fe3b61fc5c7/Epoch_Logo_Light.png",
                                    style={"width": "5%"},
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.Center(
                    style={"height": "90vh", "width": "100%"},
                    children=[
                        dash.page_container,
                    ],
                ),
                html.Div(id="popup", style={"display": "none"}),
            ],
        ),
    ],
)


@callback(
    Output("popup", "children"),
    Output("popup", "style"),
    Input("cytoscape", "tapNodeData"),
    State("popup", "style"),  # Adding this to get the current style of the popup
)
def toggle_pop_up(node_data: dict[str, Any] | None, current_style: dict[str, str | int]) -> tuple[Any, dict[str, Any]]:
    """Toggle pop-up showing cluster members.

    :param node_data: Dict with node data.
    :param current_style: Current state of the pop-up.
    :raises PreventUpdate: If the callback was not triggered by a user action.
    :return: Div element with pop-up.
    """
    ctx = callback_context
    if not ctx.triggered:
        # If callback was not triggered by a user action, do nothing
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # If the callback was triggered by clicking on the node and the popup is already visible, hide it
    if triggered_id == "cytoscape" and current_style and current_style.get("display") == "block":
        return "Click on a node to see its content", {"display": "none"}

    # Else, if the callback was triggered by clicking on the node and the popup is hidden, show it
    if node_data and "cluster_members" in node_data and isinstance(node_data["cluster_members"], list):
        documents_html = [html.Li(doc) for doc in node_data["cluster_members"]]
        summary_html = [html.Li(sentence) for sentence in node_data["summary"]]

        content = html.Div(
            [
                html.H4("Documents belonging to this node, sorted by date:", style={"color": "white", "fontWeight": "bold"}),
                html.Ul(documents_html),
                html.H4("Most important sentences in the documents:", style={"color": "white", "fontWeight": "bold"}),
                html.Ul(summary_html),
            ],
        )
        style = {
            "display": "block",
            "position": "fixed",
            "right": "5%",
            "top": "10%",
            "width": "450px",
            "height": "60%",
            "backgroundColor": "rgb(150, 150, 150)",
            "color": "white",
            "padding": "20px",
            "overflowY": "scroll",
            "zIndex": 1000,
            "-ms-overflow-style": "none",
            "scrollbar-width": "none",
            "border-radius": "15px",
        }
        return content, style
    return "Click on a node to see its content", {"display": "none"}


app.config.suppress_callback_exceptions = True

if __name__ == "__main__":
    app.run(port=8060)
