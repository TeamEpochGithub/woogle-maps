"""Home page layout."""

import base64
import shutil
from zipfile import ZipFile

import dash_mantine_components as dmc
import pandas as pd
import randomname
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State, callback, dash, dcc, html
from dash_iconify import DashIconify

from dashapp.conf import RAW_DATA_PATH, UPLOAD_DATA_PATH
from src.load_data import load_pdf_dossier

dash.register_page(__name__, path="/", title="Woogle Maps", name="Woogle Maps")
layout = (
    dmc.Stack(
        [
            dmc.Image(
                src=dash.get_asset_url("Woogle_Maps_Logo_Dark.svg"),
            ),
            dmc.Select(
                id="dataset-choice",
                data=[dossier.stem for dossier in RAW_DATA_PATH.glob("*.pkl") if dossier.is_file()],
                placeholder="Select a dossier",
                searchable=True,
                clearable=True,
                icon=DashIconify(icon="mingcute:documents-line"),
            ),
            dmc.Divider(label="or", labelPosition="center"),
            dmc.Card(
                dcc.Upload(
                    dmc.Center(
                        html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    ),
                    id="upload-data",
                    multiple=True,
                    disabled=False,
                ),
                style={
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                },
            ),
            html.Div(id="output-data-upload"),
            dcc.Location(id="map-url", refresh=True),
            dmc.NavLink(
                label="Generate Narrative Map",
                icon=DashIconify(icon="gis:map-route"),
                rightSection=DashIconify(icon="tabler-chevron-right"),
                id="generate-map",
                href="/",
                active=False,
                color="black",
                variant="filled",
            ),
        ],
        style={"width": "30vw"},
        align="stretch",
        justify="center",
    ),
)


@callback(
    Output("output-data-upload", "children", allow_duplicate=True),
    Output("generate-map", "active", allow_duplicate=True),
    Output("generate-map", "href", allow_duplicate=True),
    Input("dataset-choice", "value"),
    prevent_initial_call=True,
)
def dossier_selected(dossier_id: str | None) -> tuple[html.Div, bool, str]:
    """Update the URL when a dossier is selected.

    :param dossier_id: the id of the selected dossier.
    :raises PreventUpdate: If no dossier was selected.
    :return: The URL to navigate to.
    """
    if dossier_id is None:
        raise PreventUpdate
    return html.Div(), bool(dossier_id), f"/map/{dossier_id}"


@callback(
    Output("output-data-upload", "children", allow_duplicate=True),
    Output("dataset-choice", "value", allow_duplicate=True),
    Output("generate-map", "active", allow_duplicate=True),
    Output("generate-map", "href", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def upload_files(contents: list[str], filenames: list[str]) -> tuple[html.Div, None, bool, str]:
    """Save PDF files or Extract a zip of PDFs.

    :param contents: The contents of the uploaded files.
    :param filenames: The names of the uploaded files.
    :return: The updates to the page.
    """
    if not contents or not filenames:
        raise PreventUpdate

    # Create a directory to extract files
    upload_dir = UPLOAD_DATA_PATH / randomname.get_name()
    upload_dir.mkdir(parents=True, exist_ok=True)

    extracted_files_names: list[str] = []

    # Loop through each uploaded file or folder
    for content, filename in zip(contents, filenames, strict=False):
        if filename.endswith(".zip"):
            # Split the content from its type
            _, content_string = content.split(",")
            decoded = base64.b64decode(content_string)

            temp_zip_path = upload_dir / "temp" / filename
            (upload_dir / "temp").mkdir(parents=True, exist_ok=True)
            with open(temp_zip_path, "wb") as f:
                f.write(decoded)

            with ZipFile(temp_zip_path) as zip_file:
                pdfs = [f for f in zip_file.namelist() if f.endswith(".pdf")]
                zip_file.extractall(path=upload_dir, members=pdfs)
                extracted_files_names.extend(pdfs)

            shutil.rmtree(upload_dir / "temp")
        else:
            # accept a list of files as well
            _, content_string = content.split(",")
            decoded = base64.b64decode(content_string)
            extracted_files_names.append(filename)

            # Save the file or folder content to temporary directory
            with open(upload_dir / filename, "wb") as f:
                f.write(decoded)

    if not extracted_files_names:
        div = html.Div(
            ["No PDF files found within the uploaded files or zip file."],
        )
    else:
        files_list = html.Ul([html.Li(file) for file in extracted_files_names], style={"max-height": "200px", "overflow-y": "auto"})
        div = html.Div(
            [
                html.Div("PDF Files Extracted:", style={"font-size": "20px"}),
                files_list,
            ],
        )

    # Create pickle file
    data = load_pdf_dossier(upload_dir)
    csv_files = [pd.read_csv(file, parse_dates=["date"]) for file in list(upload_dir.glob("*.csv"))]
    if csv_files:
        csv_files.append(data)
        data = pd.concat(csv_files, ignore_index=True)

    if data.empty:
        return (
            html.Div(
                [
                    "No valid documents found in the uploaded files or zip file. Only PDF and CSV document files are accepted.",
                ],
            ),
            None,
            False,
            "/",
        )
    if len(data.index) < 5:
        return (
            html.Div(
                [
                    "Please upload at least 5 documents.",
                ],
            ),
            None,
            False,
            "/",
        )

    data.to_pickle(RAW_DATA_PATH / f"{upload_dir.name}.pkl")
    return div, None, bool(extracted_files_names), f"/map/{upload_dir.name}"
