"""Utility for fixing dtypes in a pandas DataFrame after reading from a CSV file."""

from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd

ALL_DTYPES: Final[dict[str, type]] = {
    # Bodytext
    "foi_documentId": str,
    "foi_pageNumber": int,
    "foi_bodyText": str,
    "foi_bodyTextOCR": str,
    "foi_hasOCR": bool,
    "foi_redacted": float,
    "foi_contourArea": float,
    "foi_textArea": float,
    "foi_charArea": float,
    "foi_percentageTextAreaRedacted": float,
    "foi_percentageCharAreaRedacted": float,
    # Dossier
    "dc_identifier": str,
    "dc_title": str,
    "dc_description": str,
    "dc_type": str,
    "dc_publisher_name": str,
    "dc_publisher": str,
    "dc_source": str,
    "foi_valuation": str,
    "foi_requestText": str,
    "foi_decisionText": str,
    "foi_isAdjourned": str,
    "foi_requester": str,
    # Document
    "foi_dossierId": str,
    "foi_fileName": str,
    "dc_format": str,
    "foi_nrPages": float,
    "foi_pdfDateCreated": str,
    "foi_pdfDateModified": str,
    "foi_pdfCreator": str,
    "foi_pdfProducer": str,
    "foi_pdfAuthor": str,
    "foi_pdfCompany": str,
    "foi_pdfTitle": str,
    "foi_pdfSubject": str,
    "foi_pdfKeywords": str,
    "foi_fairiscore": str,
    # Norambuena
    "id": int,
    "title": str,
    "url": str,
    # "date": datetime,
    "publication": str,
    "full_text": str,
    # "embed": list[float],
    # Extract Important Sentences
    # "filtered_text": list[str],
    # "summary": list[str],
    # Topical Distribution
    # "topical_dist": list[float],
    # Cluster Documents
    # "memberships": list[int],
    # Linear Programming
    "discarded": bool,
    # "adj_list": list[int],
    # "adj_weights": list[float],
    # Create Events
    "clusters": int,
    # Find Storylines
    "storyline": int,
    # Compute Layout
    "x": float,
    "y": float,
}


def parse_float_list(arr: str) -> list[float]:
    """Parse a string of floats into a list of floats.

    Parse a string of floats into a list of floats. The string is expected to be formatted like "[1.0 2.0 3.0]".

    :param arr: the string of floats.
    :return: the list of floats.
    """
    return [float(i.replace(",", "")) for i in arr[1:-1].strip().split(" ") if i]


def parse_int_list(arr: str) -> list[int]:
    """Parse a string of ints into a list of ints.

    Parse a string of ints into a list of ints. The string is expected to be formatted like "[1 2 3]".

    :param arr: the string of ints.
    :return: the list of ints.
    """
    return [int(i.replace(",", "")) for i in arr[1:-1].strip().split(" ") if i]


def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """Fix the data types of the columns in a pandas DataFrame.

    Fix the data types of the columns in the DataFrame. The function will convert the columns to the correct data types.
    This is useful when reading data from a CSV file where the data types are not preserved.

    :param data: the DataFrame to fix the data types.
    :return: the DataFrame with the correct data types.
    """
    data = data.astype({k: v for k, v in ALL_DTYPES.items() if k in data.columns})

    if "date" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["date"]):
        data["date"] = pd.to_datetime(data["date"]).apply(lambda x: x.replace(tzinfo=ZoneInfo("Europe/Amsterdam")) if pd.notna(x) else x)

    if "embed" in data.columns and not pd.api.types.is_list_like(data["embed"][0]):
        data["embed"] = data["embed"].map(parse_float_list)

    if "topical_dist" in data.columns and not pd.api.types.is_list_like(data["topical_dist"][0]):
        data["topical_dist"] = data["topical_dist"].map(parse_float_list)

    if "memberships" in data.columns and not pd.api.types.is_list_like(data["memberships"][0]):
        data["memberships"] = data["memberships"].map(parse_float_list)

    if "adj_list" in data.columns and not pd.api.types.is_list_like(data["adj_list"][0]):
        data["adj_list"] = data["adj_list"].map(parse_int_list)

    if "adj_weights" in data.columns and not pd.api.types.is_list_like(data["adj_weights"][0]):
        data["adj_weights"] = data["adj_weights"].map(parse_float_list)

    return data
