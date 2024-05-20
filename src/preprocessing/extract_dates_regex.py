"""Extract the creation dates from the full text of documents."""

import datetime
import re
from dataclasses import InitVar, dataclass
from typing import Final, Never
from zoneinfo import ZoneInfo

import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from pandas._libs import NaTType
from pandas.core.dtypes.common import is_string_dtype
from pytz import timezone

from src.logging.logger import Logger

DATE_REGEX: Final[re.Pattern[str]] = re.compile(
    r"(\d{1,2}-\d{1,2}-\d{4})|(\d{1,2} (januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december) \d{4})",
    re.IGNORECASE,
)

MONTH_MAP: Final[dict[str, int]] = {
    # Dutch
    "januari": 1,
    "februari": 2,
    "maart": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "augustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
    # English
    "january": 1,
    "february": 2,
    "march": 3,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "october": 10,
}


@dataclass
class ExtractDatesRegex(TransformationBlock, Logger):
    """Extract the creation date of a body of text with a regex.

    The regex tries to match all dates in the format of day-month-year and day {monthname} year. The month name can be in Dutch or English.

    min_date: The minimum date to consider in format %d-%m-%Y.
    max_date: The maximum date to consider in format %d-%m-%Y.
    """

    min_date: InitVar[str | None]
    max_date: InitVar[str | None]

    def __post_init__(self, min_date: str | None = None, max_date: str | None = None) -> None:
        """Initialize the date extractor.

        :param min_date: The minimum date to consider as a string.
        :param max_date: The maximum date to consider as a string.
        """
        super().__post_init__()
        self._min_date = pd.to_datetime(min_date) if min_date else pd.to_datetime("1950-01-01", format="%d-%m-%Y")
        self._max_date = pd.to_datetime(max_date) if max_date else pd.to_datetime(datetime.datetime.now(tz=timezone("CET")).date())

    def custom_transform(self, data: pd.DataFrame, **transform_args: Never) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Generate extracted dates from full body text.

        :param data: The data to transform.
        :param transform_args: [UNUSED] Additional keyword arguments.
        :return: The transformed data.
        """
        if "title" not in data.columns or "full_text" not in data.columns:
            self.log_to_warning("The data does not contain a valid 'title' and/or 'full_text' column. Not extracting dates...")
            return data

        if "date" not in data.columns:
            data["date"] = pd.NaT

        if "title" in data.columns and is_string_dtype(data["title"]):
            data["date"] = data["date"].fillna(data["title"].apply(self.extract_date_regex))  # type: ignore[arg-type]
        if "full_text" in data.columns and is_string_dtype(data["full_text"]):
            data["date"] = data["date"].fillna(data["full_text"].apply(self.extract_date_regex))  # type: ignore[arg-type]

        data["date"] = data["date"].apply(lambda x: x.replace(tzinfo=ZoneInfo("Europe/Amsterdam")) if pd.notna(x) else x)

        return data.drop("index", errors="ignore").dropna(axis="columns", how="all")

    def extract_date_regex(self, full_text: str | None) -> pd.Timestamp | NaTType:
        """Extract the date or creation from a full text with a regular expression.

        :param full_text: The full text to extract the date from.
        :return: The extracted date or NaT if no valid date was found.
        """
        if not full_text:
            return pd.NaT

        matches = DATE_REGEX.findall(full_text)
        date_strings = []
        for match in matches:
            if match[0]:  # Date with day-month-year.
                date_strings.append(match[0])
            else:  # Date with Dutch or English month name.
                day, month_name, year = match[1].split()
                month = str(MONTH_MAP[month_name.lower()])
                date_strings.append(f"{day}-{month}-{year}")

        if not date_strings:
            return pd.NaT

        for date_str in date_strings:
            try:
                d = pd.to_datetime(date_str, format="%d-%m-%Y")
                if self._min_date <= d <= self._max_date:
                    return d
            except ValueError:
                continue
        return pd.NaT
