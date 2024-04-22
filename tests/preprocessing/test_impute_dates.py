"""Tests impute dates block."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl

from src.preprocessing.impute_dates import ImputeDates


class TestImputeDates:
    """Test class for ImputeDate TransformerBlock."""

    def test__impute_dates_1(self) -> None:
        """First test instance."""
        block = ImputeDates()
        test_df = pd.DataFrame(
            {
                "date": [datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")), datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")), None, datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC"))],
                "embed": [np.array((-1, 0)), np.array((-1, 1)), np.array((1, 1)), np.array((1, 0.5))],
            },
        )
        res_df = block.custom_transform(test_df)
        assert res_df.equals(
            pd.DataFrame(
                {
                    "date": [
                        datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")),
                        datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")),
                        datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC")),
                        datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC")),
                    ],
                    "embed": [np.array((-1, 0)), np.array((-1, 1)), np.array((1, 1)), np.array((1, 0.5))],
                },
            ),
        )

    def test__impute_dates_2(self) -> None:
        """Second test instance with polars dataset."""
        block = ImputeDates()
        test_df = pd.DataFrame(
            {
                "date": [datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")), None, None, datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC"))],
                "embed": [np.array((-1, 0)), np.array((-0.5, 0)), np.array((1, 1)), np.array((1, 0.5))],
            },
        )
        res_df = block.transform(test_df)
        assert res_df.equals(
            pd.DataFrame(
                {
                    "date": [
                        datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")),
                        datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")),
                        datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC")),
                        datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC")),
                    ],
                    "embed": [np.array((-1, 0)), np.array((-0.5, 0)), np.array((1, 1)), np.array((1, 0.5))],
                },
            ),
        )

    def test__impute_dates_3(self) -> None:
        block = ImputeDates()
        test_df = pd.DataFrame(
            {
                "date": [datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")), pd.NaT, pd.NaT, datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC"))],
                "embed": [np.array((0, 1)), np.array((0.5, 1)), np.array((1, 0.5)), np.array((1, 0))],
            },
        )
        res_df = block.transform(test_df)
        assert res_df.equals(
            pd.DataFrame(
                {
                    "date": [
                        datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")),
                        datetime(2022, 1, 1, tzinfo=ZoneInfo("UTC")),
                        datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC")),
                        datetime(2023, 1, 2, tzinfo=ZoneInfo("UTC")),
                    ],
                    "embed": [np.array((0, 1)), np.array((0.5, 1)), np.array((1, 0.5)), np.array((1, 0))],
                },
            ),
        )
