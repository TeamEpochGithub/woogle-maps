"""Test the extract_dates_regex.py module."""

import locale
from unittest import TestCase

import pandas as pd

from src.preprocessing.extract_dates_regex import ExtractDatesRegex

locale.setlocale(locale.LC_ALL, "nl_NL")


class ExtractDateRegexTest(TestCase):
    """Test the ExtractDatesRegex class."""

    date_extractor = ExtractDatesRegex(min_date="01-01-1970", max_date="31-12-2024")

    def test_extract_date_nat(self) -> None:
        """Test if the date extractor returns NaT when the input is empty."""
        full_text = "Dit is een testzin. Er zit geen datum in."
        extracted_date = self.date_extractor.extract_date_regex(full_text)
        assert pd.isna(extracted_date)

    def test_extract_date_good(self) -> None:
        """Test if the date extractor returns the correct date when the input has a valid date."""
        full_text = "Dit is een testzin. Er zit 1 datum in: 1 januari 2022."
        extracted_date = self.date_extractor.extract_date_regex(full_text)
        assert extracted_date == pd.to_datetime("01-01-2022", format="%d-%m-%Y")

    def test_extract_date_good2(self) -> None:
        """Test if the date extractor returns the correct date when the input has 2 valid dates."""
        full_text = "Dit is een testzin. Er zitten 2 datums in: 1 januari 2022 en 2 februari 2022."
        extracted_date = self.date_extractor.extract_date_regex(full_text)
        assert extracted_date == pd.to_datetime("01-01-2022", format="%d-%m-%Y")

    def test_extract_date_mixed(self) -> None:
        """Test if the date extractor returns the correct date when the input has 3 valid dates."""
        full_text = "Dit is een testzin. Er zitten 3 datums in: 3 NART 2023, 22 februari 2023 en 3 maart 2023."
        extracted_date = self.date_extractor.extract_date_regex(full_text)
        assert extracted_date == pd.to_datetime("22-02-2023", format="%d-%m-%Y")

    def test_extract_date_bad(self) -> None:
        """Test if the date extractor returns NaT when the input has an invalid date."""
        full_text = "Dit is een testzin. Er zit een foute datum in: 32 januari 2022."
        extracted_date = self.date_extractor.extract_date_regex(full_text)
        assert pd.isna(extracted_date)

    def test_extract_date_between_min_max(self) -> None:
        """Test if the date extractor returns the correct date when the input has dates outside the min and max date."""
        full_text = "Dit is een testzin. Er zitten 3 datums in: 1 januari 1960, 1 december 2050 en 1 januari 2022."
        extracted_date = self.date_extractor.extract_date_regex(full_text)
        assert extracted_date == pd.to_datetime("01-01-2022", format="%d-%m-%Y")
