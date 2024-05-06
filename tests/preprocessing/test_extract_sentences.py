"""Tests for the ExtractImportantSentences class."""
from unittest import TestCase

import pandas as pd

from src.preprocessing.extract_important_sentences import ExtractImportantSentences
from .utils import get_test_df


class ExtractImportantSentencesTest(TestCase):
    """Tests for the ExtractImportantSentences class."""

    def test_merge_whitespace_with_valid_data(self) -> None:
        """Merge whitespace should merge the whitespace in the data."""

        data = pd.DataFrame(
            {
                "full_text": ["This is a    test\n\nsentence."],
            },
        )
        transformer = ExtractImportantSentences()
        result = transformer.merge_whitespace(data)
        self.assertEqual(result["filtered_text"].iloc[0], "This is a test\nsentence.")

    def test_merge_whitespace_with_empty_data(self) -> None:
        """Merge whitespace should not break on empty data."""

        data = pd.DataFrame(
            {
                "full_text": [""],
            },
        )
        transformer = ExtractImportantSentences()
        result = transformer.merge_whitespace(data)
        self.assertEqual(result["filtered_text"].iloc[0], "")

    def test_tokenize_sentences_with_valid_data(self) -> None:
        """tokenize_sentences should tokenize the sentences in the data."""

        data = pd.DataFrame(
            {
                "filtered_text": ["Dit is een test. Een andere test"],
            },
        )
        transformer = ExtractImportantSentences()
        result = transformer.tokenize_sentences(data)
        self.assertEqual(result["filtered_text"].iloc[0], ["Dit is een test.", "Een andere test"])

    def test_tokenize_sentences_with_empty_data(self) -> None:
        """tokenize_sentences should not break on empty data."""

        data = pd.DataFrame(
            {
                "filtered_text": [""],
            },
        )
        transformer = ExtractImportantSentences()
        result = transformer.tokenize_sentences(data)
        self.assertEqual(result["filtered_text"].iloc[0], [])

    def test_extract_important_sentences_length(self) -> None:
        """Extract important sentences should extract the most important sentences from the data."""

        data = get_test_df()
        transformer = ExtractImportantSentences()
        clean_data = transformer.merge_whitespace(data)
        clean_data = transformer.tokenize_sentences(clean_data)
        result = transformer.extract_important_sentences(clean_data)
        self.assertTrue("summary" in result.columns)
        self.assertEqual(result["summary"].apply(len).tolist(), (clean_data["filtered_text"].apply(len)).tolist())
