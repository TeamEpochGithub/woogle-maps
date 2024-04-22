"""A TransformationBlock that extracts the most important sentences from the data."""

from dataclasses import dataclass
from typing import Any, override

import nltk
import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from lexrank import LexRank
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from src.logging.logger import Logger


@dataclass
class ExtractImportantSentences(TransformationBlock, Logger):
    """A TransformationBlock that extracts the most important sentences from the data.

    Expects a dataframe with a full_text column, and gives back the most important sentences in a summary column.
    """

    @override
    def custom_transform(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Extract the most important sentences from the data.

        :param data: A pandas dataframe with a full_text column.
        :param kwargs: Additional keyword arguments (UNUSED).
        :return: A dataframe with the most important sentences in a summary column.
        """
        data = self.merge_whitespace(data)
        data = self.tokenize_sentences(data)
        return self.extract_important_sentences(data).drop("index", errors="ignore")

    def merge_whitespace(self, data: pd.DataFrame) -> pd.DataFrame:
        """Merge the whitespace in the data.

        :param data: a dataframe with a full_text column.
        :return: the dataframe with merged newlines and spaces.
        """
        data["full_text"] = data["full_text"].fillna("")
        data["filtered_text"] = data["full_text"].str.replace(r"\n+", "\n", regex=True)
        data["filtered_text"] = data["filtered_text"].str.replace(r" +", " ", regex=True)
        return data

    def tokenize_sentences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Tokenize the sentences in the data.

        :param data: a dataframe with a filtered_text column.
        :return: a dataframe where the filtered_text column is a list of sentences.
        """
        data["filtered_text"] = data["filtered_text"].str.replace(r"\n", ". ")
        data["filtered_text"] = data["filtered_text"].apply(lambda x: sent_tokenize(x, language="dutch"))
        return data

    def adjust_summary_size(self, num_sentences: int) -> int:
        """Adjust dynamically the number of sentences for the summary.

        :param num_sentences: number of sentences to summarize.
        :return: the number of sentences for the LexRank summary.
        """
        if num_sentences < 20:
            return max(1, num_sentences // 4)
        if num_sentences < 40:
            return num_sentences // 3
        return num_sentences // 2

    def extract_important_sentences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract the important sentences from the data.

        :param data: a dataframe with a filtered_text column, which is a list of sentences.
        :return: a dataframe with the most important sentences in the summary column.
        """
        try:
            # Download the necessary NLTK resources if not already downloaded
            nltk.download("punkt")
            nltk.download("stopwords")
            nltk.download("wordnet")

            trained_model = LexRank(
                data["filtered_text"].to_list(),
                stopwords=set(stopwords.words("dutch")),
            )
            data["summary"] = data["filtered_text"].apply(
                lambda sentences: trained_model.get_summary(
                    sentences,
                    summary_size=self.adjust_summary_size(len(sentences)),
                    threshold=0.3,
                ),
            )
        except ValueError as e:  # ValueError: documents are not informative
            self.log_to_warning(f"Error in extracting important sentences: {e}")

        return data
