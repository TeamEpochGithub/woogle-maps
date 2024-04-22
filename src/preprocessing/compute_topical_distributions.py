"""Generates topical distributions."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, override

import nltk
import numpy as np
import numpy.typing as npt
import pandas as pd
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas.core.dtypes.common import is_string_dtype

from src.logging.logger import Logger


@dataclass
class TopicalDistribution(TransformationBlock, Logger):
    """Initialize the topical distribution pipeline block.

    :param pretrained_model_name_or_path: LDA model name or path.
    :param _pretrained_lda: LDA model instance.
    :param _lemmatizer: WordNetLemmatizer instance.
    :param _dict: Dictionary of the corpus.
    """

    pretrained_model_name_or_path: str
    dictionary_name_or_path: str

    _pretrained_lda: LdaModel = field(init=False, repr=False)
    _lemmatizer: WordNetLemmatizer = field(init=False, repr=False)
    _dict: Dictionary = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the topical distribution pipeline block."""
        super().__post_init__()
        self._pretrained_lda = LdaModel.load(self.pretrained_model_name_or_path)
        self._lemmatizer = WordNetLemmatizer()
        self._dict = Dictionary.load(self.dictionary_name_or_path)

    @override
    def custom_transform(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Ensure the input Dataframe has the relevant columns.

        Then computes the topical distributions for each document.

        :param data: The input dataframe.
        :param kwargs: Additional keyword arguments (UNUSED).
        :return: The transformed data.
        """
        if "full_text" not in data.columns or not is_string_dtype(data["full_text"].dtype):
            self.log_to_warning("The data does not contain a valid 'full_text' column. Not computing topical distributions...")
            return data

        # Download the necessary NLTK resources if not already downloaded
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

        # Tokenize and preprocess the documents
        preprocessed_docs = self.preprocess_documents(data["full_text"])

        docs_bow = [self._dict.doc2bow(doc) for doc in preprocessed_docs]
        # Compute the topic distribution for each document
        data["topical_dist"] = [self.get_topic_dist(doc_bow).tolist() for doc_bow in docs_bow]

        return data.drop("index", errors="ignore")

    def preprocess_documents(self, docs: Iterable[str]) -> list[list[str]]:
        """Preprocess a list of documents.

        Tokenize, remove stopwords, and lemmatize the documents.

        :param docs: The list of document texts.
        :return: The preprocessed documents.
        """
        preprocessed_docs: list[list[str]] = []
        for doc in docs:
            if not doc:
                preprocessed_docs.append([])
                continue
            # Tokenize
            tokens = word_tokenize(doc.lower())
            # Remove stopwords and lemmatize
            filtered_tokens = [self._lemmatizer.lemmatize(w) for w in tokens if w.isalpha()]
            preprocessed_docs.append(filtered_tokens)
        return preprocessed_docs

    def get_topic_dist(self, doc_bow: list[tuple[int, int]]) -> npt.NDArray[np.float64]:
        """Compute the topical distribution for a given document.

        :param doc_bow: BoW representation of the document.
        :return: The topical distribution.
        """
        dist = np.zeros(self._pretrained_lda.num_topics)
        for topic, prob in self._pretrained_lda.get_document_topics(doc_bow, minimum_probability=0):
            dist[topic] = prob
        return np.array(dist)
