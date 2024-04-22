"""Generate the embeddings for the data using a RoBERTa model."""

from dataclasses import dataclass, field
from typing import Any, override

import pandas as pd
import torch
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock
from transformers import RobertaModel, RobertaTokenizer

from src.logging.logger import Logger


@dataclass
class GenerateRobertaEmbedding(TransformationBlock, Logger):
    """Generate the RoBERTa embeddings for the data.

    :param pretrained_model_name_or_path: The name or path of the pretrained model.
    """

    pretrained_model_name_or_path: str

    _tokenizer: RobertaTokenizer = field(init=False, repr=False)
    _model: RobertaModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Roberta tokenizer and model."""
        super().__post_init__()
        self._tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self._model = RobertaModel.from_pretrained(self.pretrained_model_name_or_path)
        # self._model.to('cuda' if torch.cuda.is_available() else 'cpu')

    @override
    def custom_transform(self, data: pd.DataFrame, **transform_args: Any) -> pd.DataFrame:  # noqa: DOC103  # type: ignore[misc]
        """Generate the RoBERTa embeddings for the data.

        :param data: The data to transform.
        :param transform_args: Additional keyword arguments (UNUSED).
        :return: The transformed data.
        """
        docs = data["full_text"].astype(str).tolist()

        with torch.inference_mode():
            output = self._model(**self._tokenizer(docs, padding=True, truncation=True, return_tensors="pt"))  # TODO(Jeffrey): Add device; Add progress bar
            data["embed"] = output.last_hidden_state

        return data
