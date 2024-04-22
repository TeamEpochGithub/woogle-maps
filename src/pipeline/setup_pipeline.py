"""Parse the pipeline form the configuration and set it up."""

import hydra.utils
from omegaconf import DictConfig

from src.pipeline.verbose_transformation_pipeline import VerboseTransformationPipeline


def setup_pipeline(cfg: DictConfig) -> VerboseTransformationPipeline:
    """Parse the pipeline form the configuration and set it up.

    :param cfg: The configuration for the pipeline.
    :return: The transformation pipeline.
    """
    steps = [hydra.utils.instantiate(block) for block in cfg.pipeline]

    return VerboseTransformationPipeline(steps=steps, title="Preprocessing Pipeline")
