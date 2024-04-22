"""Preprocess the data and train the model."""

from pathlib import Path

import hydra
from epochalyst.logging.section_separator import print_section_separator
from omegaconf import DictConfig
from transformers import set_seed

from src.pipeline.setup_pipeline import setup_pipeline


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run_train(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use TrainConfig instead of DictConfig
    """Train a model pipeline. Entry point for Hydra which loads the config file.

    :param cfg: Configuration for the training script
    """
    print_section_separator("Q3 National Archive - Preparing the data.")
    set_seed(42)

    pipeline = setup_pipeline(cfg)
    pipeline.run_dossier_pipeline(cfg.dossier_id, Path(cfg.raw_data_path), Path(cfg.processed_data_path), Path(cfg.final_data_path))


if __name__ == "__main__":
    run_train()
