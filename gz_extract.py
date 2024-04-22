"""Utility script for unzipping .gz data files."""

import functools
import gzip
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)


@dataclass
class GzExtractConfig:
    """Configuration for the gz_extract script.

    :param source_path: Path to the directory containing the .gz files
    :param target_path: Path to the directory where the extracted files will be saved
    :param extension: File extension of the files
    """

    source_path: Path
    target_path: Path
    extension: str


cs = ConfigStore.instance()
cs.store(name="gz_extract_config", node=GzExtractConfig)


def _extract_file(file: Path, target_path: Path) -> None:
    """Extract the .gz file and save the extracted file in the target directory.

    :param file: Path to the .gz file
    :param target_path: Path to the directory where the extracted file will be saved
    """
    with gzip.open(file, "rb") as f_in, open(target_path / file.stem, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


@hydra.main(version_base=None, config_path="conf", config_name="gz_extract")
def gz_extract(cfg: GzExtractConfig) -> None:
    """Unzip .gz files in the source directory and save the extracted files in the target directory.

    :param cfg: Configuration for the gz_extract script
    """
    files = list(Path(cfg.source_path).glob(f"**/*{cfg.extension}"))
    logger.info("Extracting %s files from %s to %s. This may take a few minutes.", len(files), cfg.source_path, cfg.target_path)
    process_map(functools.partial(_extract_file, target_path=Path(cfg.target_path)), files, max_workers=len(files))
    logger.info("Extraction complete.")


if __name__ == "__main__":
    gz_extract()
