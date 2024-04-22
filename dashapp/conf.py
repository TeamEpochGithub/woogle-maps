"""Configuration file for the dash server."""

from pathlib import Path

RAW_DATA_PATH = Path("./data/raw/")
PROCESSED_DATA_PATH = Path("./data/processed/")
FINAL_DATA_PATH = Path("./data/final/")
UPLOAD_DATA_PATH = Path("./data/uploads/")

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
FINAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
UPLOAD_DATA_PATH.mkdir(parents=True, exist_ok=True)
