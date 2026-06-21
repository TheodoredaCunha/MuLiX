from pathlib import Path
from typing import Optional

from datasets import load_dataset

DATA_FILE_NAME = "MusicBench_train_1k.json"


def get_default_data_file() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / DATA_FILE_NAME


def load_musicbench_dataset(data_file: Optional[str] = None):
    path = Path(data_file) if data_file else get_default_data_file()
    return load_dataset("json", data_files={"train": str(path)})
