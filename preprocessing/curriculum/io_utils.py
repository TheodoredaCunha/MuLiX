import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def load_json(path: PathLike) -> Any:
    with resolve_project_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: PathLike) -> Path:
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return output_path


def load_caption_columns(path: PathLike) -> Dict[str, List[str]]:
    rows = load_json(path)
    return {
        "main_caption": [row["main_caption"] for row in rows],
        "alt_caption": [row["alt_caption"] for row in rows],
    }
