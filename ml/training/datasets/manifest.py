from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ml.training.datasets.schemas import CSV_COLUMNS


def write_manifest(records: Iterable[dict], path: str | Path) -> pd.DataFrame:
    dataframe = pd.DataFrame(list(records))
    for column in CSV_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None
    dataframe = dataframe[CSV_COLUMNS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return dataframe


def load_manifest(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    for column in CSV_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None
    return dataframe[CSV_COLUMNS]


def load_manifests(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [load_manifest(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.concat(frames, ignore_index=True)
