
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def read_dataset_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Charge un dataset patient en préservant les valeurs catégorielles strings."""
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""],
    )
