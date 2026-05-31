
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def read_dataset_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Charge un dataset patient en préservant les valeurs catégorielles strings.

    En particulier, la modalité « None » de la variable `alcohol` n'est PAS
    convertie en NaN — c'est une chaîne légitime.
    """
    return pd.read_csv(
        path,
        keep_default_na=False,    # ne convertit AUCUNE chaîne en NaN par défaut
        na_values=[""],            # seule la chaîne vide reste un NaN
    )
