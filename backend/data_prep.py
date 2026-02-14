"""
Data preparation for n×p matrix construction.

Provides merge/join helpers so users can build the analysis matrix
from multiple sheets or tables inside the app (see docs/data_model.md).
"""

import logging
from typing import Optional, List, Tuple

import pandas as pd

logger = logging.getLogger("data_prep")


def merge_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    on_a: Optional[List[str]] = None,
    on_b: Optional[List[str]] = None,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge two DataFrames into one (n×p matrix for analysis).

    Args:
        df_a: Left table.
        df_b: Right table.
        on_a: Column name(s) in df_a for the join key. If None and on_b is None, uses first column.
        on_b: Column name(s) in df_b for the join key. If None, uses same as on_a when possible.
        how: Join type: 'inner', 'left', 'right', 'outer'. Default 'inner'.

    Returns:
        Merged DataFrame (single n×p table).
    """
    if on_a is None and on_b is None:
        on_a = [df_a.columns[0]]
        on_b = [df_b.columns[0]]
    elif on_b is None:
        on_b = on_a
    merged = pd.merge(df_a, df_b, left_on=on_a, right_on=on_b, how=how, suffixes=("_left", "_right"))
    logger.info(f"Merge: {df_a.shape} + {df_b.shape} -> {merged.shape} (how={how})")
    return merged
