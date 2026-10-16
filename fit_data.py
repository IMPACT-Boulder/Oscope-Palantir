"""Shared operations on exported fit tables."""

import numpy as np
import pandas as pd


def best_fit_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep the most complete row per key, preferring the last row on ties.

    Select whole rows instead of using GroupBy.apply, whose handling of
    grouping columns changed in pandas 3. Preserve the selected input order
    and support duplicate index labels and missing grouping values.
    """
    scores = frame.notna().sum(axis=1).to_numpy()
    order = np.argsort(scores, kind="stable")
    ranked = frame.iloc[order]
    keep = ~ranked.duplicated(subset=keys, keep="last")
    positions = np.sort(order[keep.to_numpy()])
    return frame.iloc[positions].reset_index(drop=True)
