"""MVCC visibility: a chunk is visible at epoch e iff born <= e < dead and its document is allowed."""
from __future__ import annotations

import numpy as np


def visible_mask(
    born: np.ndarray,
    dead: np.ndarray,
    doc_ix: np.ndarray,
    n_chunks: int,
    epoch: int,
    doc_allowed: np.ndarray,
) -> np.ndarray:
    b, d, ix = born[:n_chunks], dead[:n_chunks], doc_ix[:n_chunks]
    return (b <= epoch) & (d > epoch) & doc_allowed[ix]
