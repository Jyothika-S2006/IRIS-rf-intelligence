"""
data_loader.py — Utility for loading IQ samples by sample_id
Used by Langflow tool functions.
"""
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).parent
IQ_DIR = ROOT / "data" / "iq"


def load_iq_by_id(sample_id: str) -> np.ndarray:
    """
    Load a saved IQ sample from data/iq/<sample_id>.npy.
    Raises FileNotFoundError if not found.
    """
    path = IQ_DIR / f"{sample_id}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"IQ sample '{sample_id}' not found at {path}. "
            "Run train_all.py to generate the dataset first."
        )
    return np.load(path)


def list_sample_ids(label_name: str = None) -> list:
    """
    Return all sample IDs, optionally filtered by label name prefix.
    Example: list_sample_ids("narrowband_cw") → ["narrowband_cw_0000", ...]
    """
    if not IQ_DIR.exists():
        return []
    files = sorted(IQ_DIR.glob("*.npy"))
    ids   = [f.stem for f in files]
    if label_name:
        ids = [s for s in ids if s.startswith(label_name)]
    return ids
