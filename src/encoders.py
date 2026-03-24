import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class OrdinalCategoryEncoder:
    UNKNOWN_TOKEN = '__UNK__'

    def __init__(self, ordering: Optional[List[str]] = None):
        self.ordering    = ordering
        self.label_map   : Dict[str, int] = {}
        self.reverse_map : Dict[int, str] = {}
        self.unk_code    = -1

    def fit(self, col: pd.Series) -> 'OrdinalCategoryEncoder':
        categories = (
            self.ordering if self.ordering
            else sorted(col.dropna().unique().tolist())
        )
        self.label_map   = {cat: idx for idx, cat in enumerate(categories)}
        self.reverse_map = {idx: cat for cat, idx in self.label_map.items()}
        self.unk_code    = len(self.label_map)
        self.label_map[self.UNKNOWN_TOKEN]   = self.unk_code
        self.reverse_map[self.unk_code]      = self.UNKNOWN_TOKEN
        return self

    def encode(self, col: pd.Series) -> np.ndarray:
        return np.array(
            [self.label_map.get(v, self.unk_code) for v in col],
            dtype=np.int32
        )

    def fit_encode(self, col: pd.Series) -> np.ndarray:
        return self.fit(col).encode(col)

    def decode(self, codes: np.ndarray) -> List[str]:
        return [self.reverse_map.get(c, self.UNKNOWN_TOKEN) for c in codes]

class BinaryVectorEncoder:
    def __init__(self):
        self.known_cats   : List[str]      = []
        self.cat_position : Dict[str, int]  = {}
        self.n_bits       : int             = 0

    def fit(self, col: pd.Series) -> 'BinaryVectorEncoder':
        self.known_cats   = sorted(col.dropna().unique().tolist())
        self.cat_position = {c: i for i, c in enumerate(self.known_cats)}
        self.n_bits       = len(self.known_cats)
        return self

    def encode(self, col: pd.Series) -> np.ndarray:
        out = np.zeros((len(col), self.n_bits), dtype=np.float32)
        for row_idx, val in enumerate(col):
            bit_pos = self.cat_position.get(val)
            if bit_pos is not None:
                out[row_idx, bit_pos] = 1.0
        return out

    def fit_encode(self, col: pd.Series) -> np.ndarray:
        return self.fit(col).encode(col)

    def read_vector(self, vec: np.ndarray) -> str:
        hot = int(np.argmax(vec))
        return self.known_cats[hot] if vec[hot] > 0 else '__UNKNOWN__'
