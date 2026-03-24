import re
import torch
import numpy as np
from collections import Counter
from typing import List, Dict, Optional

STOP_TOKENS = frozenset([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once',
    'here','there','when','where','why','how','all','both','each','few',
    'more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','s','t','can','will','just','don','should',
    'now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn',
    'doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn',
    'shan','shouldn','wasn','weren','won','wouldn'
])

def split_into_tokens(text: str, drop_stops: bool = True, min_chars: int = 2) -> List[str]:
    raw_tokens = re.findall(r'[a-zA-Z]+', str(text))
    tokens     = [t.lower() for t in raw_tokens if len(t) >= min_chars]
    if drop_stops:
        tokens = [t for t in tokens if t not in STOP_TOKENS]
    return tokens

def build_ngrams(token_list: List[str], window: int) -> List[str]:
    return ['-'.join(token_list[i: i + window])
            for i in range(len(token_list) - window + 1)]

def extract_features(text: str, include_bigrams: bool = True, include_trigrams: bool = True) -> List[str]:
    tokens   = split_into_tokens(text)
    features = list(tokens)
    if include_bigrams:  features += build_ngrams(tokens, 2)
    if include_trigrams: features += build_ngrams(tokens, 3)
    return features

class FrequencyVectorizer:
    def __init__(self, vocab_limit: int = 5000, bigrams: bool = True, trigrams: bool = True):
        self.vocab_limit = vocab_limit
        self.bigrams     = bigrams
        self.trigrams    = trigrams
        self.token2col   : Dict[str, int] = {}
        self.col2token   : Dict[int, str] = {}

    def fit(self, documents: List[str]) -> 'FrequencyVectorizer':
        freq = Counter()
        for doc in documents:
            freq.update(extract_features(doc, self.bigrams, self.trigrams))
        top_tokens     = [tok for tok, _ in freq.most_common(self.vocab_limit)]
        self.token2col = {tok: col for col, tok in enumerate(top_tokens)}
        self.col2token = {col: tok for tok, col in self.token2col.items()}
        return self

    def vectorize(self, documents: List[str]) -> np.ndarray:
        V, N = len(self.token2col), len(documents)
        mat  = np.zeros((N, V), dtype=np.float32)
        for row, doc in enumerate(documents):
            feats = extract_features(doc, self.bigrams, self.trigrams)
            denom = max(len(feats), 1)
            for f in feats:
                col = self.token2col.get(f)
                if col is not None:
                    mat[row, col] += 1.0 / denom
        return mat

    def fit_vectorize(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).vectorize(documents)

class TFIDFTransformer:
    def __init__(self, vectorizer: FrequencyVectorizer, smooth: bool = True):
        self.vectorizer = vectorizer
        self.smooth     = smooth
        self.idf_vector : Optional[np.ndarray] = None

    def _idf(self, tf_mat: np.ndarray) -> np.ndarray:
        N        = tf_mat.shape[0]
        doc_freq = np.count_nonzero(tf_mat, axis=0).astype(np.float64)
        if self.smooth:
            scores = np.log((1.0 + N) / (1.0 + doc_freq)) + 1.0
        else:
            doc_freq = np.maximum(doc_freq, 1)
            scores   = np.log(N / doc_freq) + 1.0
        return scores.astype(np.float32)

    def fit(self, tf_mat: np.ndarray) -> 'TFIDFTransformer':
        self.idf_vector = self._idf(tf_mat)
        return self

    def to_sparse_gpu(self, tf_mat: np.ndarray, target_device: torch.device) -> torch.Tensor:
        weighted  = tf_mat * self.idf_vector[np.newaxis, :]
        row_norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        weighted /= np.maximum(row_norms, 1e-9)
        nz_pos    = np.argwhere(weighted != 0)
        nz_val    = weighted[nz_pos[:, 0], nz_pos[:, 1]]
        return torch.sparse_coo_tensor(
            torch.tensor(nz_pos.T, dtype=torch.long),
            torch.tensor(nz_val,   dtype=torch.float32),
            size=weighted.shape,
            device=target_device
        ).coalesce()

    def fit_transform(self, tf_mat: np.ndarray, target_device: torch.device) -> torch.Tensor:
        return self.fit(tf_mat).to_sparse_gpu(tf_mat, target_device)
