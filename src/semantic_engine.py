import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Dict, List
from src.sparse_engine import split_into_tokens

def parse_glove_file(filepath: str) -> Dict[str, np.ndarray]:
    word_vecs = {}
    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.rstrip().split(' ')
            word_vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return word_vecs

def build_embedding_layer(freq_vocab: Dict[str, int], pretrained_vecs: Dict[str, np.ndarray], vec_dim: int, compute_device: torch.device):
    unigram_subset = {k: v for k, v in freq_vocab.items() if '-' not in k}
    unigram_list   = list(unigram_subset.keys())
    
    weight_matrix  = np.zeros((len(unigram_list) + 1, vec_dim), dtype=np.float32)
    token_to_row   = {}
    oov_total      = 0

    for pos, tok in enumerate(unigram_list):
        row_idx           = pos + 1
        token_to_row[tok] = row_idx
        if tok in pretrained_vecs:
            weight_matrix[row_idx] = pretrained_vecs[tok]
        else:
            weight_matrix[row_idx] = np.random.normal(0.0, 0.01, vec_dim).astype(np.float32)
            oov_total += 1

    embed_layer = nn.Embedding.from_pretrained(
        torch.tensor(weight_matrix), freeze=True, padding_idx=0
    ).to(compute_device)

    return embed_layer, token_to_row, oov_total

@torch.no_grad()
def compute_sentence_vector(text: str, token_to_row: Dict[str, int], freq_vocab: Dict[str, int], 
                            idf_vector: np.ndarray, embed_layer: nn.Embedding, 
                            vec_dim: int, compute_device: torch.device) -> torch.Tensor:
    tokens = split_into_tokens(text)
    if not tokens:
        return torch.zeros(vec_dim, device=compute_device)
        
    local_tf     = Counter(tokens)
    total_toks   = sum(local_tf.values())
    accumulated  = torch.zeros(vec_dim, device=compute_device)
    total_weight = 0.0
    
    for tok, freq in local_tf.items():
        row_idx = token_to_row.get(tok)
        col_idx = freq_vocab.get(tok)
        if row_idx is None or col_idx is None:
            continue
        w             = (freq / total_toks) * float(idf_vector[col_idx])
        tok_embed     = embed_layer(torch.tensor([row_idx], device=compute_device)).squeeze(0)
        accumulated  += w * tok_embed
        total_weight += w
        
    if total_weight < 1e-9:
        return torch.zeros(vec_dim, device=compute_device)
        
    return F.normalize((accumulated / total_weight).unsqueeze(0), p=2, dim=1).squeeze(0)
