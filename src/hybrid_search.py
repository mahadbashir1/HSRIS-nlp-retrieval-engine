import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple
from src.config import compute_device, gpu_count
from src.sparse_engine import FrequencyVectorizer, TFIDFTransformer
from src.semantic_engine import compute_sentence_vector

class SemanticSearchModel(nn.Module):
    def __init__(self, corpus_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer('corpus_matrix', corpus_matrix)
    def forward(self, query_batch: torch.Tensor) -> torch.Tensor:
        return torch.mm(query_batch, self.corpus_matrix.T)

class HybridEngine:
    def __init__(self, tickets_df: pd.DataFrame, 
                 freq_vec: FrequencyVectorizer, tfidf_transformer: TFIDFTransformer,
                 keyword_index: torch.Tensor, semantic_idx: torch.Tensor,
                 token_to_row: dict, embed_layer: nn.Embedding, vec_dim: int):
        self.tickets = tickets_df
        self.freq_vec = freq_vec
        self.tfidf_transformer = tfidf_transformer
        self.keyword_index = keyword_index
        self.semantic_idx = semantic_idx
        self.token_to_row = token_to_row
        self.embed_layer = embed_layer
        self.vec_dim = vec_dim

    @torch.no_grad()
    def query_to_keyword_vec(self, query_text: str) -> torch.Tensor:
        tf_row    = self.freq_vec.vectorize([query_text])
        tfidf_row = tf_row * self.tfidf_transformer.idf_vector[np.newaxis, :]
        tfidf_row /= np.maximum(np.linalg.norm(tfidf_row), 1e-9)
        return torch.tensor(tfidf_row, dtype=torch.float32, device=compute_device)

    @torch.no_grad()
    def query_to_semantic_vec(self, query_text: str) -> torch.Tensor:
        return compute_sentence_vector(
            query_text, self.token_to_row, self.freq_vec.token2col, 
            self.tfidf_transformer.idf_vector, self.embed_layer, 
            self.vec_dim, compute_device
        ).unsqueeze(0)

    @torch.no_grad()
    def keyword_similarity(self, q_vec: torch.Tensor) -> torch.Tensor:
        return torch.mm(q_vec, self.keyword_index.to_dense().T).squeeze(0)

    @torch.no_grad()
    def semantic_similarity(self, q_vec: torch.Tensor) -> torch.Tensor:
        return torch.mm(q_vec, self.semantic_idx.T).squeeze(0)

    def retrieve(self, query_text: str, top_n: int = 5, beta: float = 0.4) -> pd.DataFrame:
        kw_vec   = self.query_to_keyword_vec(query_text)
        sem_vec  = self.query_to_semantic_vec(query_text)
        
        kw_sim   = self.keyword_similarity(kw_vec)
        sem_sim  = self.semantic_similarity(sem_vec)
        
        combined = beta * kw_sim + (1.0 - beta) * sem_sim
        top_pos  = torch.topk(combined, k=top_n).indices.cpu().numpy()
        
        res      = self.tickets.iloc[top_pos].copy()
        res['kw_score']  = kw_sim[top_pos].cpu().numpy()
        res['sem_score'] = sem_sim[top_pos].cpu().numpy()
        res['combined']  = combined[top_pos].cpu().numpy()
        return res
