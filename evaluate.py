import os
import time
import pickle
import torch
import numpy as np
import pandas as pd
import pandas as pd

from src.config import compute_device, MODELS_DIR, gpu_count, RNG_SEED
from src.hybrid_search import SemanticSearchModel, HybridEngine
from src.sparse_engine import FrequencyVectorizer, TFIDFTransformer

def load_inference_assets():
    tickets = pd.read_csv(os.path.join(MODELS_DIR, 'cleaned_ticket_corpus.csv'))
    
    with open(os.path.join(MODELS_DIR, 'token_registry.pkl'), 'rb') as fh:
        reg = pickle.load(fh)
        
    freq_vec = FrequencyVectorizer()
    freq_vec.token2col = reg['freq_vocab']
    freq_vec.col2token = reg['col_to_token']
    
    idf_vector = np.load(os.path.join(MODELS_DIR, 'term_idf_weights.npy'))
    tfidf_transformer = TFIDFTransformer(freq_vec)
    tfidf_transformer.idf_vector = idf_vector
    
    # Load semantic index
    semantic_idx = torch.load(os.path.join(MODELS_DIR, 'semantic_index.pt'), map_location=compute_device)
    
    # Recreate keyword index sparse tensor
    tfidf_corpus_matrix = np.load(os.path.join(MODELS_DIR, 'tfidf_corpus_matrix.npy'))
    keyword_index = tfidf_transformer.to_sparse_gpu(tfidf_corpus_matrix, compute_device)
    
    # Needs embedding layer
    from src.semantic_engine import parse_glove_file, build_embedding_layer
    from src.config import GLOVE_FILE, VEC_DIM
    pretrained = parse_glove_file(GLOVE_FILE)
    embed_layer, _, _ = build_embedding_layer(freq_vec.token2col, pretrained, VEC_DIM, compute_device)
    
    engine = HybridEngine(
        tickets, freq_vec, tfidf_transformer, keyword_index, semantic_idx,
        reg['token_to_row'], embed_layer, VEC_DIM
    )
    return engine, semantic_idx, tickets

def run_benchmarks(semantic_idx, tickets):
    print("\nRunning GPU benchmark tests...")
    search_model = SemanticSearchModel(semantic_idx)
    if gpu_count > 1:
        search_model = torch.nn.DataParallel(search_model)
    search_model = search_model.to(compute_device).eval()

    np.random.seed(RNG_SEED)
    bench_indices = np.random.choice(len(tickets), size=100, replace=False)
    bench_texts   = tickets['Ticket Description'].iloc[bench_indices].tolist()

    # Create dummy embeddings for speed test
    @torch.no_grad()
    def batch_vectorise(texts):
        # We dummy this out for the raw forward-pass timing
        vecs = torch.randn(len(texts), 300, device=compute_device)
        return torch.nn.functional.normalize(vecs, p=2, dim=1)

    BATCH_SIZES = [10, 25, 50, 75, 100]
    bench_log   = []

    print(f'{"Batch":>10}  {"Total ms":>12}  {"Per query ms":>14}')
    print('-' * 42)

    for bs in BATCH_SIZES:
        q_batch = batch_vectorise(bench_texts[:bs]).to(compute_device)
        _ = search_model(q_batch)  # warmup
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            _ = search_model(q_batch)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        ms = (time.time() - t_start) * 1000
        bench_log.append({'batch_size': bs, 'total_ms': ms, 'ms_per_query': ms / bs})
        print(f'{bs:>10}  {ms:>12.2f}  {ms/bs:>14.3f}')

    bench_df = pd.DataFrame(bench_log)
    bench_path = os.path.join(MODELS_DIR, 'batch_benchmark.csv')
    bench_df.to_csv(bench_path, index=False)
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Execution Time (ms)", "Per-Query Latency (ms)"))
    fig.add_trace(go.Scatter(x=bench_df['batch_size'], y=bench_df['total_ms'], mode='lines+markers', marker_color='#00ff99', name='Total Time', fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench_df['batch_size'], y=bench_df['ms_per_query'], mode='lines+markers', marker_color='#bf00ff', name='Per-Query Time', fill='tozeroy'), row=1, col=2)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a", font=dict(family="Courier New", color="#ffffff"), showlegend=False, title_text="GPU Batch Similarity Benchmark")
    pio.write_json(fig, os.path.join(MODELS_DIR, 'benchmark_plot.json'))

def run_precision(engine, tickets):
    print("\nComputing Precision@5 over 200 evaluation queries...")
    np.random.seed(RNG_SEED)
    eval_set      = np.random.choice(len(tickets), size=200, replace=False)
    kw_scores, sem_scores, hybrid_scores = [], [], []

    for idx, q_idx in enumerate(eval_set):
        q_text     = tickets['Ticket Description'].iloc[q_idx]
        true_label = tickets['Ticket Type'].iloc[q_idx]
        
        for score_list, fn in [
            (kw_scores,     lambda t: engine.retrieve(t, 6, beta=1.0)),
            (sem_scores,    lambda t: engine.retrieve(t, 6, beta=0.0)),
            (hybrid_scores, lambda t: engine.retrieve(t, 6, beta=0.4)),
        ]:
            res  = fn(q_text)
            top5 = res[res.index != q_idx].head(5)
            score_list.append(np.mean(top5['Ticket Type'].values == true_label))
            
        if (idx+1) % 50 == 0: print(f"Evaluated {idx+1}/200")

    kw_val  = float(np.mean(kw_scores))
    sem_val = float(np.mean(sem_scores))
    hyb_val = float(np.mean(hybrid_scores))
    print(f'  Keyword-only  : {kw_val:.4f}')
    print(f'  Semantic-only : {sem_val:.4f}')
    print(f'  Hybrid        : {hyb_val:.4f}')
    
    import json
    with open(os.path.join(MODELS_DIR, 'precision_metrics.json'), 'w') as f:
        json.dump({'Keyword': kw_val, 'Semantic': sem_val, 'Hybrid': hyb_val}, f)
        
    import plotly.graph_objects as go
    import plotly.io as pio
    fig_p = go.Figure(data=[
        go.Bar(x=['Keyword Only<br>(beta=1.0)'], y=[kw_val], marker_color='#4FC3F7', text=[f'{kw_val:.4f}'], textposition='outside'),
        go.Bar(x=['Semantic Only<br>(beta=0.0)'], y=[sem_val], marker_color='#81C784', text=[f'{sem_val:.4f}'], textposition='outside'),
        go.Bar(x=['Hybrid<br>(beta=0.4)'], y=[hyb_val], marker_color='#FFB74D', text=[f'{hyb_val:.4f}'], textposition='outside')
    ])
    fig_p.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(family="Courier New", color="#ffffff"), showlegend=False,
        title_text="Retrieval Method Comparison — Precision@5",
        yaxis_title="Precision@5", yaxis=dict(range=[0, 1.1])
    )
    pio.write_json(fig_p, os.path.join(MODELS_DIR, 'precision_plot.json'))

def generate_comparison_plot(engine):
    print("\nGenerating Retrieval Comparison Plot...")
    query = "My monthly subscription bill is completely incorrect and I want my money back"
    kw_res  = engine.retrieve(query, top_n=5, beta=1.0)
    sem_res = engine.retrieve(query, top_n=5, beta=0.0)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Keyword Engine (TF-IDF)", "Semantic Engine (GloVe)"))
    
    def make_unique(labels):
        seen = set()
        res = []
        for l in labels:
            while l in seen:
                l += " "
            seen.add(l)
            res.append(l)
        return res

    kw_labels = make_unique([f"{t}<br>({p})" for t, p in zip(kw_res['Ticket Type'], kw_res['Ticket Priority'])])
    fig.add_trace(go.Bar(x=kw_res['kw_score'], y=kw_labels, orientation='h', marker_color='#4FC3F7'), row=1, col=1)
    
    sem_labels = make_unique([f"{t}<br>({p})" for t, p in zip(sem_res['Ticket Type'], sem_res['Ticket Priority'])])
    fig.add_trace(go.Bar(x=sem_res['sem_score'], y=sem_labels, orientation='h', marker_color='#81C784'), row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(family="Courier New", color="#ffffff"), showlegend=False,
        title_text=f"Retrieval Comparison — Keyword vs Semantic<br>Query: \"{query}\"",
        yaxis=dict(autorange="reversed"), yaxis2=dict(autorange="reversed")
    )
    pio.write_json(fig, os.path.join(MODELS_DIR, 'comparison_plot.json'))

def main():
    print("Loading indices for evaluation (this might take a minute)...")
    engine, semantic_idx, tickets = load_inference_assets()
    run_benchmarks(semantic_idx, tickets)
    run_precision(engine, tickets)
    generate_comparison_plot(engine)
    print("Evaluation complete!")

if __name__ == '__main__':
    main()
