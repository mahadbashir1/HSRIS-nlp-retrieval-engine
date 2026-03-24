import os
import time
import pickle
import torch
import numpy as np

from src.config import compute_device, MODELS_DIR, VEC_DIM, VOCAB_LIMIT, GLOVE_FILE
from src.data_processing import load_and_clean_data
from src.encoders import OrdinalCategoryEncoder, BinaryVectorEncoder
from src.sparse_engine import FrequencyVectorizer, TFIDFTransformer
from src.semantic_engine import parse_glove_file, build_embedding_layer, compute_sentence_vector

def main():
    print("="*60)
    print("  HSRIS Pipeline Builder")
    print("="*60)
    
    # 1. Load Data
    print(f"Loading data from config paths...")
    try:
        tickets, raw_df = load_and_clean_data()
    except FileNotFoundError:
        print("ERROR: Data files not found. Did you download them to the data/ folder?")
        return

    print(f"Working corpus size: {len(tickets):,} tickets")
    
    # 2. Categorical Encoders
    print("Fitting Categorical Encoders...")
    priority_enc = OrdinalCategoryEncoder(ordering=['Low', 'Medium', 'High'])
    priority_codes = priority_enc.fit_encode(tickets['Ticket Priority'])

    channel_enc = BinaryVectorEncoder()
    channel_matrix = channel_enc.fit_encode(tickets['Ticket Channel'])

    type_enc = OrdinalCategoryEncoder()
    type_codes = type_enc.fit_encode(tickets['Ticket Type'])

    encoder_bundle = {
        'priority_encoder': priority_enc,
        'channel_encoder' : channel_enc,
        'type_encoder'    : type_enc
    }
    with open(os.path.join(MODELS_DIR, 'encoder_bundle.pkl'), 'wb') as fh:
        pickle.dump(encoder_bundle, fh)
        
    # 3. Sparse Representation
    print('Constructing vocabulary from corpus ...')
    tick = time.time()
    desc_corpus = tickets['Ticket Description'].tolist()
    freq_vec    = FrequencyVectorizer(vocab_limit=VOCAB_LIMIT)
    term_freq   = freq_vec.fit_vectorize(desc_corpus)
    print(f'  Completed in: {time.time()-tick:.2f}s, Vocab size: {len(freq_vec.token2col):,}')

    print('Building TF-IDF weighted sparse tensor on GPU ...')
    tick = time.time()
    tfidf_transformer = TFIDFTransformer(freq_vec)
    keyword_index     = tfidf_transformer.fit_transform(term_freq, compute_device)
    print(f'  Completed in: {time.time()-tick:.2f}s')

    # 4. Dense Representation
    print(f'Parsing GloVe file...')
    pretrained_vecs = parse_glove_file(GLOVE_FILE)
    
    embed_layer, token_to_row, oov_total = build_embedding_layer(
        freq_vec.token2col, pretrained_vecs, VEC_DIM, compute_device
    )

    print('Encoding corpus via TF-IDF weighted GloVe pooling ...')
    tick = time.time()
    N_tickets    = len(tickets)
    semantic_idx = torch.zeros((N_tickets, VEC_DIM), device=compute_device)

    for i, desc in enumerate(desc_corpus):
        semantic_idx[i] = compute_sentence_vector(
            desc, token_to_row, freq_vec.token2col, 
            tfidf_transformer.idf_vector, embed_layer, 
            VEC_DIM, compute_device
        )
        if (i + 1) % 1000 == 0:
            print(f'  Encoded {i+1:,} / {N_tickets:,}')

    print(f'  Finished in {time.time()-tick:.1f}s')

    # 5. Save Artifacts
    print("Saving Models and Indices...")
    tickets['priority_code'] = priority_codes
    tickets['type_code']     = type_codes
    tickets.to_csv(os.path.join(MODELS_DIR, 'cleaned_ticket_corpus.csv'), index=False)
    
    torch.save(semantic_idx.cpu(), os.path.join(MODELS_DIR, 'semantic_index.pt'))
    
    token_registry = {'token_to_row': token_to_row,
                      'freq_vocab'  : freq_vec.token2col,
                      'col_to_token': freq_vec.col2token}
    with open(os.path.join(MODELS_DIR, 'token_registry.pkl'), 'wb') as fh:
        pickle.dump(token_registry, fh)

    np.save(os.path.join(MODELS_DIR, 'term_idf_weights.npy'),      tfidf_transformer.idf_vector)
    np.save(os.path.join(MODELS_DIR, 'tfidf_corpus_matrix.npy'),   term_freq * tfidf_transformer.idf_vector[np.newaxis, :])
    np.save(os.path.join(MODELS_DIR, 'channel_binary_matrix.npy'), channel_matrix)
    np.save(os.path.join(MODELS_DIR, 'priority_ordinal_codes.npy'), priority_codes)

    print("\n[✔] Pipeline build complete. All assets saved to models/")

if __name__ == '__main__':
    main()
