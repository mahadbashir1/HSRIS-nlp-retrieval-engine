# Hybrid Semantic Retrieval & Intelligence System

## Overview
This repository contains a modular Python implementation of a **multi-stage NLP pipeline** designed for processing and efficiently retrieving customer support tickets. 

Built entirely from scratch using **base PyTorch integrations** (explicitly avoiding high-level wrappers like Scikit-Learn), this engine integrates foundational statistical keyword scoring (TF-IDF) alongside deep neural semantic embeddings (GloVe 300d) to create a robust, hybrid search architecture. The frontend is powered by a beautiful, fully interactive **Gradio Dashboard** designed with a custom Cyberpunk UI.

## Features
- **Statistical Pipeline (Keyword Engine):** Custom implementations of N-gram algorithms, Frequency Vectorizers, and TF-IDF transformers optimized for sparse GPU tensors.
- **Semantic Pipeline (Neural Engine):** Custom GloVe embedding logic using TF-IDF weighted pooling arrays to capture deep textual context and semantic synonym overlap.
- **Multi-Tab Dashboard:**
  - **Terminal Search Node:** Fully functional interface to run user queries and control the semantic/keyword hybrid mixing weight (Beta).
  - **Exploratory Data Analysis:** Real-time data telemetry containing interactive Plotly visual graphics (Pie charts, Bar charts, Histograms) dynamically generated based on the currently loaded `.csv` file.
  - **Hardware Benchmarks:** Live dual-GPU compute telemetry tracking extraction times over batch queries.
  - **Evaluation Report:** Native visual presentation of Precision@5 benchmarking accuracy and qualitative side-by-side Retrieval comparisons mapping GloVe's superiority over TF-IDF.

## Directory Structure
```
HSRIS/
│
├── src/
│   ├── config.py              # Path definitions & Device setup
│   ├── data_processing.py     # CSV parsing & Plotly generation
│   ├── encoders.py            # Custom Ordinal & Binary algorithms
│   ├── sparse_engine.py       # Regex tokenizers & TF-IDF matrices
│   ├── semantic_engine.py     # GloVe embeddings & Weighted Pooling
│   └── hybrid_search.py       # Cosine Similarity Engine
│
├── build_indices.py           # Training pipeline (Run this FIRST)
├── evaluate.py                # Dual-GPU Metrics Generator
├── app.py                     # Gradio Local UI Server
└── README.md
```

## Setup Instructions
1. Clone this repository locally or to your cloud instance.
2. Ensure you have Python 3.10+ installed. Install the dependencies:
   ```bash
   pip install torch numpy pandas gradio plotly
   ```
3. Create a `data/` directory in the root of the project.
4. Download the `customer_support_tickets.csv` Kaggle Dataset and the `glove.6B.300d.txt` language model parameters, and place them inside the `data/` directory.
5. Create an empty `models/` directory in the root of the project.

### Running the Project
**Step 1:** Build the Neural and Sparse Indices. *This precomputes your GPU tensors and saves them via Pickle to `./models/` to prevent real-time lag during evaluation interactions.*
```bash
python build_indices.py
```
**Step 2:** Generate the automated Quantitative Metrics (`Precision@5` traces, Similarity charts, etc.)
```bash
python evaluate.py
```
**Step 3:** Launch the Dashboard locally!
```bash
python app.py
```
The application will launch on `http://127.0.0.1:7860`.
