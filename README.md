<div align="center">
  <h1>🌌 Hybrid Semantic Retrieval & Intelligence System</h1>
  <p><strong>A Multi-Stage NLP Pipeline for Real-Time Customer Support Issue Retrieval</strong></p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gradio-Cyberpunk_UI-ff69b4?style=for-the-badge&logo=gradio" alt="Gradio">
  <img src="https://img.shields.io/badge/Numpy-Vectors-013243?style=for-the-badge&logo=numpy" alt="Numpy">
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Plotly-Analytics-3F4F75?style=for-the-badge&logo=plotly" alt="Plotly">
</div>

---

## 📖 Overview

This repository contains a highly optimized, modular Python Python implementation of a **multi-stage NLP search pipeline** designed for processing and efficiently retrieving customer support tickets.

Built entirely from scratch using **base PyTorch tensor operations** (explicitly avoiding high-level wrappers like Scikit-Learn), this engine integrates foundational statistical keyword scoring (TF-IDF) alongside deep neural semantic embeddings (GloVe 300d) to create a robust, hybrid search architecture. The frontend is powered by a beautiful, fully interactive **Gradio Dashboard** designed with a custom Cyberpunk UI.

---

## ✨ Core Features

- **Statistical Pipeline (Keyword Engine):** Custom implementations of N-gram algorithms, Frequency Vectorizers, and TF-IDF transformers optimized natively for sparse GPU tensors.
- **Semantic Pipeline (Neural Engine):** Custom GloVe embedding logic using TF-IDF weighted pooling arrays to capture deep textual context and semantic synonym overlap.
- **Hybrid Similarity Mapping:** Dynamic weight balancing (`beta`) using Cosine Similarity matrices to merge statistical occurrences natively with semantic space vectors.
- **Multi-Tab Dashboard System:**
  - **Terminal Search Node:** Fully functional interface to run user queries and control the semantic/keyword hybrid mixing weight (`beta` parameter).
  - **Exploratory Data Analysis:** Real-time data telemetry containing interactive Plotly visual graphics dynamically generated based on the dataset.
  - **Hardware Benchmarks:** Live dual-GPU/CPU compute telemetry tracking extraction times over batch queries.
  - **Evaluation Report:** Native visual presentation of Precision@5 benchmarking accuracy and qualitative side-by-side Retrieval comparisons mapping GloVe\'s superiority over TF-IDF.

---

## 📂 Project Structure

```text
HSRIS/
│
├── data/                      # Place dataset and word vectors here
├── models/                    # Serialized indices, graphs, and system telemetry cache
├── src/                       # Source Code Library
│   ├── config.py              # Path definitions & Device setup
│   ├── data_processing.py     # CSV parsing & Plotly generation
│   ├── encoders.py            # Custom Ordinal & Binary algorithms
│   ├── sparse_engine.py       # Regex tokenizers & TF-IDF matrices
│   ├── semantic_engine.py     # GloVe embeddings & Weighted Pooling
│   └── hybrid_search.py       # Cosine Similarity Engine
│
├── build_indices.py           # Training pipeline (Run this FIRST)
├── evaluate.py                # Metrics Generator
├── app.py                     # Gradio Local UI Server
├── requirements.txt           # Project Dependencies
└── README.md                  # Project Documentation
```

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/mahadbashir1/HSRIS-nlp-retrieval-engine.git
cd HSRIS-nlp-retrieval-engine
```

**2. Install Dependencies**
Ensure you have Python 3.10+ installed. Install the required libraries via the provided `requirements.txt` file setup:
```bash
pip install -r requirements.txt
```

**3. Prepare Data**
- Create a `data/` directory.
- Download the *Customer Support Tickets* CSV dataset from Kaggle.
- Download the `glove.6B.300d.txt` language model parameters.
- Place both files strictly inside the `data/` folder.

**4. Prepare Models Cache**
- The project will cache matrices efficiently. Make sure the `models/` directory natively exists in your project.

---

## 💻 Running the Application

### Step 1: Build the Indices
*This process precomputes your GPU tensors and saves them via Pickle to `./models/` to prevent real-time lag during evaluation interactions.*
```bash
python build_indices.py
```

### Step 2: Generate Telemetry and Evaluations
*Generate the automated Quantitative Metrics (`Precision@5` traces, Similarity charts, exploratory data analytics, etc.)*
```bash
python evaluate.py
```

### Step 3: Launch UI Dashboard
*Starts the local GRadio Cyberpunk UI on port 7860.*
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:7860`
