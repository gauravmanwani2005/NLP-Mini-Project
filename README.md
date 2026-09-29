# 🎓 EduMind: Intelligent Multilingual Academic Knowledge Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-BERT-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-00599C.svg?style=flat)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EduMind** is an end-to-end intelligent multilingual academic knowledge assistant designed for college NLP coursework and real-world student advisory services. It resolves the lexical gap between formal administrative policies and conversational student inquiries by combining **Classical NLP Preprocessing**, a **Fine-Tuned BERT Intent Classifier**, **FAISS Dense Semantic Retrieval**, **Retrieval-Augmented Generation (RAG)**, **Multi-Section Summarization**, and **Multilingual Query Routing (English, Hindi, Marathi)**.

---

## 📋 Table of Contents
- [Empirical Highlights](#-empirical-highlights)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Setting Up on Another Device (Quick Start)](#-setting-up-on-another-device-quick-start)
  - [Prerequisites](#1-prerequisites)
  - [Step 1: Clone Repository](#2-step-1-clone-repository)
  - [Step 2: Create and Activate Virtual Environment](#3-step-2-create-and-activate-virtual-environment)
  - [Step 3: Install Dependencies](#4-step-3-install-dependencies)
  - [Step 4: The `model.safetensors` Large File Setup](#5-step-4-the-modelsafetensors-large-file-setup)
  - [Step 5: Launch the Web Application](#6-step-5-launch-the-web-application)
- [Handling `model.safetensors` (Why it is in `.gitignore`)](#-handling-modelsafetensors-why-it-is-in-gitignore)
- [Running Individual NLP Modules](#-running-individual-nlp-modules)
- [NLP Preprocessing Pipeline](#-nlp-preprocessing-pipeline)
- [Benchmark Results](#-benchmark-results)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [Authors & License](#-authors--license)

---

## 🏆 Empirical Highlights

| Evaluation Metric | Component | Result | Baseline Benchmark |
|---|---|---|---|
| **Intent Classification Accuracy** | Fine-Tuned BERT (`bert-base-uncased`) | **98.76%** | Baseline TF-IDF: 98.35% / Pretrained Zero-shot: 8.50% |
| **Macro F1 Score** | Fine-Tuned BERT (12 Intent Classes) | **98.65%** | Baseline TF-IDF: 98.39% |
| **Dense Retrieval (MRR)** | FAISS (`IndexFlatIP` + MiniLM) | **0.8750** | Precision@1: 83.33% / Precision@3: 72.22% |
| **Generation BERTScore Recall** | Grounded QA Synthesis | **0.7705** | Mean F1: 0.5754 (vs ground truth) |
| **Summarization ROUGE-1 F1** | Multi-Section Circular Summarizer | **0.2775** | ROUGE-L: 0.2545 |

---

## 🏗️ System Architecture

```text
Student Query (English / Hindi / Marathi)
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ 1. Multilingual Routing & Query Normalization          │
│    (Script Detection + Cross-lingual Lexicon Mapping)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Mandatory NLP Preprocessing                         │
│    (Sentence Split, Word Tokens, WordNet Lemmatization,│
│     Penn POS Tagging, WordNet Semantic Distance)       │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌──────────────────────────┐┌────────────────────────────┐
│ 3. BERT Intent Classifier││ 4. FAISS Semantic Search   │
│    (Fine-Tuned BERT-base ││    (Sentence-Transformers  │
│     or TF-IDF fallback)  ││     all-MiniLM-L6-v2)      │
└────────────┬─────────────┘└────────────┬───────────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Context-Grounded Answer Generation & Synthesis      │
│    (Anti-Hallucination Guardrails + Source Citations)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                 Grounded Response Output
```

---

## 📂 Repository Structure

```text
edumind/
├── app/
│   └── app.py                               # Interactive Streamlit Web Application
├── classification/
│   ├── baseline_tfidf.py                    # Classical ML Baseline (TF-IDF + Logistic Regression)
│   ├── train_bert.py                        # BERT fine-tuning script with PyTorch & Hugging Face
│   ├── evaluate_bert.py                     # Inference and test evaluation module
│   └── model/
│       ├── baseline_tfidf.joblib            # Tracked lightweight TF-IDF model (~290 KB)
│       └── bert_finetuned/                  # Best fine-tuned BERT checkpoint directory
│           ├── config.json                  # Model architecture config (Tracked in Git)
│           ├── label_map.json               # 12 Intent labels mapping (Tracked in Git)
│           ├── tokenizer.json               # Tokenizer vocabulary (Tracked in Git)
│           ├── tokenizer_config.json        # Tokenizer config (Tracked in Git)
│           └── model.safetensors            # ⚠️ BERT model weights (Ignored in Git, see below)
├── data/
│   ├── raw/
│   │   ├── college_queries.csv              # 1,608 labeled queries across 12 academic intents
│   │   └── documents/                       # 6 official institutional policy documents (PDF/TXT)
│   └── processed/
│       ├── train.csv                        # Stratified training split (1,125 rows, 70%)
│       ├── validation.csv                   # Validation split (241 rows, 15%)
│       └── test.csv                         # Unseen test split (242 rows, 15%)
├── preprocessing/
│   ├── sentence_segmentation.py             # NLTK Punkt Sentence Boundary Detection
│   ├── tokenization.py                      # Word Tokenization & Type-Token Ratio (TTR)
│   ├── lemmatization.py                     # WordNet POS Lemmatization vs Porter Stemming
│   ├── pos_tagging.py                       # Penn Treebank POS Syntactic Tagging
│   └── wordnet_analysis.py                  # Synsets, Hypernyms & Semantic Similarity
├── retrieval/
│   ├── document_loader.py                   # Loader for PDF, TXT, and Markdown files
│   ├── chunking.py                          # Sentence-aware chunker with sliding overlap
│   ├── embeddings.py                        # Dense embeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py                      # FAISS IndexFlatIP Cosine Vector Store
│   ├── retriever.py                         # Dense, Sparse (TF-IDF), and Hybrid RRF Retriever
│   └── index/                               # Persisted FAISS vector index & metadata
├── rag/
│   ├── prompt.py                            # Context prompts with anti-hallucination guardrails
│   ├── generator.py                         # Answer synthesis engine with citations
│   └── qa_pipeline.py                       # Unified End-to-End Question Answering Pipeline
├── summarization/
│   └── summarizer.py                        # Key-section document summarizer
├── multilingual/
│   ├── language_detection.py                # Fast script & n-gram language detector (En, Hi, Mr)
│   └── translation.py                       # Academic lexicon dictionary & query normalization
├── evaluation/
│   ├── classification_metrics.py            # Aggregates evaluation results across all models
│   ├── retrieval_metrics.py                 # Benchmarks Precision@K, Recall@K, and MRR
│   ├── rouge_evaluation.py                  # Evaluates ROUGE-1, ROUGE-2, and ROUGE-L
│   ├── bertscore_evaluation.py              # Evaluates contextual embedding generation quality
│   └── results/                             # Saved JSON metrics and Confusion Matrix heatmap
├── notebooks/                               # Jupyter exploratory notebooks (01 to 06)
├── report/                                  # Full academic project report (PDF & Markdown)
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore rules (includes model.safetensors)
└── README.md                                # Project documentation
```

---

## 💻 Setting Up on Another Device (Quick Start)

Follow these steps to run EduMind on any fresh machine (**macOS, Windows, or Linux**).

### 1. Prerequisites
- **Python**: Version `3.10`, `3.11`, `3.12`, or `3.13` installed. Verify with `python3 --version` or `python --version`.
- **Git**: Installed on your system.

### 2. Step 1: Clone Repository
```bash
git clone https://github.com/gauravmanwani2005/NLP-Mini-Project.git
cd NLP-Mini-Project
```
*(Or navigate into your cloned directory if named `edumind`)*

### 3. Step 2: Create and Activate Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Step 3: Install Dependencies
Upgrade pip and install all required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch (Windows / Linux with NVIDIA GPU):**  
> If you wish to use GPU acceleration on Windows/Linux with CUDA, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> On Apple Silicon (M1/M2/M3/M4), standard `pip install torch` automatically utilizes **Metal Performance Shaders (MPS)**.

### 5. Step 4: The `model.safetensors` Large File Setup
Because `model.safetensors` (~438 MB) is gitignored, you have two quick options:

#### 👉 Option A: Re-train the model locally in 2–3 minutes (Recommended!)
All training datasets (`train.csv`, `val.csv`) are included in `data/processed/`. Simply run:
```bash
python classification/train_bert.py
```
This fine-tunes `bert-base-uncased` for 3 epochs using your GPU (CUDA), Apple MPS, or CPU, and directly outputs `model.safetensors` to `classification/model/bert_finetuned/`.

#### 👉 Option B: Copy an existing `model.safetensors` file
If you already have `model.safetensors` on your original machine, flash drive, or cloud storage:
- Place `model.safetensors` directly into:
  ```text
  classification/model/bert_finetuned/model.safetensors
  ```

#### 👉 Option C: Run without `model.safetensors` (Automatic Fallback)
If you don't run Step 4 right away, **the application will still work!** EduMind features an automatic fallback that switches to the tracked classical ML model (`classification/model/baseline_tfidf.joblib`) with 98.35% accuracy.

### 6. Step 5: Launch the Web Application
Launch the interactive Streamlit user interface:
```bash
streamlit run app/app.py
```
Open your web browser at:
```text
http://localhost:8501
```

---

## 📦 Handling `model.safetensors` (Why it is in `.gitignore`)

### Why was `model.safetensors` gitignored?
- **GitHub's 100 MB Limit**: GitHub rejects pushes containing any single file larger than 100 MB. The fine-tuned BERT model weights (`model.safetensors`) are approximately **438 MB**, which exceeds standard Git limits.
- **Git Best Practice**: Storing large binary weights directly in Git repositories bloats repository clone sizes and slows down version control.

### What is tracked in Git for the model?
The directory `classification/model/bert_finetuned/` retains all metadata and configurations necessary to work seamlessly:
- `config.json` (BERT model configuration & class count)
- `label_map.json` (Mapping of all 12 academic intent classes)
- `tokenizer.json` and `tokenizer_config.json` (Full BERT WordPiece tokenizer)
- `baseline_tfidf.joblib` (Full classical baseline model)

### How to obtain or generate `model.safetensors`:

| Method | Command / Action | Time Required | Description |
|---|---|---|---|
| **1. Retrain Locally (Best)** | `python classification/train_bert.py` | 2–5 min | Trains from included splits; saves `model.safetensors` automatically. |
| **2. Manual File Placement** | Copy to `classification/model/bert_finetuned/` | 10 sec | Place file downloaded from your drive/storage. |
| **3. Automated TF-IDF Fallback** | *None (Built-in)* | Instant | If absent, app automatically falls back to `baseline_tfidf.joblib`. |

---

## 🧪 Running Individual NLP Modules

You can execute and test individual components directly from your command line:

### 1. Classical NLP Preprocessing Modules
```bash
# Sentence Boundary Detection (Punkt)
python preprocessing/sentence_segmentation.py

# Word Tokenization & Type-Token Ratio (TTR)
python preprocessing/tokenization.py

# Lemmatization vs Stemming comparison
python preprocessing/lemmatization.py

# Syntactic POS Tagging (Penn Treebank)
python preprocessing/pos_tagging.py

# WordNet Semantic Distance & Synset Analysis
python preprocessing/wordnet_analysis.py
```

### 2. Intent Classification & Baselines
```bash
# Train/Evaluate classical TF-IDF + Logistic Regression baseline
python classification/baseline_tfidf.py

# Fine-tune BERT on domain queries (creates model.safetensors)
python classification/train_bert.py

# Evaluate Fine-Tuned BERT on test split & generate confusion matrix
python classification/evaluate_bert.py
```

### 3. Document Retrieval & Vector Store
```bash
# Build / rebuild FAISS index from documents in data/raw/documents/
python -c "from retrieval.retriever import EduMindRetriever; r = EduMindRetriever(); print('FAISS Index Ready!')"
```

### 4. Comprehensive Evaluation Benchmarks
```bash
# Run Retrieval Benchmarks (Precision@1, Precision@3, MRR)
python evaluation/retrieval_metrics.py

# Run ROUGE evaluation on Document Summaries
python evaluation/rouge_evaluation.py

# Run BERTScore semantic evaluation on Grounded QA synthesis
python evaluation/bertscore_evaluation.py
```

---

## 🎯 NLP Preprocessing Pipeline

EduMind satisfies all foundational academic NLP requirements with modular implementations:

1. **Sentence Boundary Identification (`preprocessing/sentence_segmentation.py`)**:
   - Uses NLTK's Punkt unsupervised boundary detector to cleanly segment complex multi-clause academic regulations.
2. **Tokenization & Lexical Analysis (`preprocessing/tokenization.py`)**:
   - Computes query and document Type-Token Ratio (TTR) to assess lexical richness and vocabulary diversity.
3. **Lemmatization vs. Stemming (`preprocessing/lemmatization.py`)**:
   - Contrasts aggressive Porter Stemming against context-aware POS-guided WordNet Lemmatization (e.g., `studying` -> `study` vs `studi`).
4. **Syntactic POS Tagging (`preprocessing/pos_tagging.py`)**:
   - Assigns Penn Treebank tags and identifies grammatical constructs (Modals like `can`/`must`, WH-interrogatives, and Subject Nouns).
5. **Semantic Analysis via WordNet (`preprocessing/wordnet_analysis.py`)**:
   - Extracts taxonomic hypernyms, definitions, and calculates Wu-Palmer / Path Similarity distances between concepts.

---

## 📊 Benchmark Results

### 1. Intent Classification Performance (12 Classes)
Tested on the unseen test split (242 samples):

| Model | Architecture | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| **Classical Baseline** | TF-IDF (1-2 ngrams) + Logistic Regression | 98.35% | 98.39% | 98.39% |
| **Pretrained BERT** | `bert-base-uncased` (Zero-shot / Untrained head) | 8.50% | 8.08% | 8.08% |
| **Fine-Tuned BERT** | `bert-base-uncased` (EduMind 3 Epochs) | **98.76%** | **98.76%** | **98.65%** |

### 2. Information Retrieval Metrics (FAISS Dense Vector Store)
- **Mean Reciprocal Rank (MRR)**: `0.8750`
- **Precision@1**: `83.33%`
- **Precision@3**: `72.22%`
- **Recall@3**: `85.00%`

### 3. Generation & Summarization Metrics
- **QA BERTScore Recall**: `0.7705` (Mean generation F1: `0.5754` vs reference ground truth)
- **Summarization ROUGE-1 F1**: `0.2775`
- **Summarization ROUGE-L F1**: `0.2545`

---

## 🛠️ Troubleshooting & FAQ

### Q1: `FileNotFoundError: Fine-tuned BERT not found at classification/model/bert_finetuned`
- **Cause**: `model.safetensors` was omitted due to `.gitignore`.
- **Solution**: Run `python classification/train_bert.py` to generate it in under 3 minutes, or rely on the automatic TF-IDF fallback.

### Q2: Missing NLTK resources (`punkt`, `wordnet`, etc.)
- All preprocessing scripts automatically download their required NLTK corpora silently. If you are behind an offline proxy or firewall, manually run:
  ```python
  python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords'])"
  ```

### Q3: `ModuleNotFoundError: No module named 'preprocessing'`
- Make sure you are executing commands from the root `edumind` / `NLP-Mini-Project` directory:
  ```bash
  cd NLP-Mini-Project
  python preprocessing/tokenization.py
  ```

### Q4: Streamlit port already in use
- Specify a custom port when launching the application:
  ```bash
  streamlit run app/app.py --server.port 8502
  ```

---

## 👨‍💻 Authors & Academic Citation

Developed for the **Natural Language Processing (NLP) Mini-Project Curriculum**.

If you use or reference EduMind in your coursework or academic research, please cite:
```bibtex
@misc{edumind2025,
  title={EduMind: Intelligent Multilingual Academic Knowledge Assistant},
  author={Gaurav Manwani},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/gauravmanwani2005/NLP-Mini-Project}}
}
```
