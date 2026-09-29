# EduMind: Intelligent Multilingual Academic Knowledge Assistant

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-BERT-yellow.svg)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

EduMind is an intelligent multilingual academic knowledge assistant designed for college NLP mini-project requirements. It bridges the lexical gap between formal institutional circulars and student conversational queries by combining **Mandatory Classical NLP Preprocessing**, a **Fine-Tuned BERT Intent Classifier**, **FAISS Dense Semantic Retrieval**, **Retrieval-Augmented Generation (RAG)**, **Automatic Document Summarization**, and **Multilingual Routing (Hindi & Marathi)**.

---

## 🏆 Empirical Highlights

| Evaluation Metric | Component | Result | Benchmark Baseline |
|---|---|---|---|
| **Intent Classification Accuracy** | Fine-Tuned BERT (`bert-base-uncased`) | **98.76%** | Baseline TF-IDF: 98.35% / Pretrained Zero-shot: 8.50% |
| **Macro F1 Score** | Fine-Tuned BERT (12 Intent Classes) | **98.65%** | Baseline TF-IDF: 98.39% |
| **Mean Reciprocal Rank (MRR)** | Dense Retrieval (FAISS + MiniLM) | **0.8750** | Precision@1: 83.33% / Precision@3: 72.22% |
| **Generation BERTScore Recall** | Grounded QA Synthesis | **0.7705** | Mean F1: 0.5754 (vs ground truth) |
| **Summarization ROUGE-1 F1** | Multi-Section Circular Summarizer | **0.2775** | ROUGE-L: 0.2545 |

---

## 📂 Project Repository Structure

```
edumind/
├── data/
│   ├── raw/
│   │   ├── college_queries.csv              # 1,608 labeled domain-specific queries across 12 intents
│   │   └── documents/                       # 6 official institutional policy documents (TXT/PDF)
│   └── processed/
│       ├── train.csv                        # Stratified training split (1,125 rows, 70%)
│       ├── validation.csv                   # Validation split (241 rows, 15%)
│       └── test.csv                         # Unseen test split (242 rows, 15%)
├── preprocessing/
│   ├── sentence_segmentation.py             # 1. Sentence Boundary Detection (NLTK Punkt)
│   ├── tokenization.py                      # 2. Word Tokenization & Lexical Diversity (TTR)
│   ├── lemmatization.py                     # 3. WordNet Lemmatization vs Porter Stemming
│   ├── pos_tagging.py                       # 4. Syntactic Analysis (Penn Treebank POS Tags)
│   └── wordnet_analysis.py                  # 5. Semantic Analysis (Synsets, Hypernyms, Similarity)
├── classification/
│   ├── baseline_tfidf.py                    # Classical ML Baseline (TF-IDF + Logistic Regression)
│   ├── train_bert.py                        # Fine-tuning script for bert-base-uncased on Apple MPS/GPU
│   ├── evaluate_bert.py                     # Evaluation script generating metrics & Confusion Matrix
│   └── model/
│       ├── baseline_tfidf.joblib            # Serialized TF-IDF baseline model
│       └── bert_finetuned/                  # Best fine-tuned BERT checkpoint & tokenizer
├── retrieval/
│   ├── document_loader.py                   # Loader for PDF, TXT, and Markdown files
│   ├── chunking.py                          # Sentence-aware recursive chunker with overlap
│   ├── embeddings.py                        # Dense embeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py                      # FAISS IndexFlatIP (Cosine Similarity) persistence
│   └── retriever.py                         # Dense, Sparse (TF-IDF), and Hybrid RRF Retriever
├── rag/
│   ├── prompt.py                            # Context-grounded prompts with anti-hallucination guardrails
│   ├── generator.py                         # Answer synthesis engine with citations
│   └── qa_pipeline.py                       # End-to-end question answering pipeline
├── summarization/
│   └── summarizer.py                        # Multi-section document summarizer (Dates, Eligibility, Fees)
├── multilingual/
│   ├── language_detection.py                # Script and n-gram language detection (En, Hi, Mr)
│   └── translation.py                       # Cross-lingual lexicon mapping and response formatting
├── evaluation/
│   ├── classification_metrics.py            # Aggregates evaluation results across all models
│   ├── retrieval_metrics.py                 # Benchmarks Precision@K, Recall@K, and MRR
│   ├── rouge_evaluation.py                  # Evaluates ROUGE-1, ROUGE-2, and ROUGE-L
│   ├── bertscore_evaluation.py              # Evaluates contextual embedding generation quality
│   └── results/                             # Saved JSON metrics and Confusion Matrix heatmap
├── app/
│   └── app.py                               # Full-featured Streamlit interactive web application
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_nlp_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_bert_finetuning.ipynb
│   ├── 05_semantic_search.ipynb
│   └── 06_rag_evaluation.ipynb
├── report/
│   ├── NLP_Mini_Project_Report.md           # Complete 34-section academic report
│   ├── NLP_Mini_Project_Report.pdf          # Formatted PDF publication
│   └── generate_pdf.py                      # PDF compilation script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone and enter directory
cd edumind

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Individual NLP Modules from Terminal
```bash
# Test mandatory NLP preprocessing
python preprocessing/sentence_segmentation.py
python preprocessing/tokenization.py
python preprocessing/lemmatization.py
python preprocessing/pos_tagging.py
python preprocessing/wordnet_analysis.py

# Run Intent Prediction with Fine-Tuned BERT
python classification/evaluate_bert.py

# Run Retrieval & Evaluation Benchmark
python evaluation/retrieval_metrics.py
python evaluation/rouge_evaluation.py
python evaluation/bertscore_evaluation.py
```

---

## 🎯 Mandatory NLP Preprocessing Breakdown

1. **Sentence Boundary Identification (`preprocessing/sentence_segmentation.py`):**
   - Identifies boundaries in complex institutional circulars using NLTK's unsupervised Punkt tokenizer.
2. **Tokenization & Lexical Analysis (`preprocessing/tokenization.py`):**
   - Tokenizes queries and documents, calculating the Type-Token Ratio (lexical diversity).
3. **Lemmatization vs Stemming (`preprocessing/lemmatization.py`):**
   - Implements POS-guided WordNet lemmatization. Demonstrates that stemming produces truncated roots (e.g. `studying` -> `studi`), whereas lemmatization resolves genuine canonical forms (e.g. `corpora` -> `corpus`).
4. **Syntactic POS Tagging (`preprocessing/pos_tagging.py`):**
   - Assigns Penn Treebank tags and categorizes grammatical roles (modals, nouns, WH-question words).
5. **Semantic Analysis via WordNet (`preprocessing/wordnet_analysis.py`):**
   - Extracts synsets, definitions, hypernyms (taxonomic parents), and calculates Path & Wu-Palmer semantic distance.

---

## 📊 Model Comparison Benchmark (Section 26)

| Model | Architecture | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| **Baseline ML** | TF-IDF + Logistic Regression | 98.35% | 98.39% | 98.39% |
| **Pretrained BERT** | `bert-base-uncased` (Zero-Shot) | 8.50% | 8.08% | 8.08% |
| **Fine-Tuned BERT** | `bert-base-uncased` (EduMind) | **98.76%** | **98.76%** | **98.65%** |

---

## 🎓 Viva Voce Key Talking Points

1. **Why is BERT fine-tuning superior to traditional TF-IDF?**
   - TF-IDF treats words independently (bag-of-words) and struggles with unseen synonyms and complex syntactic structure. BERT's bidirectional self-attention mechanisms capture deep contextual semantics across word positions.
2. **What role does BERTScore serve in this project?**
   - BERTScore is applied to evaluate generated answers and summaries against ground truth reference text. Rather than penalizing synonymous words (as BLEU/ROUGE do), BERTScore uses contextual embeddings to evaluate semantic similarity.
3. **How does EduMind prevent LLM hallucinations?**
   - EduMind enforces a similarity cutoff threshold (< 0.35). If retrieved passages do not support the query, the system returns an explicit fallback rather than inventing speculative academic policies.
4. **How does FAISS achieve fast semantic retrieval?**
   - Dense embeddings are L2-normalized upon creation. FAISS performs inner product search (`IndexFlatIP`), which is mathematically equivalent to cosine similarity but computationally optimized via SIMD vector operations.

---

## 📜 Authors & Academic Citation
**EduMind** was developed for the College NLP Mini-Project Curriculum adhering to all guidelines for Natural Language Processing, Transformer Fine-Tuning, Information Retrieval, and RAG architectures.
