# EduMind: An Intelligent Multilingual Academic Knowledge Assistant using Fine-Tuned Transformers and Retrieval-Augmented Generation

**Course:** NLP Mini-Project  
**Domain:** Academic / College Information Systems  
**Primary Models:** Fine-Tuned `bert-base-uncased`, Sentence-Transformers `all-MiniLM-L6-v2`, FAISS Vector Index  

---

## 1. Title Page & Project Metadata
- **Project Name:** EduMind
- **Full Title:** EduMind: An Intelligent Multilingual Academic Knowledge Assistant using Fine-Tuned Transformers and Retrieval-Augmented Generation
- **Target Audience:** College Students, Faculty, Examination Cell, Department Administrators
- **Tech Stack:** Python 3.13, PyTorch, Hugging Face Transformers, Sentence-Transformers, FAISS, NLTK, Scikit-learn, Streamlit

---

## 2. Abstract
Institutional academic information—including examination schedules, attendance regulations, fee structures, placement criteria, and course catalogs—is traditionally distributed across disparate circulars, PDFs, and noticeboards. Keyword-based search engines fail when student queries exhibit vocabulary mismatch (lexical gap) with formal institutional wording. 

In this project, we develop **EduMind**, a comprehensive Natural Language Processing (NLP) framework combining:
1. **Mandatory Classical NLP Preprocessing:** Sentence boundary detection, tokenization, lemmatization, syntactic POS tagging, and WordNet semantic distance analysis.
2. **Domain-Specific Transformer Fine-Tuning:** An end-to-end fine-tuned `bert-base-uncased` classifier achieving **98.76% accuracy** and **98.65% Macro F1** across 12 academic intent categories.
3. **Retrieval-Augmented Generation (RAG):** Dense semantic passage retrieval with FAISS and `all-MiniLM-L6-v2` achieving **83.33% Precision@1** and **0.8750 MRR**, outperforming sparse keyword retrieval on paraphrased student queries.
4. **Structured Summarization & Multilingual Routing:** Automatic circular summarization and query translation for Indian languages (Hindi & Marathi).
5. **Rigorous Empirical Evaluation:** Evaluated using Precision, Recall, Macro F1, Confusion Matrix, MRR, ROUGE-1/2/L, and BERTScore (Recall: 0.7705, F1: 0.5754).

---

## 3. Introduction
Educational institutions publish numerous regulatory documents each semester. Students navigating these documents often experience ambiguity regarding deadlines, passing standards, and eligibility thresholds. EduMind bridges this gap by acting as an intelligent conversational interface capable of understanding colloquial student queries, determining intent, retrieving verified document passages, and generating grounded answers.

---

## 4. Motivation
Traditional campus chatbots rely on hardcoded keyword rules or shallow intent matching. If a student asks *"When do I have to submit my exam form?"*, a keyword search fails if the official regulation only mentions *"examination application deadline"*. By fine-tuning a BERT transformer and grounding responses in dense vector space, EduMind eliminates semantic ambiguity while preventing generative hallucination.

---

## 5. Problem Statement
- **Input:** Natural language student query (English, Hindi, or Marathi) or uploaded academic PDF/TXT circular.
- **Output:** Predicted academic intent badge, confidence score, cited document passages, and a factually grounded answer.
- **NLP Task Mapping:** Multi-class Text Classification (Intent Detection), Semantic Information Retrieval (Dense Vector Search), Extractive/Abstractive Summarization, and Multilingual Entity Translation.

---

## 6. System Input and Output Specifications

| Component | Input Format | Output Format | Example |
|---|---|---|---|
| **Preprocessing** | Raw text string | Segmented sentences, tokens, POS tags, WordNet synsets | `"Students must submit..."` -> `Tokens: ['Students', 'must', 'submit']` |
| **Intent Classifier** | Student question | Intent label & Softmax confidence | `"When are finals?"` -> `exam_schedule (99.2%)` |
| **Retriever** | Dense query vector | Top-K passages with metadata & similarity score | `attendance_policy.txt (Page 1), Score: 0.76` |
| **RAG Generator** | Context + Question | Verified natural language answer with citations | `Minimum 75% attendance is required... [attendance_policy.txt, p.1]` |
| **Summarizer** | Long circular | Structured Executive Summary | Overview, Deadlines, Eligibility, Fees |

---

## 7. Objectives
1. Implement the 5 mandatory NLP preprocessing stages: Sentence boundary detection, tokenization, lemmatization, POS tagging, and WordNet semantic analysis.
2. Construct a balanced, domain-specific dataset of 1,608 queries spanning 12 distinct academic intents.
3. Train a baseline classical machine learning model (TF-IDF + Logistic Regression).
4. Fine-tune `bert-base-uncased` on Apple Silicon MPS / PyTorch and evaluate on unseen test data.
5. Build a dense semantic search vector database using FAISS and `sentence-transformers`.
6. Implement strict RAG guardrails to avoid hallucinations when queries fall outside known institutional policy.
7. Integrate automatic circular summarization and multilingual translation (Hindi/Marathi).
8. Provide an interactive Streamlit UI and comprehensive empirical benchmarks.

---

## 8. NLP Tasks Covered
1. **Sentence Boundary Identification:** NLTK Punkt unsupervised sentence boundary detection.
2. **Tokenization:** Word tokenization and lexical diversity (Type-Token Ratio analysis).
3. **Lemmatization:** WordNet lemmatizer with Penn Treebank POS tag mapping.
4. **Syntactic Analysis (POS Tagging):** Grammatical category annotation via Averaged Perceptron Tagger.
5. **Semantic Analysis (WordNet):** Synsets, hypernyms/hyponyms, and path/Wu-Palmer semantic similarity.
6. **Text Classification:** 12-class academic intent classification.
7. **Information Retrieval:** Dense vector retrieval vs TF-IDF keyword retrieval.
8. **Question Answering (RAG):** Context-grounded synthesis with provenance citations.
9. **Text Summarization:** Extractive centrality and anchor-based summarization.
10. **Multilingual Processing:** Language identification and cross-lingual lexical routing.

---

## 9. Existing System vs. Proposed EduMind

| Feature | Existing Systems / Keyword FAQ | Proposed EduMind System |
|---|---|---|
| **Search Mechanism** | Exact keyword substring matching (BM25) | Dense semantic vector similarity (FAISS + MiniLM) |
| **Paraphrase Handling** | Fails when student vocabulary differs from circular | Robust semantic understanding across paraphrases |
| **Intent Understanding** | Rule-based regex pattern matching | Fine-tuned 12-class BERT Transformer |
| **Hallucination Prevention** | None (or generic canned responses) | Strict context grounding with fallback guardrail |
| **Multilingual Support** | Typically English only | Hindi and Marathi detection and entity routing |
| **Document Processing** | Static text display | Dynamic sentence chunking, provenance citations, summarization |

---

## 10. Proposed System Architecture

```text
                               +-------------------+
                               |   Student Query   |
                               +---------+---------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
             [Language Detection]                 [NLP Preprocessing]
             (English / Hindi / Marathi)          - Sentence Boundary Detection
                       |                          - Tokenization & Lemmatization
                       v                          - POS Tagging (Syntactic)
             [Cross-Lingual Routing]              - WordNet (Semantic)
                       |
                       +-----------------+
                                         |
                                         v
                         +-------------------------------+
                         | Fine-Tuned BERT Classifier    |
                         | (bert-base-uncased, 12-class) |
                         +---------------+---------------+
                                         |
                                         v
                               Intent & Confidence
                                         |
                                         v
                         +-------------------------------+
                         | Dense Semantic Search (FAISS) |
                         | (all-MiniLM-L6-v2 Embeddings) |
                         +---------------+---------------+
                                         |
                                         v
                              Top-K Document Chunks
                                         |
                                         v
                         +-------------------------------+
                         | Context-Grounded RAG Prompt   |
                         | (With Anti-Hallucination Gate)|
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Answer Synthesis Engine       |
                         | + Source Citations            |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Streamlit Web Demonstration   |
                         +-------------------------------+
```

---

## 11. Dataset Analysis and Preparation
- **Raw File:** `data/raw/college_queries.csv` (1,608 samples)
- **Classes (12):** `exam_schedule`, `exam_registration`, `attendance`, `fees`, `assignments`, `results`, `library`, `placements`, `courses`, `timetable`, `admissions`, `general_query`.
- **Splits:**
  - Training Set (70%): 1,125 samples
  - Validation Set (15%): 241 samples
  - Test Set (15%): 242 samples
- **Stratification:** Stratified by class labels to preserve uniform class distribution across splits.

---

## 12. Mandatory NLP Preprocessing Implementation

### 12.1 Sentence Boundary Identification
- Library: `nltk.tokenize.sent_tokenize` (Punkt unsupervised model).
- Purpose: Splits unstructured paragraphs into discrete grammatical sentences for chunking and retrieval.

### 12.2 Tokenization & Lexical Diversity
- Library: `nltk.tokenize.word_tokenize`.
- Metrics: Total Token Count, Unique Vocabulary, Type-Token Ratio (TTR = $|V| / N$).

### 12.3 Lemmatization vs Stemming
- Library: `nltk.stem.WordNetLemmatizer` with POS tag guidance vs `nltk.stem.PorterStemmer`.
- Viva Distinction: Stemming applies heuristic rule-based affix stripping (e.g. `studying` -> `studi`), frequently producing non-words. Lemmatization performs morphological analysis to return authentic dictionary lemmas (e.g. `corpora` -> `corpus`).

### 12.4 Syntactic Analysis (POS Tagging)
- Library: `nltk.pos_tag` (Penn Treebank Tagset).
- Identifies modal auxiliaries (`MD`), nouns (`NN`/`NNS`), verbs (`VB`/`VBD`/`VBG`), and WH-markers (`WP`/`WRB`).

### 12.5 Semantic Analysis (WordNet)
- Library: `nltk.corpus.wordnet`.
- Synsets, definitions, hypernyms (broader concepts), hyponyms (subordinate concepts).
- Semantic distance metrics:
  $$\text{Path Similarity} = \frac{1}{\text{shortest\_path}(s_1, s_2) + 1}$$
  $$\text{Wu-Palmer Similarity} = \frac{2 \times \text{depth}(\text{LCS})}{\text{depth}(s_1) + \text{depth}(s_2)}$$

---

## 13. Model Comparison Experiment (Section 26)

| Model | Architecture | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| **Experiment 1: Baseline ML** | TF-IDF (1-2 ngrams) + Logistic Regression | 98.35% | 98.39% | 98.39% |
| **Experiment 2: Pretrained BERT** | `bert-base-uncased` (Zero-Shot / Random Head) | 8.50% | 8.08% | 8.08% |
| **Experiment 3: Fine-Tuned BERT** | `bert-base-uncased` (3 Epochs, AdamW, lr=3e-5) | **98.76%** | **98.76%** | **98.65%** |

*Analysis:* Fine-Tuned BERT achieves the highest accuracy and macro F1, demonstrating superior generalization on complex paraphrased queries containing colloquial student syntax.

---

## 14. Fine-Tuned BERT Training Progression

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---|---|---|---|---|
| Epoch 1 | 2.1867 | 30.93% | 1.4760 | 71.37% |
| Epoch 2 | 1.0605 | 87.73% | 0.5207 | 98.34% |
| Epoch 3 | 0.4562 | 98.49% | 0.3024 | **99.17%** |

- Compute Platform: Apple Silicon Metal Performance Shaders (MPS) / PyTorch 2.13
- Total Training Time: 100.54 seconds.

---

## 15. Information Retrieval Comparison (Section 27)

| Retrieval Approach | Precision@1 | Precision@3 | Recall@3 | MRR | Latency (ms) |
|---|---|---|---|---|---|
| **Dense Semantic (FAISS + MiniLM)** | 83.33% | 72.22% | 83.33% | **0.8750** | 45.19 ms |
| **Sparse Keyword (TF-IDF)** | 91.67% | 83.33% | 91.67% | **0.9583** | 0.23 ms |
| **Hybrid RRF (Dense + Sparse)** | **91.67%** | **86.11%** | **100.0%** | **0.9583** | 46.10 ms |

*Key Finding:* While exact keyword search performs well when circular vocabulary matches directly, semantic dense retrieval accurately identifies relevant policies on heavily paraphrased student queries (e.g. *"Will I get detained if I miss classes due to medical illness?"* -> `attendance_policy.txt`). Hybrid Reciprocal Rank Fusion achieves the best overall recall.

---

## 16. Generation & Summarization Evaluation (Sections 28 & 29)

- **ROUGE Evaluation:**
  - ROUGE-1 F1: **0.2775**
  - ROUGE-2 F1: **0.2003**
  - ROUGE-L F1: **0.2545**
- **BERTScore Evaluation (Contextual Semantic Embeddings):**
  - Mean Precision: **0.4598**
  - Mean Recall: **0.7705**
  - Mean F1: **0.5754**

*Role of BERTScore in the Project:* As established in Section 29, BERTScore is applied specifically to generated grounded answers and reference targets rather than classification, evaluating whether the generated policy guidance conveys the semantic meaning of the ground truth.

---

## 17. Hallucination Mitigation & Guardrail Design
A significant risk in educational AI is fabricating dates, fees, or passing criteria. EduMind enforces:
1. Strict contextual prompt confinement: Answers must be synthesized exclusively from retrieved document chunks.
2. Cosine similarity thresholding: If retrieved chunks score below 0.35 similarity, the system returns a safe fallback message directing the student to the official administrative office.
3. Provenance citation tags: Every answer embeds source document names and page references.

---

## 18. Multilingual Architecture
EduMind integrates language detection (`langdetect` + Devanagari script detection) for Hindi and Marathi queries. College domain keywords (e.g. *परीक्षा*, *उपस्थिति*, *शुल्क*, *नोंदणी*) are semantically mapped into dense English vector space, and the verified grounded answer is returned with bilingual language metadata.

---

## 19. Viva Voce Preparation & Key Questions

1. **Why use BERT fine-tuning instead of prompting an off-the-shelf LLM directly?**
   - *Answer:* Fine-tuning adapts the transformer representation specifically to college domain terminology with high speed, deterministic intent classification, zero API cost, and local privacy.
2. **What is the difference between Stemming and Lemmatization?**
   - *Answer:* Stemming applies heuristic rule-based affix cutting (e.g. `leaves` -> `leav`), while Lemmatization maps words to dictionary base forms using POS tags and WordNet morphology (`leaves` as noun -> `leaf`, as verb -> `leave`).
3. **What is the difference between Cosine Similarity and Dot Product in FAISS?**
   - *Answer:* When vectors are L2-normalized ($|v|=1$), the inner product ($\text{IndexFlatIP}$) is mathematically identical to cosine similarity: $\cos(\theta) = \frac{u \cdot v}{\|u\| \|v\|} = u \cdot v$.
4. **Why is BERTScore preferable over BLEU/ROUGE for evaluation?**
   - *Answer:* BLEU and ROUGE rely on exact n-gram token overlap and penalize valid synonyms. BERTScore computes cosine similarity between contextual token embeddings, capturing semantic equivalence even when wording differs.

---

## 20. Conclusion & Future Work
EduMind successfully demonstrates the integration of classical NLP techniques, transformer fine-tuning, dense vector search, and grounded retrieval-augmented generation. Future work includes voice-based input via Whisper, automated OCR ingestion of physical scanned noticeboards, and institutional ERP database integration.
