"""
Creates the 6 planned Jupyter notebooks with reproducible code and explanations.
"""

import json
import os

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

# Notebook 1: Dataset Analysis
nb1_cells = [
    md_cell("# EduMind Notebook 01: Domain-Specific Dataset Analysis & Exploration\nThis notebook inspects the college academic query dataset, analyzes class balance, sequence length distributions, and token vocabulary."),
    code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_df = pd.read_csv('../data/processed/train.csv')
val_df = pd.read_csv('../data/processed/validation.csv')
test_df = pd.read_csv('../data/processed/test.csv')

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")"""),
    code_cell("""# Inspect class distribution
print("Train Class Distribution:")
print(train_df['intent'].value_counts())"""),
    code_cell("""# Plot class balance
plt.figure(figsize=(12, 5))
sns.countplot(data=train_df, x='intent', order=train_df['intent'].value_counts().index, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Training Set Class Balance across 12 Academic Intents')
plt.tight_layout()
plt.show()"""),
    code_cell("""# Sentence length distribution
train_df['char_length'] = train_df['text'].apply(len)
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))

print("Summary statistics of query word count:")
print(train_df['word_count'].describe())""")
]

# Notebook 2: NLP Preprocessing
nb2_cells = [
    md_cell("# EduMind Notebook 02: Mandatory NLP Preprocessing\nDemonstrates Sentence Boundary Detection, Tokenization, Lemmatization, POS Tagging, and WordNet Semantic Analysis."),
    code_cell("""import sys
sys.path.insert(0, '..')

from preprocessing.sentence_segmentation import segment_sentences
from preprocessing.tokenization import tokenize_words, analyze_tokens
from preprocessing.lemmatization import lemmatize_text, compare_stemming_and_lemmatization
from preprocessing.pos_tagging import tag_pos, extract_syntactic_patterns
from preprocessing.wordnet_analysis import get_synsets_info, compute_wordnet_similarity"""),
    code_cell("""sample_text = 'The semester examination begins on Monday. Students must carry their hall tickets! Attendance below 75% will lead to detention.'
print('=== 1. Sentence Boundary Detection ===')
sentences = segment_sentences(sample_text)
for i, s in enumerate(sentences, 1):
    print(f'[{i}] {s}')"""),
    code_cell("""print('=== 2. Tokenization & Lexical Stats ===')
stats = analyze_tokens(sample_text)
print('Total Tokens:', stats['token_count'])
print('Type-Token Ratio:', stats['type_token_ratio'])
print('Tokens:', stats['raw_tokens'])"""),
    code_cell("""print('=== 3. Lemmatization vs Stemming ===')
words = ['examinations', 'studying', 'corpora', 'better', 'leaves']
comp = compare_stemming_and_lemmatization(words)
import pandas as pd
print(pd.DataFrame(comp))"""),
    code_cell("""print('=== 4. Syntactic POS Tagging ===')
tagged = tag_pos(sentences[0])
for w, t, desc in tagged:
    print(f'{w:15} | {t:6} | {desc}')"""),
    code_cell("""print('=== 5. WordNet Semantic Analysis ===')
sim = compute_wordnet_similarity('examination', 'test')
print('Path Similarity between examination & test:', sim['path_similarity'])
print('Wu-Palmer Similarity:', sim['wup_similarity'])""")
]

# Notebook 3: Baseline Model
nb3_cells = [
    md_cell("# EduMind Notebook 03: Baseline Model (TF-IDF + Logistic Regression)\nEvaluates traditional machine learning text classification on the 12 college intents."),
    code_cell("""import sys
sys.path.insert(0, '..')
import json
from classification.baseline_tfidf import train_and_evaluate_baseline, predict_intent

metrics = train_and_evaluate_baseline()
print('Baseline Metrics:', json.dumps(metrics, indent=2))"""),
    code_cell("""# Test single query prediction
query = 'When do I need to pay my tuition fee and what is the late penalty?'
res = predict_intent(query)
print(f'Query: {query}')
print(f'Predicted Intent: {res[\"intent\"]} (Confidence: {res[\"confidence\"]:.2%})')""")
]

# Notebook 4: BERT Fine-Tuning
nb4_cells = [
    md_cell("# EduMind Notebook 04: BERT Fine-Tuning and Model Evaluation\nDemonstrates fine-tuning `bert-base-uncased` with PyTorch, evaluating test metrics and generating the confusion matrix."),
    code_cell("""import sys
sys.path.insert(0, '..')
import json
from classification.evaluate_bert import evaluate_test_set, predict_bert_intent

test_metrics = evaluate_test_set()
print('BERT Test Set Performance:', json.dumps(test_metrics, indent=2))"""),
    code_cell("""# Compare Models
with open('../evaluation/results/model_comparison.json', 'r') as f:
    comp = json.load(f)
import pandas as pd
pd.DataFrame(comp)"""),
    code_cell("""# Test Fine-Tuned BERT Inference
q = 'Can attendance between 65% and 75% be condoned on medical grounds?'
pred = predict_bert_intent(q)
print(f'Query: {q}')
print(f'Intent: {pred[\"intent\"]} (Confidence: {pred[\"confidence\"]:.2%})')""")
]

# Notebook 5: Semantic Search
nb5_cells = [
    md_cell("# EduMind Notebook 05: Document Chunking and Semantic Search with FAISS\nDemonstrates PDF/TXT loading, sentence chunking, dense embedding with MiniLM, and FAISS indexing."),
    code_cell("""import sys
sys.path.insert(0, '..')
from retrieval.document_loader import DocumentLoader
from retrieval.chunking import DocumentChunker
from retrieval.vector_store import FAISSVectorStore

loader = DocumentLoader('../data/raw/documents')
docs = loader.load_all_documents()
print(f'Loaded {len(docs)} official documents.')

chunker = DocumentChunker(target_chunk_size=350, overlap=50)
chunks = chunker.chunk_all_documents(docs)
print(f'Generated {len(chunks)} sentence-aware chunks.')"""),
    code_cell("""store = FAISSVectorStore()
store.load()
query = 'What is the passing criteria for end-semester examinations?'
matches = store.search(query, top_k=3)
for i, m in enumerate(matches, 1):
    print(f'\\nMatch {i} [Score: {m[\"similarity_score\"]}] Source: {m[\"source\"]} (Page {m[\"page_number\"]})')
    print(m['content'])""")
]

# Notebook 6: RAG Evaluation
nb6_cells = [
    md_cell("# EduMind Notebook 06: End-to-End RAG Pipeline & Generation Evaluation\nDemonstrates full question answering with citations, ROUGE evaluation, and BERTScore evaluation."),
    code_cell("""import sys
sys.path.insert(0, '..')
import json
from rag.qa_pipeline import EduMindQAPipeline

pipeline = EduMindQAPipeline()
res = pipeline.run_pipeline('What happens if attendance is lower than 65%?')
print('Intent:', res['predicted_intent'])
print('Answer:\\n', res['answer'])
print('Sources:', res['sources'])
print('Latency:', res['latency'])"""),
    code_cell("""# Display ROUGE evaluation results
with open('../evaluation/results/rouge_evaluation.json', 'r') as f:
    rouge = json.load(f)
print('ROUGE-1 F1:', rouge['rouge1_f1'])
print('ROUGE-2 F1:', rouge['rouge2_f1'])
print('ROUGE-L F1:', rouge['rougeL_f1'])"""),
    code_cell("""# Display BERTScore results
with open('../evaluation/results/bertscore_evaluation.json', 'r') as f:
    bs = json.load(f)
print('BERTScore Precision:', bs['mean_precision'])
print('BERTScore Recall:', bs['mean_recall'])
print('BERTScore F1:', bs['mean_f1'])""")
]

notebooks = {
    "01_dataset_analysis.ipynb": nb1_cells,
    "02_nlp_preprocessing.ipynb": nb2_cells,
    "03_baseline_model.ipynb": nb3_cells,
    "04_bert_finetuning.ipynb": nb4_cells,
    "05_semantic_search.ipynb": nb5_cells,
    "06_rag_evaluation.ipynb": nb6_cells
}

for filename, cells in notebooks.items():
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created notebook: {filepath}")
