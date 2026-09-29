"""
Baseline Model: TF-IDF + Logistic Regression for EduMind Intent Classification.
Provides the traditional ML benchmark against which fine-tuned BERT is compared.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

MODEL_DIR = "classification/model"
RESULTS_DIR = "evaluation/results"

def train_and_evaluate_baseline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df["text"]
    y_train = train_df["intent"]
    
    X_test = test_df["text"]
    y_test = test_df["intent"]
    
    # 1. Feature Extraction: TF-IDF with unigrams & bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 2. Classifier: Logistic Regression
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # 3. Evaluation on Test Set
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted": round(float(rec), 4),
        "f1_weighted": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "test_samples": len(y_test)
    }
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save artifacts
    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": list(clf.classes_)
    }
    joblib.dump(model_bundle, os.path.join(MODEL_DIR, "baseline_tfidf.joblib"))
    
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(RESULTS_DIR, "baseline_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print("=== Baseline TF-IDF + Logistic Regression Results ===")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved model to {MODEL_DIR}/baseline_tfidf.joblib")
    return metrics

def predict_intent(query: str):
    """Utility to predict intent of a single query using saved baseline."""
    bundle_path = os.path.join(MODEL_DIR, "baseline_tfidf.joblib")
    if not os.path.exists(bundle_path):
        train_and_evaluate_baseline()
    bundle = joblib.load(bundle_path)
    vec = bundle["vectorizer"].transform([query])
    probs = bundle["classifier"].predict_proba(vec)[0]
    best_idx = probs.argmax()
    return {
        "intent": bundle["classes"][best_idx],
        "confidence": float(probs[best_idx])
    }

if __name__ == "__main__":
    train_and_evaluate_baseline()
    demo_q = "How much attendance is required to sit for semester examinations?"
    res = predict_intent(demo_q)
    print(f"\nTest prediction for: '{demo_q}' -> {res['intent']} (Confidence: {res['confidence']:.2%})")
