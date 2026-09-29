"""
Classification Metrics Compilation Module for EduMind.
Aggregates performance comparisons across TF-IDF, Pretrained BERT, and Fine-Tuned BERT.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = "evaluation/results"

def load_all_metrics() -> Dict[str, Any]:
    """Loads all saved evaluation results into a single dashboard-friendly structure."""
    data = {
        "model_comparison": [],
        "bert_report": {},
        "baseline_report": {},
        "retrieval_comparison": {},
        "rouge_scores": {},
        "bertscore": {}
    }
    
    comp_file = os.path.join(RESULTS_DIR, "model_comparison.json")
    if os.path.exists(comp_file):
        with open(comp_file, "r") as f:
            data["model_comparison"] = json.load(f)
            
    bert_rep_file = os.path.join(RESULTS_DIR, "bert_test_report.json")
    if os.path.exists(bert_rep_file):
        with open(bert_rep_file, "r") as f:
            data["bert_report"] = json.load(f)
            
    base_rep_file = os.path.join(RESULTS_DIR, "baseline_report.json")
    if os.path.exists(base_rep_file):
        with open(base_rep_file, "r") as f:
            data["baseline_report"] = json.load(f)
            
    ret_file = os.path.join(RESULTS_DIR, "retrieval_comparison.json")
    if os.path.exists(ret_file):
        with open(ret_file, "r") as f:
            data["retrieval_comparison"] = json.load(f)
            
    rouge_file = os.path.join(RESULTS_DIR, "rouge_evaluation.json")
    if os.path.exists(rouge_file):
        with open(rouge_file, "r") as f:
            data["rouge_scores"] = json.load(f)
            
    bertscore_file = os.path.join(RESULTS_DIR, "bertscore_evaluation.json")
    if os.path.exists(bertscore_file):
        with open(bertscore_file, "r") as f:
            data["bertscore"] = json.load(f)
            
    return data

def get_comparison_dataframe() -> pd.DataFrame:
    metrics = load_all_metrics()
    if metrics["model_comparison"]:
        df = pd.DataFrame(metrics["model_comparison"])
        return df
    return pd.DataFrame()

if __name__ == "__main__":
    m = load_all_metrics()
    print("Loaded evaluation metrics summary:")
    print(get_comparison_dataframe())
