"""
BERTScore Evaluation Module for EduMind.
Evaluates semantic text generation quality using contextual embeddings,
as specified in Section 29 of the project plan.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any
from bert_score import score
from evaluation.rouge_evaluation import QA_TEST_CASES
from rag.qa_pipeline import EduMindQAPipeline

def evaluate_bertscore() -> Dict[str, Any]:
    pipeline = EduMindQAPipeline()
    
    candidates = []
    references = []
    
    print("Generating answers for BERTScore evaluation...")
    for tc in QA_TEST_CASES:
        out = pipeline.run_pipeline(tc["query"])
        candidates.append(out["answer"])
        references.append(tc["reference"])
        
    print(f"Computing BERTScore across {len(candidates)} grounded QA pairs using bert-base-uncased...")
    # Compute BERTScore using bert-base-uncased
    P, R, F1 = score(candidates, references, model_type="bert-base-uncased", verbose=False)
    
    mean_p = round(float(P.mean().item()), 4)
    mean_r = round(float(R.mean().item()), 4)
    mean_f1 = round(float(F1.mean().item()), 4)
    
    detailed = []
    for i, tc in enumerate(QA_TEST_CASES):
        detailed.append({
            "query": tc["query"],
            "reference": tc["reference"],
            "generated": candidates[i],
            "bertscore_precision": round(float(P[i].item()), 4),
            "bertscore_recall": round(float(R[i].item()), 4),
            "bertscore_f1": round(float(F1[i].item()), 4)
        })
        
    results = {
        "metric": "BERTScore",
        "model_type": "bert-base-uncased",
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "mean_f1": mean_f1,
        "test_cases": detailed
    }
    
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/bertscore_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("=== BERTScore Evaluation Results ===")
    print(f"BERTScore Precision : {mean_p:.4f}")
    print(f"BERTScore Recall    : {mean_r:.4f}")
    print(f"BERTScore F1        : {mean_f1:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_bertscore()
