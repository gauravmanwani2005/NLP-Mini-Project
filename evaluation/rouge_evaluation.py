"""
ROUGE Evaluation Module for EduMind Question Answering & Summarization.
Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
from rouge_score import rouge_scorer

QA_TEST_CASES = [
    {
        "query": "What is the minimum attendance requirement for theory and laboratory courses?",
        "reference": "Students must maintain minimum 75% attendance in theory courses and 80% attendance in laboratories to be eligible for semester examinations.",
        "intent": "attendance"
    },
    {
        "query": "What is the late fee penalty for exam registration?",
        "reference": "Late exam registration carries a penalty of Rs. 500 permitted up to 7 days after the regular deadline.",
        "intent": "exam_registration"
    },
    {
        "query": "What is the CGPA eligibility criteria for Tier-1 placement companies?",
        "reference": "Tier-1 product companies require a minimum CGPA of 8.0 without any live or past standing backlogs.",
        "intent": "placements"
    },
    {
        "query": "How many books can undergraduate students borrow and for how many days?",
        "reference": "Undergraduate B.Tech students may borrow up to 4 books simultaneously for 14 calendar days.",
        "intent": "library"
    },
    {
        "query": "What are the passing marks required in end-semester examinations?",
        "reference": "A student must secure a minimum of 40% marks in the End-Semester Examination and an aggregate of 45% to pass.",
        "intent": "results"
    }
]

def evaluate_rouge_scores():
    from rag.qa_pipeline import EduMindQAPipeline
    pipeline = EduMindQAPipeline()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_f1s = []
    rouge2_f1s = []
    rougeL_f1s = []
    
    detailed_results = []
    
    print("Evaluating Grounded Answer Generation with ROUGE metrics...")
    for tc in QA_TEST_CASES:
        out = pipeline.run_pipeline(tc["query"])
        gen_answer = out["answer"]
        ref_answer = tc["reference"]
        
        scores = scorer.score(ref_answer, gen_answer)
        
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure
        
        rouge1_f1s.append(r1)
        rouge2_f1s.append(r2)
        rougeL_f1s.append(rl)
        
        detailed_results.append({
            "query": tc["query"],
            "reference": ref_answer,
            "generated": gen_answer,
            "rouge1": round(r1, 4),
            "rouge2": round(r2, 4),
            "rougeL": round(rl, 4)
        })
        
    avg_scores = {
        "rouge1_f1": round(sum(rouge1_f1s) / len(rouge1_f1s), 4),
        "rouge2_f1": round(sum(rouge2_f1s) / len(rouge2_f1s), 4),
        "rougeL_f1": round(sum(rougeL_f1s) / len(rougeL_f1s), 4),
        "num_test_cases": len(QA_TEST_CASES),
        "detailed_results": detailed_results
    }
    
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/rouge_evaluation.json", "w") as f:
        json.dump(avg_scores, f, indent=2)
        
    print("=== ROUGE Evaluation Results ===")
    print(f"ROUGE-1 F1: {avg_scores['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {avg_scores['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {avg_scores['rougeL_f1']:.4f}")
    
    return avg_scores

if __name__ == "__main__":
    evaluate_rouge_scores()
