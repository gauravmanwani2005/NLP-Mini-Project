"""
Information Retrieval Evaluation Module for EduMind.
Computes Precision@K, Recall@K, and Mean Reciprocal Rank (MRR)
comparing Semantic Dense Retrieval vs Keyword (TF-IDF) Retrieval.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
from retrieval.retriever import EduMindRetriever

BENCHMARK_QUERIES = [
    {
        "query": "What is the minimum attendance required to appear for semester examination?",
        "expected_doc": "attendance_policy.txt",
        "paraphrased": False
    },
    {
        "query": "Will I get detained if I miss classes due to medical illness?",
        "expected_doc": "attendance_policy.txt",
        "paraphrased": True
    },
    {
        "query": "When is the deadline to fill the semester exam registration form?",
        "expected_doc": "examination_regulations.txt",
        "paraphrased": False
    },
    {
        "query": "Can I see my checked answer sheets and apply for revaluation?",
        "expected_doc": "examination_regulations.txt",
        "paraphrased": True
    },
    {
        "query": "What is the annual tuition fee and development charges for engineering?",
        "expected_doc": "fee_structure_and_scholarships.txt",
        "paraphrased": False
    },
    {
        "query": "Are there concessions or fee discounts for top scoring students?",
        "expected_doc": "fee_structure_and_scholarships.txt",
        "paraphrased": True
    },
    {
        "query": "How many books can a B.Tech student borrow and what is the overdue fine?",
        "expected_doc": "library_and_digital_resources.txt",
        "paraphrased": False
    },
    {
        "query": "How can I access digital journals like IEEE and ScienceDirect from home?",
        "expected_doc": "library_and_digital_resources.txt",
        "paraphrased": True
    },
    {
        "query": "What is the placement criteria for Tier-1 companies and maximum package?",
        "expected_doc": "placement_and_training_handbook.txt",
        "paraphrased": False
    },
    {
        "query": "Can students with backlogs sit for campus recruitment drives?",
        "expected_doc": "placement_and_training_handbook.txt",
        "paraphrased": True
    },
    {
        "query": "How many total credits are required for completing the degree under CBCS?",
        "expected_doc": "academic_calendar_and_courses.txt",
        "paraphrased": False
    },
    {
        "query": "What are the daily class timings and lecture period hours?",
        "expected_doc": "academic_calendar_and_courses.txt",
        "paraphrased": True
    }
]

def evaluate_retrieval():
    retriever = EduMindRetriever()
    k_vals = [1, 3]
    
    results = {
        "semantic": {"p@1": 0.0, "p@3": 0.0, "rec@3": 0.0, "mrr": 0.0, "latency_ms": 0.0},
        "keyword": {"p@1": 0.0, "p@3": 0.0, "rec@3": 0.0, "mrr": 0.0, "latency_ms": 0.0}
    }
    
    n_queries = len(BENCHMARK_QUERIES)
    
    for method in ["semantic", "keyword"]:
        p1_hits = 0
        p3_hits = 0
        rec3_hits = 0
        reciprocal_ranks = []
        latencies = []
        
        for item in BENCHMARK_QUERIES:
            q = item["query"]
            target = item["expected_doc"]
            
            t0 = time.time()
            if method == "semantic":
                retrieved = retriever.retrieve_semantic(q, top_k=3)
            else:
                retrieved = retriever.retrieve_keyword(q, top_k=3)
            latencies.append((time.time() - t0) * 1000)
            
            retrieved_sources = [r["source"] for r in retrieved]
            
            # P@1
            if retrieved_sources and retrieved_sources[0] == target:
                p1_hits += 1
                
            # P@3 & Rec@3
            target_matches = sum(1 for s in retrieved_sources[:3] if s == target)
            p3_hits += (target_matches / 3.0)
            if target_matches > 0:
                rec3_hits += 1
                
            # MRR
            rank = None
            for idx, s in enumerate(retrieved_sources, 1):
                if s == target:
                    rank = idx
                    break
            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                
        results[method]["p@1"] = round(p1_hits / n_queries, 4)
        results[method]["p@3"] = round(p3_hits / n_queries, 4)
        results[method]["rec@3"] = round(rec3_hits / n_queries, 4)
        results[method]["mrr"] = round(sum(reciprocal_ranks) / n_queries, 4)
        results[method]["latency_ms"] = round(sum(latencies) / n_queries, 2)
        
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/retrieval_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("=== Information Retrieval Evaluation Benchmark ===")
    print(f"Semantic Retrieval -> P@1: {results['semantic']['p@1']:.2%} | P@3: {results['semantic']['p@3']:.2%} | MRR: {results['semantic']['mrr']:.4f} | Latency: {results['semantic']['latency_ms']} ms")
    print(f"Keyword Retrieval  -> P@1: {results['keyword']['p@1']:.2%} | P@3: {results['keyword']['p@3']:.2%} | MRR: {results['keyword']['mrr']:.4f} | Latency: {results['keyword']['latency_ms']} ms")
    
    return results

if __name__ == "__main__":
    evaluate_retrieval()
