"""
End-to-End Question Answering Pipeline for EduMind.
Unifies NLP Preprocessing, Fine-Tuned BERT Intent Classification,
Semantic FAISS Retrieval, and Grounded Answer Generation.
"""

import time
from typing import Dict, Any, Optional
from preprocessing.tokenization import tokenize_words
from preprocessing.lemmatization import lemmatize_text
from preprocessing.pos_tagging import tag_pos, extract_syntactic_patterns
from preprocessing.wordnet_analysis import analyze_query_semantics
from classification.evaluate_bert import predict_bert_intent
from retrieval.retriever import EduMindRetriever
from rag.generator import EduMindGenerator

class EduMindQAPipeline:
    def __init__(self):
        print("Initializing EduMind End-to-End QA Pipeline...")
        self.retriever = EduMindRetriever()
        self.generator = EduMindGenerator()
        print("Pipeline components ready.")

    def run_pipeline(
        self,
        query: str,
        top_k: int = 3,
        retrieval_method: str = "semantic"
    ) -> Dict[str, Any]:
        """
        Executes the full EduMind pipeline on a student query.
        
        Args:
            query: Input user query string.
            top_k: Number of document passages to retrieve.
            retrieval_method: 'semantic', 'keyword', or 'hybrid'.
            
        Returns:
            Structured dictionary of all pipeline intermediate outputs and the final answer.
        """
        start_time = time.time()
        
        # 1. Mandatory NLP Preprocessing
        tokens = tokenize_words(query)
        lemmas = lemmatize_text(query)
        pos_tags = tag_pos(query)
        syntactic_patterns = extract_syntactic_patterns(query)
        wordnet_semantics = analyze_query_semantics(query)
        
        nlp_preprocessing_results = {
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "syntactic_patterns": syntactic_patterns,
            "wordnet_semantics": wordnet_semantics
        }
        
        # 2. Intent Classification via Fine-Tuned BERT
        intent_start = time.time()
        try:
            intent_res = predict_bert_intent(query)
            predicted_intent = intent_res["intent"]
            intent_confidence = intent_res["confidence"]
            all_intent_scores = intent_res["all_scores"]
        except Exception as e:
            # Fallback to TF-IDF baseline if BERT not ready
            from classification.baseline_tfidf import predict_intent as predict_baseline
            intent_res = predict_baseline(query)
            predicted_intent = intent_res["intent"]
            intent_confidence = intent_res["confidence"]
            all_intent_scores = {predicted_intent: intent_confidence}
        intent_latency_ms = round((time.time() - intent_start) * 1000, 2)
        
        # 3. Document Retrieval
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k, method=retrieval_method)
        retrieval_latency_ms = round((time.time() - retrieval_start) * 1000, 2)
        
        # 4. Context Grounded Answer Generation
        gen_start = time.time()
        gen_output = self.generator.generate_grounded_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            intent=predicted_intent
        )
        gen_latency_ms = round((time.time() - gen_start) * 1000, 2)
        
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "query": query,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "all_intent_scores": all_intent_scores,
            "answer": gen_output["answer"],
            "sources": gen_output["sources"],
            "is_grounded": gen_output["is_grounded"],
            "retrieved_chunks": retrieved_chunks,
            "nlp_preprocessing": nlp_preprocessing_results,
            "latency": {
                "intent_ms": intent_latency_ms,
                "retrieval_ms": retrieval_latency_ms,
                "generation_ms": gen_latency_ms,
                "total_ms": total_time_ms
            }
        }

if __name__ == "__main__":
    pipeline = EduMindQAPipeline()
    sample = "What is the penalty if I submit my exam registration form late?"
    res = pipeline.run_pipeline(sample)
    print("\n=== EduMind Full Pipeline Execution ===")
    print(f"Query: {res['query']}")
    print(f"Intent: {res['predicted_intent']} (Confidence: {res['intent_confidence']:.2%})")
    print(f"Answer:\n{res['answer']}")
    print(f"Sources: {res['sources']}")
    print(f"Total Latency: {res['latency']['total_ms']} ms")
