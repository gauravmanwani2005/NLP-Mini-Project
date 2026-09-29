"""
Response Generator Module for EduMind RAG.
Synthesizes context-grounded responses from retrieved document chunks,
enforces anti-hallucination guardrails, and attaches precise citations.
"""

from typing import List, Dict, Any, Optional
import re
from preprocessing.sentence_segmentation import segment_sentences
from retrieval.embeddings import EmbeddingGenerator
import numpy as np

FALLBACK_MESSAGE = (
    "Based on the official college documents provided, this specific information is not currently "
    "available in the institutional knowledge base. Please consult the departmental academic coordinator "
    "or the Office of Student Affairs for official clarification."
)

class EduMindGenerator:
    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold
        self.embedding_gen = EmbeddingGenerator()

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes an answer strictly from retrieved chunks.
        
        Args:
            query: The student question.
            retrieved_chunks: Top-K chunks returned by retriever.
            intent: Predicted academic intent.
            
        Returns:
            dict containing:
                - 'answer': Grounded answer text
                - 'sources': List of cited documents
                - 'is_grounded': bool indicating whether supported info was found
                - 'confidence': float
        """
        if not retrieved_chunks:
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "is_grounded": False,
                "confidence": 0.0
            }
            
        # Check top similarity score
        top_score = retrieved_chunks[0].get("similarity_score", 0.0)
        if top_score < self.similarity_threshold:
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [c["source"] for c in retrieved_chunks[:1]],
                "is_grounded": False,
                "confidence": round(float(top_score), 4)
            }
            
        # Extract and rank individual sentences from the top retrieved chunks
        candidate_sentences = []
        for chunk in retrieved_chunks:
            source = chunk.get("source", "Document")
            page = chunk.get("page_number", 1)
            raw_content = chunk.get("content", "")
            sents = segment_sentences(raw_content)
            
            for s in sents:
                clean_s = s.strip()
                if len(clean_s) > 25 and not clean_s.startswith("EDUMIND") and not clean_s.startswith("Document Code"):
                    candidate_sentences.append({
                        "sentence": clean_s,
                        "source": source,
                        "page": page
                    })
                    
        if not candidate_sentences:
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [c["source"] for c in retrieved_chunks],
                "is_grounded": False,
                "confidence": 0.0
            }
            
        # Rank sentences against query using semantic similarity
        sent_texts = [cs["sentence"] for cs in candidate_sentences]
        query_emb = self.embedding_gen.embed_query(query)
        sent_embs = self.embedding_gen.embed_texts(sent_texts)
        
        scores = np.dot(sent_embs, query_emb)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Select top 3-4 distinct sentences
        selected_sents = []
        cited_sources = set()
        seen_texts = set()
        
        for idx in ranked_indices[:6]:
            item = candidate_sentences[idx]
            stext = item["sentence"]
            # Deduplicate similar sentences
            if stext[:30].lower() in seen_texts:
                continue
            seen_texts.add(stext[:30].lower())
            
            ref_tag = f"[{item['source']}, p.{item['page']}]"
            selected_sents.append(f"{stext} {ref_tag}")
            cited_sources.add(f"{item['source']} (Page {item['page']})")
            
            if len(selected_sents) >= 3:
                break
                
        if not selected_sents:
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "is_grounded": False,
                "confidence": round(float(top_score), 4)
            }
            
        # Format structured grounded answer
        answer_body = " ".join(selected_sents)
        
        # Add summary intro based on intent if available
        intent_prefix = ""
        if intent:
            readable_intent = intent.replace("_", " ").title()
            intent_prefix = f"**Official Academic Policy regarding {readable_intent}:**\n\n"
            
        final_answer = f"{intent_prefix}{answer_body}"
        
        return {
            "answer": final_answer,
            "sources": sorted(list(cited_sources)),
            "is_grounded": True,
            "confidence": round(float(top_score), 4)
        }

if __name__ == "__main__":
    from retrieval.retriever import EduMindRetriever
    retriever = EduMindRetriever()
    gen = EduMindGenerator()
    
    test_q = "What is the passing criteria and minimum marks in semester exams?"
    chunks = retriever.retrieve_semantic(test_q, top_k=3)
    resp = gen.generate_grounded_answer(test_q, chunks, "results")
    print(f"Query: {test_q}")
    print(f"Grounded: {resp['is_grounded']} (Confidence: {resp['confidence']})")
    print(f"Answer:\n{resp['answer']}")
    print(f"Sources: {resp['sources']}")
