"""
Document Summarization Module for EduMind.
Extracts structured executive summaries, key dates, eligibility rules,
and action items from long academic circulars.
"""

from typing import Dict, Any, List
import re
from preprocessing.sentence_segmentation import segment_sentences
from retrieval.embeddings import EmbeddingGenerator
import numpy as np

class AcademicSummarizer:
    def __init__(self):
        self.embedding_gen = EmbeddingGenerator()

    def summarize_document(self, text: str, max_sentences: int = 5) -> Dict[str, Any]:
        """
        Generates a structured multi-section academic summary.
        
        Sections:
        1. Executive Overview
        2. Key Deadlines and Dates
        3. Eligibility and Requirements
        4. Fees and Penalties
        """
        sentences = segment_sentences(text)
        if not sentences:
            return {"error": "Empty text provided."}
            
        # Section-specific semantic anchor queries
        anchors = {
            "key_dates": "deadline submission date commencement schedule time period",
            "eligibility": "minimum requirement eligibility criteria qualification compulsory attendance",
            "fees_penalties": "fee charge penalty cost installment fine refund"
        }
        
        sent_embeddings = self.embedding_gen.embed_texts(sentences)
        
        def get_top_sentences(anchor_query: str, k: int = 2) -> List[str]:
            q_emb = self.embedding_gen.embed_query(anchor_query)
            scores = np.dot(sent_embeddings, q_emb)
            ranked = np.argsort(scores)[::-1]
            selected = []
            for idx in ranked[:k]:
                if scores[idx] > 0.25:
                    selected.append(sentences[idx].strip())
            return selected

        dates_sents = get_top_sentences(anchors["key_dates"], k=2)
        eligibility_sents = get_top_sentences(anchors["eligibility"], k=2)
        fee_sents = get_top_sentences(anchors["fees_penalties"], k=2)
        
        # Centrality-based executive summary: sentence closest to mean document embedding
        doc_mean_emb = np.mean(sent_embeddings, axis=0)
        doc_mean_emb = doc_mean_emb / np.linalg.norm(doc_mean_emb)
        centrality_scores = np.dot(sent_embeddings, doc_mean_emb)
        top_central_indices = np.argsort(centrality_scores)[::-1]
        
        overview_sents = []
        for idx in top_central_indices[:3]:
            overview_sents.append(sentences[idx].strip())
            
        formatted_summary = (
            "### 📌 Executive Overview\n" +
            "\n".join([f"- {s}" for s in overview_sents]) +
            "\n\n### 🗓️ Key Dates & Deadlines\n" +
            ("\n".join([f"- {s}" for s in dates_sents]) if dates_sents else "- No specific calendar dates found.") +
            "\n\n### 📋 Eligibility & Policies\n" +
            ("\n".join([f"- {s}" for s in eligibility_sents]) if eligibility_sents else "- Standard institutional regulations apply.") +
            "\n\n### 💳 Financial & Penalty Details\n" +
            ("\n".join([f"- {s}" for s in fee_sents]) if fee_sents else "- No specific financial charges mentioned.")
        )
        
        return {
            "formatted_summary": formatted_summary,
            "overview_sentences": overview_sents,
            "key_dates": dates_sents,
            "eligibility": eligibility_sents,
            "financial_notes": fee_sents,
            "sentence_count_input": len(sentences),
            "sentence_count_summary": len(overview_sents) + len(dates_sents) + len(eligibility_sents) + len(fee_sents)
        }

if __name__ == "__main__":
    from retrieval.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    if docs:
        summarizer = AcademicSummarizer()
        sample_doc = docs[0]
        print(f"=== Summarizing {sample_doc['source']} ===")
        res = summarizer.summarize_document(sample_doc['content'])
        print(res["formatted_summary"])
