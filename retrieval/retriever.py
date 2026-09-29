"""
High-Level Retriever Module for EduMind.
Implements Semantic Retrieval (FAISS + MiniLM) and Keyword Retrieval (BM25/TF-IDF)
to enable the experimental comparison specified in Section 27 of the project specification.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from retrieval.vector_store import FAISSVectorStore
from retrieval.document_loader import DocumentLoader
from retrieval.chunking import DocumentChunker

class EduMindRetriever:
    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        self.vector_store = vector_store or FAISSVectorStore()
        # Ensure index exists or build it
        if not self.vector_store.load():
            print("FAISS index not found on disk. Building from data/raw/documents...")
            loader = DocumentLoader()
            docs = loader.load_all_documents()
            chunker = DocumentChunker()
            chunks = chunker.chunk_all_documents(docs)
            self.vector_store.build_from_chunks(chunks)
            
        self.chunks = self.vector_store.chunks_metadata
        self._init_keyword_retriever()

    def _init_keyword_retriever(self):
        """Initializes TF-IDF index over all document chunks for comparison experiments."""
        self.corpus = [c["content"] for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        if self.corpus:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        else:
            self.tfidf_matrix = None

    def retrieve_semantic(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search via dense vector cosine similarity."""
        results = self.vector_store.search(query, top_k=top_k)
        for r in results:
            r["retrieval_method"] = "semantic_dense"
        return results

    def retrieve_keyword(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Keyword-based search via sparse TF-IDF cosine similarity."""
        if self.tfidf_matrix is None or not self.corpus:
            return []
            
        query_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            meta = dict(self.chunks[idx])
            meta["similarity_score"] = round(score, 4)
            meta["retrieval_method"] = "keyword_tfidf"
            results.append(meta)
            
        return results

    def retrieve(self, query: str, top_k: int = 3, method: str = "semantic") -> List[Dict[str, Any]]:
        """Retrieves chunks by specified method ('semantic', 'keyword', or 'hybrid')."""
        if method == "keyword":
            return self.retrieve_keyword(query, top_k=top_k)
        elif method == "hybrid":
            sem_results = self.retrieve_semantic(query, top_k=top_k * 2)
            kw_results = self.retrieve_keyword(query, top_k=top_k * 2)
            
            # Combine via Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            chunk_map = {}
            
            for rank, r in enumerate(sem_results):
                cid = r["chunk_id"]
                chunk_map[cid] = r
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank + 1))
                
            for rank, r in enumerate(kw_results):
                cid = r["chunk_id"]
                chunk_map[cid] = r
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank + 1))
                
            sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
            combined = []
            for cid in sorted_cids:
                item = dict(chunk_map[cid])
                item["rrf_score"] = round(rrf_scores[cid], 5)
                item["retrieval_method"] = "hybrid_rrf"
                combined.append(item)
            return combined
        else:
            return self.retrieve_semantic(query, top_k=top_k)

    def format_context_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved passages into a structured context string with citations."""
        if not retrieved_chunks:
            return "No relevant context found."
            
        context_blocks = []
        for i, c in enumerate(retrieved_chunks, 1):
            source = c.get("source", "Notice")
            page = c.get("page_number", 1)
            content = c.get("content", "").strip()
            score = c.get("similarity_score", 0.0)
            context_blocks.append(f"[Source {i}: {source} (Page {page}) - Relevance: {score:.2f}]\n{content}")
            
        return "\n\n".join(context_blocks)

if __name__ == "__main__":
    retriever = EduMindRetriever()
    q = "What happens if student attendance falls below 65%?"
    print(f"=== Semantic Retrieval for: '{q}' ===")
    sem = retriever.retrieve_semantic(q, top_k=2)
    for s in sem:
        print(f"[{s['similarity_score']}] {s['source']} (p.{s['page_number']}): {s['content'][:120]}...")
        
    print(f"\n=== Keyword Retrieval for: '{q}' ===")
    kw = retriever.retrieve_keyword(q, top_k=2)
    for k in kw:
        print(f"[{k['similarity_score']}] {k['source']} (p.{k['page_number']}): {k['content'][:120]}...")
