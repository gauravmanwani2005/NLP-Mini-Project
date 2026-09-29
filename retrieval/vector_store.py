"""
FAISS Vector Store for EduMind.
Stores dense vector embeddings and chunk metadata, enabling sub-millisecond Top-K semantic retrieval.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from retrieval.embeddings import EmbeddingGenerator

DEFAULT_INDEX_DIR = "retrieval/index"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunks_metadata.json"

class FAISSVectorStore:
    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR, embedding_generator: EmbeddingGenerator = None):
        self.index_dir = index_dir
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.index_dir, INDEX_FILE)
        self.meta_path = os.path.join(self.index_dir, METADATA_FILE)

    def build_from_chunks(self, chunks: List[Dict[str, Any]]):
        """Builds a new FAISS index from document chunks and saves to disk."""
        if not chunks:
            raise ValueError("No chunks provided to build vector index.")
            
        print(f"Generating embeddings for {len(chunks)} document chunks...")
        texts = [c["content"] for c in chunks]
        embeddings = self.embedding_gen.embed_texts(texts)
        
        dim = embeddings.shape[1]
        # Inner Product with normalized vectors is exact Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks_metadata = chunks
        
        self.save()
        print(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def save(self):
        """Serializes FAISS index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_metadata, f, indent=2)
            print(f"Index persisted to {self.index_path}")

    def load(self) -> bool:
        """Loads FAISS index and metadata from disk if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)
            print(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
            return True
        return False

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves Top-K most semantically similar chunks for a query.
        
        Returns:
            List of dicts: chunk metadata + 'similarity_score'.
        """
        if self.index is None or not self.chunks_metadata:
            loaded = self.load()
            if not loaded:
                raise RuntimeError("Vector store is empty and no saved index found. Build index first.")
                
        query_vector = self.embedding_gen.embed_query(query).reshape(1, -1)
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self.chunks_metadata[idx])
            meta["similarity_score"] = round(float(dist), 4)
            results.append(meta)
            
        return results

if __name__ == "__main__":
    from retrieval.document_loader import DocumentLoader
    from retrieval.chunking import DocumentChunker
    
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    chunker = DocumentChunker()
    all_chunks = chunker.chunk_all_documents(docs)
    
    store = FAISSVectorStore()
    store.build_from_chunks(all_chunks)
    
    test_q = "Can attendance between 65% and 75% be condoned?"
    print(f"\nSearching for: '{test_q}'")
    matches = store.search(test_q, top_k=2)
    for i, m in enumerate(matches, 1):
        print(f"\nMatch {i} [Score: {m['similarity_score']}] Source: {m['source']} (p.{m['page_number']})")
        print(f"Content: {m['content']}")
