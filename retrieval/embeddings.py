"""
Dense Semantic Embedding Module for EduMind.
Generates sentence and document embeddings using sentence-transformers (all-MiniLM-L6-v2).
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingGenerator:
    _instance = None
    
    def __new__(cls, model_name: str = DEFAULT_EMBEDDING_MODEL):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        if self._initialized:
            return
        self.model_name = model_name
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        print(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Computes dense L2-normalized embeddings for a list of strings.
        
        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            
        Returns:
            2D numpy array of shape (len(texts), embedding_dim) as float32.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Computes normalized 1D embedding for a single search query."""
        emb = self.model.encode([query], normalize_embeddings=True)
        return np.array(emb[0], dtype=np.float32)

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    test_sentences = [
        "What is the minimum attendance required for examinations?",
        "Students must maintain at least 75% attendance to avoid detention."
    ]
    vectors = generator.embed_texts(test_sentences)
    sim = np.dot(vectors[0], vectors[1])
    print(f"Embeddings shape: {vectors.shape}")
    print(f"Cosine similarity between query & policy: {sim:.4f}")
