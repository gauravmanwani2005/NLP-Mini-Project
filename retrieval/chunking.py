"""
Document Chunking Module for EduMind.
Implements sentence-boundary aware recursive chunking with configurable overlap and metadata preservation.
"""

from typing import List, Dict, Any
from preprocessing.sentence_segmentation import segment_sentences

class DocumentChunker:
    def __init__(self, target_chunk_size: int = 400, overlap: int = 60):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a loaded document into sentence-aware chunks.
        
        Args:
            doc: Document dict containing 'source', 'file_type', and 'pages'.
            
        Returns:
            List of chunk dicts containing chunk text and provenance metadata.
        """
        chunks = []
        source = doc.get("source", "unknown")
        global_chunk_idx = 0
        
        for page in doc.get("pages", []):
            page_num = page.get("page_number", 1)
            raw_text = page.get("text", "")
            if not raw_text.strip():
                continue
                
            sentences = segment_sentences(raw_text)
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                sent_len = len(sent)
                if current_length + sent_len > self.target_chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    chunks.append({
                        "chunk_id": f"{source}_p{page_num}_c{global_chunk_idx}",
                        "source": source,
                        "page_number": page_num,
                        "chunk_index": global_chunk_idx,
                        "content": chunk_text,
                        "char_count": len(chunk_text)
                    })
                    global_chunk_idx += 1
                    
                    # Compute overlap using the tail sentence(s)
                    overlap_sents = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) <= self.overlap:
                            overlap_sents.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break
                            
                    current_chunk = overlap_sents + [sent]
                    current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                else:
                    current_chunk.append(sent)
                    current_length += sent_len + 1
                    
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                chunks.append({
                    "chunk_id": f"{source}_p{page_num}_c{global_chunk_idx}",
                    "source": source,
                    "page_number": page_num,
                    "chunk_index": global_chunk_idx,
                    "content": chunk_text,
                    "char_count": len(chunk_text)
                })
                global_chunk_idx += 1
                
        return chunks

    def chunk_all_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunks a collection of documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

if __name__ == "__main__":
    from retrieval.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    chunker = DocumentChunker(target_chunk_size=350, overlap=50)
    chunks = chunker.chunk_all_documents(docs)
    print(f"Total chunks created across {len(docs)} documents: {len(chunks)}")
    if chunks:
        print("\nSample Chunk 0:")
        print(f"ID: {chunks[0]['chunk_id']}")
        print(f"Source: {chunks[0]['source']} (Page {chunks[0]['page_number']})")
        print(f"Content: {chunks[0]['content']}")
