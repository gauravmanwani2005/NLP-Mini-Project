"""
Document Loader Module for EduMind.
Extracts clean text and structural metadata from TXT, PDF, and Markdown academic circulars.
"""

import os
from typing import List, Dict, Any
import pypdf

class DocumentLoader:
    def __init__(self, doc_dir: str = "data/raw/documents"):
        self.doc_dir = doc_dir

    def load_txt(self, filepath: str) -> Dict[str, Any]:
        """Loads a text document with basic paragraph identification."""
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        return {
            "source": filename,
            "filepath": filepath,
            "file_type": "txt",
            "content": content,
            "pages": [{"page_number": 1, "text": content}]
        }

    def load_pdf(self, filepath: str) -> Dict[str, Any]:
        """Extracts text page-by-page from a PDF using pypdf."""
        filename = os.path.basename(filepath)
        reader = pypdf.PdfReader(filepath)
        pages = []
        full_text = []
        
        for idx, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page_number": idx, "text": text.strip()})
            full_text.append(text.strip())
            
        return {
            "source": filename,
            "filepath": filepath,
            "file_type": "pdf",
            "content": "\n\n".join(full_text),
            "pages": pages
        }

    def load_file(self, filepath: str) -> Dict[str, Any]:
        """Loads any supported academic document format."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return self.load_pdf(filepath)
        else:
            return self.load_txt(filepath)

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Scans the documents directory and loads all supported files."""
        documents = []
        if not os.path.exists(self.doc_dir):
            return documents
            
        for root, _, files in os.walk(self.doc_dir):
            for file in sorted(files):
                if file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in [".txt", ".pdf", ".md"]:
                    path = os.path.join(root, file)
                    try:
                        doc = self.load_file(path)
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        
        return documents

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    print(f"Loaded {len(docs)} documents from data/raw/documents:")
    for d in docs:
        print(f" - {d['source']} ({d['file_type']}, {len(d['content'])} chars)")
