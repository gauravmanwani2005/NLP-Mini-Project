"""
Sentence Boundary Identification Module for EduMind.
Demonstrates classical NLP sentence segmentation using NLTK and regex-based boundaries.
"""

from typing import List
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

def segment_sentences(text: str) -> List[str]:
    """
    Identifies sentence boundaries in a text and returns a list of sentences.
    
    Args:
        text: Input raw string (single paragraph or multi-sentence document).
        
    Returns:
        List of segmented sentences with leading/trailing whitespace stripped.
    """
    if not text or not text.strip():
        return []
    
    # NLTK punkt unsupervised sentence boundary detection
    sentences = sent_tokenize(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def format_sentence_segmentation(text: str) -> str:
    """Returns a formatted inspection string for sentence segmentation."""
    sentences = segment_sentences(text)
    output = [f"Input length: {len(text)} chars | Identified sentences: {len(sentences)}"]
    for i, s in enumerate(sentences, 1):
        output.append(f"  [{i}] {s}")
    return "\n".join(output)

if __name__ == "__main__":
    sample = (
        "The examination starts on Monday. Students must carry their ID cards! "
        "Dr. Sharma announced that hall tickets are ready at exams.edumind.ac.in. "
        "Is attendance mandatory? Yes, minimum 75% is required."
    )
    print("=== Sentence Boundary Identification Demo ===")
    print(format_sentence_segmentation(sample))
