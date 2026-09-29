"""
Tokenization Module for EduMind.
Splits text into discrete linguistic units (tokens) handling punctuation and contraction rules.
"""

from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

def tokenize_words(text: str, remove_punct: bool = False, to_lower: bool = False) -> List[str]:
    """
    Tokenizes input string into a list of word tokens.
    
    Args:
        text: String to tokenize.
        remove_punct: Whether to strip pure punctuation tokens.
        to_lower: Whether to lowercase each token.
        
    Returns:
        List of token strings.
    """
    if not text or not text.strip():
        return []
    
    tokens = word_tokenize(text.strip())
    
    if to_lower:
        tokens = [t.lower() for t in tokens]
        
    if remove_punct:
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]
        
    return tokens

def analyze_tokens(text: str) -> Dict[str, Any]:
    """Computes basic lexical statistics for a given text."""
    raw_tokens = tokenize_words(text, remove_punct=False, to_lower=False)
    cleaned_tokens = tokenize_words(text, remove_punct=True, to_lower=True)
    vocab = set(cleaned_tokens)
    ttr = (len(vocab) / len(cleaned_tokens)) if cleaned_tokens else 0.0
    
    return {
        "raw_tokens": raw_tokens,
        "token_count": len(raw_tokens),
        "cleaned_tokens": cleaned_tokens,
        "vocab_size": len(vocab),
        "type_token_ratio": round(ttr, 4)
    }

if __name__ == "__main__":
    sample = "Students must carry their ID cards, or they won't be permitted into the exam hall!"
    print("=== Tokenization Demo ===")
    analysis = analyze_tokens(sample)
    print(f"Sample: {sample}")
    print(f"Raw tokens ({analysis['token_count']}): {analysis['raw_tokens']}")
    print(f"Cleaned words ({len(analysis['cleaned_tokens'])}): {analysis['cleaned_tokens']}")
    print(f"Type-Token Ratio (Lexical Diversity): {analysis['type_token_ratio']}")
