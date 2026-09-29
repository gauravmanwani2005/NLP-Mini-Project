"""
Semantic Analysis Module (WordNet) for EduMind.
Demonstrates synset extraction, lexical relations (hypernyms/hyponyms),
and semantic path similarity between academic concepts.
"""

from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import wordnet

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

def get_synsets_info(word: str) -> List[Dict[str, Any]]:
    """
    Retrieves synset definitions, lemmas, and examples for a word.
    
    Args:
        word: A single word string.
        
    Returns:
        List of dictionaries with synset details.
    """
    clean_word = word.strip().lower()
    synsets = wordnet.synsets(clean_word)
    
    results = []
    for s in synsets:
        hypernyms = [h.name().split('.')[0] for h in s.hypernyms()]
        hyponyms = [hp.name().split('.')[0] for hp in s.hyponyms()]
        
        results.append({
            "synset_name": s.name(),
            "pos": s.pos(),
            "definition": s.definition(),
            "examples": s.examples(),
            "lemmas": [l.name() for l in s.lemmas()],
            "hypernyms": hypernyms[:5],
            "hyponyms": hyponyms[:5]
        })
    return results

def compute_wordnet_similarity(word1: str, word2: str) -> Dict[str, Optional[float]]:
    """
    Computes semantic similarity metrics between two words using WordNet hierarchy.
    Calculates Path Similarity and Wu-Palmer (wup) Similarity.
    """
    syns1 = wordnet.synsets(word1.strip().lower())
    syns2 = wordnet.synsets(word2.strip().lower())
    
    if not syns1 or not syns2:
        return {"path_similarity": None, "wup_similarity": None, "best_syn1": None, "best_syn2": None}
    
    max_path = -1.0
    max_wup = -1.0
    best_pair = (syns1[0].name(), syns2[0].name())
    
    for s1 in syns1:
        for s2 in syns2:
            psim = s1.path_similarity(s2)
            wsim = s1.wup_similarity(s2)
            if psim is not None and psim > max_path:
                max_path = psim
                best_pair = (s1.name(), s2.name())
            if wsim is not None and wsim > max_wup:
                max_wup = wsim
                
    return {
        "path_similarity": round(max_path, 4) if max_path >= 0 else None,
        "wup_similarity": round(max_wup, 4) if max_wup >= 0 else None,
        "best_syn1": best_pair[0],
        "best_syn2": best_pair[1]
    }

def analyze_query_semantics(query: str) -> List[Dict[str, Any]]:
    """Runs semantic extraction for all non-trivial words in a query."""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
    tokens = word_tokenize(query)
    content_words = [w for w in tokens if w.isalnum() and w.lower() not in stop_words]
    
    analysis = []
    for w in content_words:
        info = get_synsets_info(w)
        if info:
            analysis.append({
                "word": w,
                "primary_synset": info[0]["synset_name"],
                "definition": info[0]["definition"],
                "synonyms": info[0]["lemmas"],
                "hypernyms": info[0]["hypernyms"]
            })
    return analysis

if __name__ == "__main__":
    print("=== WordNet Semantic Analysis Demo ===")
    word = "examination"
    syns = get_synsets_info(word)
    print(f"Synsets for '{word}': {len(syns)}")
    if syns:
        print(f"Primary Synset: {syns[0]['synset_name']}")
        print(f"Definition: {syns[0]['definition']}")
        print(f"Hypernyms: {syns[0]['hypernyms']}")
        
    print("\n=== Semantic Similarity Matrix ===")
    pairs = [
        ("examination", "test"),
        ("examination", "study"),
        ("student", "pupil"),
        ("student", "car"),
        ("professor", "teacher")
    ]
    for w1, w2 in pairs:
        sim = compute_wordnet_similarity(w1, w2)
        print(f"Sim({w1}, {w2}) -> Path Sim: {sim['path_similarity']} | Wu-Palmer: {sim['wup_similarity']}")
