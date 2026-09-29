"""
Lemmatization Module for EduMind.
Reduces inflected words to their canonical dictionary base form (lemma) using WordNet.
Also provides comparison with Porter Stemmer for viva demonstration.
"""

from typing import List, Tuple, Dict
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def _get_wordnet_pos(treebank_tag: str):
    """Maps Penn Treebank POS tag to WordNet POS constant."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text: str, pos_guided: bool = True) -> List[Tuple[str, str]]:
    """
    Lemmatizes tokens in text.
    
    Args:
        text: Input string.
        pos_guided: If True, uses POS tags to select the appropriate lemma.
        
    Returns:
        List of (token, lemma) tuples.
    """
    tokens = word_tokenize(text)
    if not tokens:
        return []
    
    results = []
    if pos_guided:
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            wn_pos = _get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            results.append((word, lemma))
    else:
        for word in tokens:
            lemma = lemmatizer.lemmatize(word.lower())
            results.append((word, lemma))
            
    return results

def compare_stemming_and_lemmatization(words: List[str]) -> List[Dict[str, str]]:
    """
    Compares Porter Stemmer vs WordNet Lemmatizer on a list of words.
    Crucial for academic evaluation and viva questions.
    """
    comparison = []
    for w in words:
        stem = stemmer.stem(w)
        lemma_noun = lemmatizer.lemmatize(w.lower(), pos=wordnet.NOUN)
        lemma_verb = lemmatizer.lemmatize(w.lower(), pos=wordnet.VERB)
        comparison.append({
            "original": w,
            "porter_stem": stem,
            "lemma_as_noun": lemma_noun,
            "lemma_as_verb": lemma_verb
        })
    return comparison

if __name__ == "__main__":
    sample_text = "The students were submitting examinations and studying various courses."
    print("=== Lemmatization Demo ===")
    pairs = lemmatize_text(sample_text)
    for token, lemma in pairs:
        print(f"  {token:15} -> {lemma}")
        
    print("\n=== Stemming vs Lemmatization Comparison ===")
    demo_words = ["studies", "studying", "corpora", "examinations", "better", "leaves"]
    comp = compare_stemming_and_lemmatization(demo_words)
    for c in comp:
        print(f"Word: {c['original']:12} | Stem: {c['porter_stem']:12} | Lemma(N): {c['lemma_as_noun']:12} | Lemma(V): {c['lemma_as_verb']}")
