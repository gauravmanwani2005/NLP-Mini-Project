"""
Syntactic Analysis (POS Tagging) Module for EduMind.
Assigns grammatical parts of speech to tokens and interprets Penn Treebank tags.
"""

from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Penn Treebank human-readable description map
POS_DESCRIPTIONS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal auxiliary",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb"
}

def tag_pos(text: str) -> List[Tuple[str, str, str]]:
    """
    Performs POS tagging on input text.
    
    Args:
        text: Input string.
        
    Returns:
        List of tuples: (token, tag, human_readable_description)
    """
    tokens = word_tokenize(text)
    if not tokens:
        return []
    
    tagged = pos_tag(tokens)
    return [(word, tag, POS_DESCRIPTIONS.get(tag, "Punctuation / Symbol")) for word, tag in tagged]

def extract_syntactic_patterns(text: str) -> Dict[str, List[str]]:
    """Extracts nouns, verbs, modals, and WH-question markers from text."""
    tagged = tag_pos(text)
    nouns = [w for w, t, _ in tagged if t.startswith("NN")]
    verbs = [w for w, t, _ in tagged if t.startswith("VB")]
    modals = [w for w, t, _ in tagged if t == "MD"]
    wh_words = [w for w, t, _ in tagged if t.startswith("W")]
    adjectives = [w for w, t, _ in tagged if t.startswith("JJ")]
    
    return {
        "nouns": nouns,
        "verbs": verbs,
        "modals": modals,
        "wh_words": wh_words,
        "adjectives": adjectives
    }

if __name__ == "__main__":
    sample = "Students must submit assignments before the examination deadline."
    print("=== Syntactic Analysis (POS Tagging) Demo ===")
    tags = tag_pos(sample)
    for w, t, desc in tags:
        print(f"  {w:15} | {t:6} | {desc}")
        
    print("\nSyntactic Patterns:")
    patterns = extract_syntactic_patterns(sample)
    for k, v in patterns.items():
        print(f"  {k:12}: {v}")
