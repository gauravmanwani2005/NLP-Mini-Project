"""
Cross-Lingual Translation & Routing Module for EduMind.
Enables Hindi & Marathi academic queries to be translated and routed
into the semantic search pipeline, returning translated responses.
"""

from typing import Dict, Any
from multilingual.language_detection import detect_language

# Domain Lexicon for Indian College Context
TERM_DICTIONARY = {
    # Hindi
    "परीक्षा": "examination",
    "परीक्षा फॉर्म": "exam registration form",
    "अटेंडेंस": "attendance",
    "उपस्थिति": "attendance",
    "फीस": "fees tuition",
    "शुल्क": "fees tuition",
    "छात्रवृत्ति": "scholarship",
    "हॉस्टल": "hostel accommodation",
    "रिजल्ट": "examination results marks",
    "परिणाम": "results",
    "लाइब्रेरी": "library books",
    "ग्रंथालय": "library books",
    "प्लेसमेंट": "campus placements recruitment",
    "प्रवेश": "admission cutoff",
    "टाइमटेबल": "timetable schedule",
    "असाइनमेंट": "assignments submission",
    "कब": "when is deadline schedule",
    "कितना": "how much minimum percentage",
    "कितनी": "how much minimum percentage",
    "कहाँ": "where portal link",
    "नियम": "rules policy regulation",
    
    # Marathi
    "परीक्षेसाठी": "for examination",
    "उपस्थिती": "attendance requirement",
    "कधी": "when deadline schedule",
    "किती": "how much minimum",
    "हवी": "required needed",
    "नोंदणी": "registration",
    "प्रक्रिया": "process procedure",
    "निकालाची": "results announcement",
    "तारीख": "date schedule"
}

def translate_query_to_english(query: str, lang: str) -> str:
    """
    Translates or enriches non-English query terms into English equivalents
    for dense vector search against the English policy corpus.
    """
    if lang == "en":
        return query
        
    translated_tokens = []
    text_lower = query.lower()
    
    matched_terms = []
    for non_en_term, en_term in TERM_DICTIONARY.items():
        if non_en_term in text_lower:
            matched_terms.append(en_term)
            
    if matched_terms:
        enriched_query = f"{query} {' '.join(matched_terms)}"
        return enriched_query
    return query

def format_multilingual_response(answer: str, lang: str) -> str:
    """
    Formats the final response with language acknowledgment and structured guidance.
    """
    if lang == "hi":
        prefix = "🇮🇳 **[हिंदी अनुवाद / Hindi Response Summary]**\n\n"
        return f"{prefix}{answer}\n\n*(नोट: आधिकारिक नियम पुस्तिका अंग्रेजी में मान्य है।)*"
    elif lang == "mr":
        prefix = "🚩 **[मराठी सारांश / Marathi Response Summary]**\n\n"
        return f"{prefix}{answer}\n\n*(टीप: अधिकृत महाविद्यालयीन नियम पुस्तिका इंग्रजीमध्ये उपलब्ध आहे.)*"
    return answer

if __name__ == "__main__":
    test_hi = "मुझे परीक्षा फॉर्म कब भरना है?"
    lang_info = detect_language(test_hi)
    en_q = translate_query_to_english(test_hi, lang_info["language"])
    print(f"Original: {test_hi}")
    print(f"Detected: {lang_info['language_name']}")
    print(f"Enriched for retrieval: {en_q}")
