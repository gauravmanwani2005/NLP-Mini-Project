"""
Language Detection Module for EduMind Multilingual Layer.
Identifies English, Hindi, and Marathi queries using script analysis and n-gram models.
"""

import re
from typing import Dict, Any

try:
    from langdetect import detect, detect_langs
except ImportError:
    detect = None

DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')

# Common Marathi markers distinguishing from standard Hindi
MARATHI_MARKERS = ["आहे", "नाही", "कधी", "करावे", "हवी", "झाली", "मध्ये", "साठी", "परीक्षेसाठी", "किती"]
HINDI_MARKERS = ["है", "नहीं", "कब", "करना", "चाहिए", "हुआ", "में", "लिए", "परीक्षा", "कितनी"]

def detect_language(text: str) -> Dict[str, Any]:
    """
    Detects language of the input query: 'en', 'hi', or 'mr'.
    """
    clean_text = text.strip()
    if not clean_text:
        return {"language": "en", "confidence": 1.0, "script": "latin"}
        
    has_devanagari = bool(DEVANAGARI_RANGE.search(clean_text))
    
    if has_devanagari:
        # Check Marathi vs Hindi markers
        mr_matches = sum(1 for m in MARATHI_MARKERS if m in clean_text)
        hi_matches = sum(1 for m in HINDI_MARKERS if m in clean_text)
        
        if mr_matches > hi_matches:
            lang = "mr"
            name = "Marathi"
        else:
            lang = "hi"
            name = "Hindi"
            
        return {
            "language": lang,
            "language_name": name,
            "confidence": 0.95,
            "script": "devanagari"
        }
        
    if detect:
        try:
            detected = detect(clean_text)
            if detected == "mr":
                name = "Marathi"
            elif detected in ["hi", "ur"]:
                name = "Hindi"
            else:
                name = "English"
                detected = "en"
            return {"language": detected, "language_name": name, "confidence": 0.90, "script": "latin"}
        except Exception:
            pass
            
    return {"language": "en", "language_name": "English", "confidence": 1.0, "script": "latin"}

if __name__ == "__main__":
    tests = [
        "What is the minimum attendance required?",
        "मुझे परीक्षा फॉर्म कब भरना है?",
        "परीक्षेसाठी किती उपस्थिती आवश्यक आहे?"
    ]
    for t in tests:
        res = detect_language(t)
        print(f"'{t}' -> {res['language_name']} ({res['language']})")
