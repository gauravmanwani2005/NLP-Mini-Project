"""
Prompt Engineering and Guardrails Module for EduMind RAG.
Constructs strict context-grounded prompts preventing hallucination.
"""

from typing import List, Dict, Any

RAG_SYSTEM_PROMPT = """You are EduMind, the official AI Academic Knowledge Assistant for the college.
Your task is to answer the student's question accurately using ONLY the provided official college context passages.

CRITICAL RULES:
1. Ground your response strictly in the retrieved context below.
2. If the context does not contain sufficient details to answer the query, reply with:
   "Based on the official college documents provided, this specific information is not currently available. Please contact the department academic office or examination cell for further assistance."
3. Cite the document name and section/page when citing policies or numerical figures (e.g., percentages, deadlines, fees).
4. Be clear, polite, and academically professional.
"""

def build_rag_prompt(query: str, retrieved_context: str, predicted_intent: str = None) -> str:
    """
    Builds the complete grounded RAG prompt.
    """
    intent_line = f"Detected Query Intent: {predicted_intent}\n" if predicted_intent else ""
    
    prompt = f"""{RAG_SYSTEM_PROMPT}

{intent_line}
=== RETRIEVED ACADEMIC CONTEXT ===
{retrieved_context}
==================================

Student Question:
{query}

Official Grounded Answer:"""
    return prompt

if __name__ == "__main__":
    sample_ctx = "[Source 1: attendance_policy.txt (Page 1)]\nMinimum attendance is 75% for theory and 80% for labs."
    p = build_rag_prompt("What attendance do I need for labs?", sample_ctx, "attendance")
    print(p)
