"""
Generates an official formatted PDF report for the EduMind NLP Mini-Project using ReportLab.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf(output_path="report/NLP_Mini_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E1B4B'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4338CA'),
        alignment=1,
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E1B4B'),
        spaceBefore=14,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    story = []

    # Title Banner
    story.append(Paragraph("EDUMIND: ACADEMIC KNOWLEDGE ASSISTANT", title_style))
    story.append(Paragraph("NLP Mini-Project Specification & Empirical Evaluation Report<br/>Fine-Tuned Transformers • Semantic Retrieval • RAG • WordNet Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4338CA'), spaceAfter=15))

    # Abstract
    story.append(Paragraph("1. Executive Summary & Abstract", h1_style))
    story.append(Paragraph(
        "EduMind is an intelligent multilingual academic knowledge assistant designed to solve the lexical gap "
        "between formal institutional circulars and natural student queries. The system incorporates all five mandatory "
        "NLP preprocessing stages, domain-adapted BERT intent classification (achieving 98.76% accuracy), FAISS dense semantic "
        "search (0.8750 MRR), anti-hallucination grounded RAG generation, text summarization, and Indian language routing (Hindi/Marathi).",
        body_style
    ))

    # Preprocessing
    story.append(Paragraph("2. Mandatory Classical NLP Preprocessing", h1_style))
    story.append(Paragraph("• <b>Sentence Boundary Identification:</b> NLTK Punkt model segmenting raw documents into sentence units.", bullet_style))
    story.append(Paragraph("• <b>Tokenization & Lexical Stats:</b> Word-level tokenization computing lexical diversity (Type-Token Ratio).", bullet_style))
    story.append(Paragraph("• <b>Lemmatization vs Stemming:</b> WordNetLemmatizer with POS tag mapping vs heuristic Porter Stemmer.", bullet_style))
    story.append(Paragraph("• <b>Syntactic Analysis (POS Tagging):</b> Penn Treebank tag assignment (NN, VB, MD, JJ, WRB).", bullet_style))
    story.append(Paragraph("• <b>Semantic Analysis (WordNet):</b> Synset hierarchies, hypernyms, and Path/Wu-Palmer similarity metrics.", bullet_style))

    # Model Evaluation Table
    story.append(Paragraph("3. Model Comparison Benchmark (Section 26)", h1_style))
    table_data = [
        ["Model Architecture", "Type", "Accuracy", "Weighted F1", "Macro F1"],
        ["TF-IDF + Logistic Reg.", "Classical ML", "98.35%", "98.39%", "98.39%"],
        ["Pretrained BERT (Untrained)", "Zero-Shot", "8.50%", "8.08%", "8.08%"],
        ["Fine-Tuned BERT (EduMind)", "Fine-Tuned Transformer", "98.76%", "98.76%", "98.65%"]
    ]
    t = Table(table_data, colWidths=[150, 110, 80, 80, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#312E81')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Retrieval Benchmark Table
    story.append(Paragraph("4. Information Retrieval Evaluation (Section 27)", h1_style))
    ret_data = [
        ["Retrieval Method", "Precision@1", "Precision@3", "Recall@3", "MRR", "Latency"],
        ["Dense Semantic (FAISS + MiniLM)", "83.33%", "72.22%", "83.33%", "0.8750", "45.19 ms"],
        ["Sparse Keyword (TF-IDF)", "91.67%", "83.33%", "91.67%", "0.9583", "0.23 ms"],
        ["Hybrid RRF (Dense + Keyword)", "91.67%", "86.11%", "100.0%", "0.9583", "46.10 ms"]
    ]
    t_ret = Table(ret_data, colWidths=[160, 75, 75, 70, 60, 60])
    t_ret.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#065F46')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F0FDF4'), colors.white]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_ret)
    story.append(Spacer(1, 12))

    # Generation Quality Metrics
    story.append(Paragraph("5. Generation Quality (ROUGE & BERTScore)", h1_style))
    story.append(Paragraph("• <b>ROUGE-1 F1:</b> 0.2775 | <b>ROUGE-2 F1:</b> 0.2003 | <b>ROUGE-L F1:</b> 0.2545", bullet_style))
    story.append(Paragraph("• <b>BERTScore (Contextual Token Embeddings):</b> Precision: 0.4598 | Recall: 0.7705 | F1: 0.5754", bullet_style))
    story.append(Paragraph(
        "<i>Note on BERTScore:</i> Per Section 29 of the specification, BERTScore is applied specifically to evaluate grounded answers "
        "and summaries against ground truth reference text, capturing contextual semantic meaning rather than mere n-gram overlap.",
        body_style
    ))

    # Viva Questions
    story.append(Paragraph("6. Key Viva Voce Concepts", h1_style))
    story.append(Paragraph("<b>Q1: Why fine-tune BERT for intent classification?</b><br/>"
                           "Fine-tuning adapts the transformer self-attention representations specifically to institutional vocabulary, "
                           "providing deterministic intent classification with low latency and zero external API dependencies.", body_style))
    story.append(Paragraph("<b>Q2: What is the benefit of RAG over vanilla LLM prompting?</b><br/>"
                           "RAG grounds the answer strictly in retrieved official passages with exact page citations, eliminating hallucinated policies.", body_style))

    doc.build(story)
    print(f"Generated PDF report: {output_path}")

if __name__ == "__main__":
    generate_pdf()
