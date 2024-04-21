#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline

nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def query_pdf_with_llm(pdf_text, question):
    result = nlp(question=question, context=pdf_text)
    return result["answer"]

if __name__ == "__main__":
    # Example usage
    pdf_file_path = "example.pdf"
    query = input()

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_file_path)

    # Query the LLM
    answer = query_pdf_with_llm(pdf_text, query)
    print(f"Answer: {answer}")


# In[ ]:




