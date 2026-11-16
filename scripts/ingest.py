"""
This script serves as the Document Processing Layer of your architecture:
- Format Handling: Ingests PDFs, JSON, and TXT files.
- NLP Processing: Performs tokenization, stop-word removal, and punctuation cleaning.
- Metadata Attachment: Extracts author, title, and creation dates from file headers.
- Structured Output: Packages data for the Knowledge Graph and Vector store.
"""

import os
import json
import fitz  # PyMuPDF
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Configuration
RAW_DATA_PATH = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed')

def preprocess_text(text):
    """NLP Cleanup: Tokenizes and removes stop-words/punctuation."""
    # 1. Tokenization
    tokens = word_tokenize(text.lower())
    
    # 2. Stop-word and Punctuation removal
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    cleaned_tokens = [
        w for w in tokens 
        if w not in stop_words and w not in punctuation and w.isalnum()
    ]
    
    return " ".join(cleaned_tokens)

def semantic_chunker(text, max_chars=1000, overlap=200):
    """
    Robust chunker that splits by double-newlines first, 
    but falls back to character limits with overlap if sections are too large.
    """
    sections = text.split('\n\n')
    chunks = []
    
    for section in sections:
        section = section.strip()
        if not section: continue
        
        # If section is small enough, keep it whole
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Recursive split large sections by character count with overlap
            start = 0
            while start < len(section):
                end = start + max_chars
                chunk = section[start:end]
                chunks.append(chunk)
                start += (max_chars - overlap)
                
    return chunks

def extract_metadata(file_path):
    """Extracts internal document metadata based on file type."""
    filename = os.path.basename(file_path)
    metadata = {
        "source": filename,
        "company": "Peachtree Pantry & Beverage Co.",
        "ingested_at": datetime.now().isoformat(),
        "author": "Unknown",
        "title": filename,
        "date_created": "Unknown"
    }

    if filename.endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            doc_meta = doc.metadata
            metadata["author"] = doc_meta.get("author") or "Unknown"
            metadata["title"] = doc_meta.get("title") or filename
            metadata["date_created"] = doc_meta.get("creationDate") or "Unknown"
            doc.close()
        except Exception as e:
            print(f"[!] Metadata extraction error for {filename}: {e}")
            
    return metadata

def process_corpus():
    """Ingests raw files, extracts metadata, chunks text, and saves processed JSON."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    for filename in os.listdir(RAW_DATA_PATH):
        file_path = os.path.join(RAW_DATA_PATH, filename)
        full_text = ""
        doc_type = "unstructured"

        # --- FORMAT HANDLING ---
        if filename.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                full_text += page.get_text()
            doc.close()
        
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

        elif filename.endswith(".json"):
            doc_type = "structured"
            with open(file_path, 'r') as f:
                content = json.load(f)
            # Use whole content for JSON, or specific logic for directory.json
            if "employees" in content:
                chunks = [json.dumps(emp) for emp in content["employees"]]
            else:
                chunks = [json.dumps(content)]
            full_text = json.dumps(content) # For cleaning
        else:
            continue # Unsupported format

        # --- SAVE OUTPUT ---
        metadata = extract_metadata(file_path)
        metadata["type"] = doc_type
        if doc_type == "structured":
            metadata["count"] = len(chunks)
        
        # Clean text (stored separately to preserve embedding context)
        cleaned_text_sample = preprocess_text(full_text[:2000]) # Sample for metadata
        metadata["keywords"] = list(set(cleaned_text_sample.split()))[:10]

        # Chunk the original text for Vector Embedding (preserves LLM context)
        if doc_type == "unstructured":
            chunks = semantic_chunker(full_text)

        output = {
            "metadata": metadata,
            "chunks": chunks,
            "cleaned_chunks": [preprocess_text(c) for c in chunks] # NLP Requirement
        }
        
        # Avoid double .json extension
        base_name = filename if not filename.lower().endswith(".json") else filename[:-5]
        extension = ".pdf.json" if filename.lower().endswith(".pdf") else ".json"
        save_path = os.path.join(PROCESSED_DATA_PATH, f"{base_name}{extension}")
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"[+] Processed, Cleaned, and Saved: {filename}")

if __name__ == "__main__":
    process_corpus()
