import fitz
import os

def check_pdf_text():
    path = 'data/raw/employee_handbook.pdf'
    doc = fitz.open(path)
    raw_text = "".join([page.get_text() for page in doc])
    print(f"Total Length: {len(raw_text)}")
    print("--- FIRST 500 CHARS ---")
    print(repr(raw_text[:500]))
    
    sections = raw_text.split('\n\n')
    print(f"\nNumber of '\\n\\n' splits: {len(sections)}")
    
    sections_single = raw_text.split('\n')
    print(f"Number of '\\n' splits: {len(sections_single)}")
    
    doc.close()

if __name__ == "__main__":
    check_pdf_text()
