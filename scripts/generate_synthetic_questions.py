import json
import os
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "https://gpu-01.insight.gsu.edu:11443")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama3.1:latest")
headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

def generate_questions_for_chunk(llm, chunk_text):
    prompt = f"""
    Given the following HR policy chunk, generate exactly 3 short, natural language questions that an employee might ask that are answered by this text.
    
    TEXT:
    {chunk_text}
    
    Format: Return ONLY a JSON list of strings.
    Example: ["What is the dress code for cooks?", "Do I need non-slip shoes?", "Are jeans allowed?"]
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        # Basic cleanup for JSON extraction
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"  [!] Error generating questions: {e}")
    return []

def augment_processed_data():
    processed_dir = 'data/processed'
    llm = ChatOllama(model=FAST_LLM_MODEL, base_url=OLLAMA_URL, headers=headers)
    
    # Only process PDFs (Policy files)
    target_files = [f for f in os.listdir(processed_dir) if f.endswith(".pdf.json")]
    
    print(f"[*] Starting Synthetic Question Augmentation for {len(target_files)} files...")
    
    for filename in target_files:
        path = os.path.join(processed_dir, filename)
        with open(path, 'r') as f:
            data = json.load(f)
            
        if 'synthetic_questions' in data:
            print(f"[*] Skipping {filename} (already augmented).")
            continue
            
        print(f"[*] Processing {filename} ({len(data['chunks'])} chunks)...")
        all_synthetic = []
        
        for i, chunk in enumerate(data['chunks']):
            print(f"  > Generating for chunk {i+1}/{len(data['chunks'])}...", end='\r')
            questions = generate_questions_for_chunk(llm, chunk)
            all_synthetic.append(questions)
            
        data['synthetic_questions'] = all_synthetic
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n[+] Successfully augmented {filename}")

if __name__ == "__main__":
    augment_processed_data()
