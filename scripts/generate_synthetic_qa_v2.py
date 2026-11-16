import os
import json
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "https://gpu-01.insight.gsu.edu:11443")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
SMART_LLM_MODEL = os.getenv("SMART_LLM_MODEL", "llama3.1:latest")
PROCESSED_DATA_PATH = os.path.join('data', 'processed')
OUTPUT_FILE = os.path.join('tests', 'fixtures', 'retrieval_gold_set.json')

headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
llm = ChatOllama(model=SMART_LLM_MODEL, base_url=OLLAMA_URL, headers=headers)

def generate_factoid_qa(chunks, source_name, num_queries=5):
    print(f"[*] Generating {num_queries} QA pairs for {source_name}...")
    
    # Randomly sample chunks to generate questions from
    import random
    sampled_indices = random.sample(range(len(chunks)), min(num_queries, len(chunks)))
    
    qa_pairs = []
    for idx in sampled_indices:
        chunk_text = chunks[idx]
        
        prompt = f"""
        Based on the following HR document chunk from "{source_name}", 
        generate a realistic question an employee would ask that is answered by this text.
        
        CHUNK:
        {chunk_text}
        
        Return ONLY a JSON object with "question" and "answer" fields.
        Example: {{"question": "How do I clock in?", "answer": "Use the POS system."}}
        """
        
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            # Extract JSON
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                qa = json.loads(match.group())
                qa_pairs.append({
                    "query_id": f"factoid_{source_name}_{idx}",
                    "category": "Factoid",
                    "query_text": qa["question"],
                    "expected_answer": qa["answer"],
                    "relevance_grades": {
                        f"{source_name}:chunk_{idx}": 2
                    }
                })
        except Exception as e:
            print(f"[!] Error generating QA for chunk {idx}: {e}")
            
    return qa_pairs

def main():
    if not os.path.exists(OUTPUT_FILE):
        gold_set = []
    else:
        with open(OUTPUT_FILE, 'r') as f:
            gold_set = json.load(f)

    files_to_process = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.endswith('.pdf.json')]
    
    for filename in files_to_process:
        file_path = os.path.join(PROCESSED_DATA_PATH, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        source_name = data['metadata']['source']
        chunks = data['chunks']
        
        qa_pairs = generate_factoid_qa(chunks, source_name, num_queries=10)
        gold_set.extend(qa_pairs)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(gold_set, f, indent=4)
    print(f"[+] Master Gold Set updated at {OUTPUT_FILE}. Total queries: {len(gold_set)}")

if __name__ == "__main__":
    main()
