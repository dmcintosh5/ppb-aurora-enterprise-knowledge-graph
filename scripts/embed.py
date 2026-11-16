import json
import os
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Pull values from the environment variables you set in docker-compose
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "https://gpu-01.insight.gsu.edu:11443")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

PROCESSED_PATH = os.path.join('data', 'processed')

# Headers for GSU ARC Authentication
headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

# Initialize the remote client instead of a local model
embeddings_client = OllamaEmbeddings(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    client_kwargs={"headers": headers}
)

def generate_embeddings():
    """Generates vector embeddings for processed chunks in batches."""
    batch_size = 5  # Optimal for 16GB RAM / i7 Processor

    for filename in os.listdir(PROCESSED_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(PROCESSED_PATH, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Skip if embeddings already exist or if it's not a chunk-based file (like graph adjacency)
            if 'chunks' not in data:
                print(f"[*] Skipping {filename}: No 'chunks' key found.")
                continue
                
            if 'embeddings' in data and len(data['embeddings']) == len(data['chunks']):
                print(f"[*] Skipping {filename}: Embeddings already exist and match chunk count.")
                continue

            chunks = data['chunks']
            all_embeddings = []

            print(f"[*] Batch-embedding {len(chunks)} chunks for {filename}...")
            
            # Process chunks in small batches to prevent thermal throttling
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                # Sending smaller requests prevents the Ollama service from timing out
                batch_vectors = embeddings_client.embed_documents(batch)
                all_embeddings.extend(batch_vectors)
                
            data['embeddings'] = all_embeddings
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"[+] Embeddings completed for {filename}")

if __name__ == "__main__":
    generate_embeddings()