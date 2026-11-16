import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ARCHITECT NOTE: This prioritizes the GitLab Service alias, then falls back to local
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

def upload_to_neo4j():
    """Uploads processed documents, metadata, and embeddings to Neo4j."""
    driver = GraphDatabase.driver(URI, auth=AUTH)
    processed_dir = 'data/processed'
    
    if not os.path.exists(processed_dir):
        print(f"[!] Processed directory {processed_dir} not found.")
        return

    with driver.session() as session:
        # 0. Create/Update Indexes if they don't exist
        print("[*] Ensuring Neo4j Indexes (Vector + Full-Text) exist...")
        session.run("""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 768,
             `vector.similarity_function`: 'cosine'
            }}
        """)
        session.run("CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
        session.run("CREATE FULLTEXT INDEX employee_name_index IF NOT EXISTS FOR (e:Employee) ON EACH [e.name]")

        for filename in os.listdir(processed_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(processed_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # SAFETY GUARD: Skip files without metadata (like the graph adjacency file)
                if 'metadata' not in data:
                    print(f"[*] Skipping {filename}: No document metadata found.")
                    continue
                    
                meta = data['metadata']
                source = meta['source']
                
                # SAFEGUARD: Skip if this is a PDF but has no embeddings. 
                if filename.endswith(".pdf.json") and 'embeddings' not in data:
                    print(f"[!] Skipping {filename}: Run embed.py first!")
                    continue

                print(f"[*] Uploading {source} with metadata to Neo4j...")
                
                # 1. Create/Update Document Node with full metadata
                session.run("""
                    MERGE (d:Document {name: $name})
                    SET d.author = $author,
                        d.title = $title,
                        d.date_created = $date_created,
                        d.company = $company,
                        d.ingested_at = $ingested_at,
                        d.type = $type,
                        d.keywords = $keywords
                """, 
                name=source, 
                author=meta.get('author', 'Unknown'),
                title=meta.get('title', source),
                date_created=meta.get('date_created', 'Unknown'),
                company=meta.get('company', 'Unknown'),
                ingested_at=meta.get('ingested_at', 'Unknown'),
                type=meta.get('type', 'unstructured'),
                keywords=meta.get('keywords', [])
                )
                
                # 2. Upload Chunks
                for i, chunk_text in enumerate(data['chunks']):
                    emb = data['embeddings'][i] if 'embeddings' in data else None
                    
                    # Graph schema: We establish a (Document)-[:HAS_CHUNK]->(Chunk) relationship. This allows for hybrid retrieval.
                    # We can traverse relationships. If we find a relevant chunk, we can hop to its parent document to pull metadata
                    # like author, date_created, or department without extra lookup.
                    session.run("""
                        MATCH (d:Document {name: $doc_name})
                        CREATE (c:Chunk {index: $idx, text: $text, embedding: $emb})
                        CREATE (d)-[:HAS_CHUNK]->(c)
                    """, doc_name=source, idx=i, text=chunk_text, emb=emb)
                    
    driver.close()
    print("SUCCESS: Data and Metadata persisted to Neo4j.")

if __name__ == "__main__":
    upload_to_neo4j()
