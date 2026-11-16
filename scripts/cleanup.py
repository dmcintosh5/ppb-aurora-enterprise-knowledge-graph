import os
import shutil
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

def cleanup():
    """Deletes temporary directory files and clears associated data in Neo4j."""
    # 1. Delete files
    files_to_delete = [
        os.path.join('data', 'raw', 'directory.json'),
        os.path.join('data', 'processed', 'directory.json'),
        os.path.join('data', 'processed', 'directory.json.json') # Just in case of double extension
    ]
    
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"[+] Deleted file: {f}")
        else:
            print(f"[*] File not found (already gone?): {f}")

    # 2. Clean Neo4j
    print(f"[*] Connecting to Neo4j at {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            # Delete Document "directory.json" and its chunks
            print("[*] Deleting Document 'directory.json' and its Chunks...")
            session.run("""
                MATCH (d:Document {name: 'directory.json'})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE d, c
            """)
            
            # Since we don't know which Employees came from which file easily,
            # and the user wants to use the 'final' directory, 
            # it might be best to wipe all Employee nodes and re-upload them from the final file.
            # But let's first see if there are any other 'directory' documents.
            
            print("[*] Neo4j cleanup complete for 'directory.json'.")
        driver.close()
    except Exception as e:
        print(f"[!] Neo4j Error during cleanup: {e}")

if __name__ == "__main__":
    cleanup()
