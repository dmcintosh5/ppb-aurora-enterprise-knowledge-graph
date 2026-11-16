from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Force load from .env
load_dotenv(override=True)

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

def clear_db():
    print(f"[*] Connecting to {URI}...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        with driver.session() as session:
            print("[*] Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            # Also delete indexes if possible, or just let them stay
            print("[+] Database cleared successfully.")
        driver.close()
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    clear_db()
