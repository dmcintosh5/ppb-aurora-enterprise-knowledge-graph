from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv(override=True)

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

def debug():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        print("--- Documents ---")
        res = session.run("MATCH (d:Document) RETURN d.name as name")
        for r in res:
            print(f"- {r['name']}")
        
        print("\n--- Employee Count ---")
        res = session.run("MATCH (e:Employee) RETURN count(e) as count")
        print(f"Count: {res.single()['count']}")

        print("\n--- Top 5 Employees ---")
        res = session.run("MATCH (e:Employee) RETURN e.name as name, e.title as title LIMIT 5")
        for r in res:
            print(f"- {r['name']} ({r['title']})")
            
    driver.close()

if __name__ == "__main__":
    debug()
