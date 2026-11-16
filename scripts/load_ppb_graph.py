import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

def load_graph_to_neo4j():
    input_file = 'data/processed/ppb_graph_adjacency.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run transform_to_graph.py first.")
        return

    with open(input_file, 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    print(f"[*] Loading {len(nodes)} nodes and {len(edges)} edges into Neo4j...")

    with driver.session() as session:
        # 1. Load Nodes
        for node in nodes:
            label = node['label']
            node_id = node['id']
            properties = node.get('properties', {})
            
            # Use APOC or dynamic Cypher for labels if needed, but here labels are fixed
            # We use a map of parameters for properties
            query = f"MERGE (n:{label} {{id: $node_id}}) SET n += $properties"
            session.run(query, node_id=node_id, properties=properties)

        # 2. Load Edges
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            rel_type = edge['type']
            properties = edge.get('properties', {})

            # We need to find the source and target nodes. 
            # Since we don't know their labels here easily without a lookup, 
            # and Neo4j MERGE/MATCH requires labels for performance, 
            # but we can use a generic 'Node' label or just match by id if we indexed it.
            # Best practice: ensure all nodes have a common property like 'id'.
            
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            """
            session.run(query, source_id=source_id, target_id=target_id, properties=properties)

        # 3. Link Structural Graph to Data Layer (Guideline -> Data Store)
        print("[*] Linking entities to source document chunks...")
        link_query = """
        MATCH (e:Employee)
        MATCH (d:Document {name: 'ppb_directory_v7_historical.json'})-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text CONTAINS e.id
        MERGE (e)-[:HAS_SOURCE_DATA]->(c)
        """
        session.run(link_query)

    driver.close()
    print("[+] Knowledge Graph loaded into Neo4j successfully.")

if __name__ == "__main__":
    load_graph_to_neo4j()
