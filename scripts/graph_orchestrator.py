import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import fitz  # PyMuPDF
import string
import nltk
import uuid
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j import GraphDatabase
from scripts.state import HRProjectState
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- CONFIGURATION ---
STATE_MAP = {
    "Georgia": "GA",
    "New York": "NY",
    "California": "CA",
    "Virginia": "VA"
}

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "https://gpu-01.insight.gsu.edu:11443")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# Task-Specific Models
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama3.1:latest")
SMART_LLM_MODEL = os.getenv("SMART_LLM_MODEL", "gemma3:latest")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))
RAW_DATA_PATH = os.path.join('data', 'raw')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONTOLOGY_PATH = os.path.join(BASE_DIR, 'data', 'ppb_hr_ontology.owl')

headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

# Initialize Clients
llm = ChatOllama(model=SMART_LLM_MODEL, base_url=OLLAMA_URL, headers=headers)
embeddings_client = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL, client_kwargs={"headers": headers})     
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# --- HELPERS ---
def preprocess_text(text):
    """NLP Cleanup for Milestone compliance."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned = [w for w in tokens if w not in set(string.punctuation) and w not in stop_words and w.isalnum()]
    return " ".join(cleaned)

def extract_metadata(file_path):
    """Extracts internal document properties and metadata."""
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
        except: pass
    return metadata

def semantic_chunker(text, chunk_size=500, overlap=100):
    """Chunks text into fixed sizes with overlap to ensure Gold Set alignment."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text):
            break
    return chunks

# --- NODE IMPLEMENTATION (The Agents) ---

def router(state: HRProjectState) -> HRProjectState:
    """Ingests files, performs NLP cleaning, and routes to specialist agents."""
    print(f"[*] Node: Router - Ingesting and Cleaning {state['doc_name']}")
    file_path = state['doc_path']
    filename = state['doc_name'].lower()
    raw_text = ""

    if filename.endswith(".json"):
        with open(file_path, 'r') as f:
            content = json.load(f)

        if "directory" in filename:
            chunks = [json.dumps(emp) for emp in content.get("employees", [])]
            doc_type = "directory"
            next_node = "contact_specialist"
        else:
            chunks = [json.dumps(content)]
            doc_type = "unstructured"
            next_node = END

    elif filename.endswith(".csv"):
        # Handle the Jurisdictional Specifics CSV with encoding resilience
        import pandas as pd
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='utf-16')
            except:
                df = pd.read_csv(file_path, encoding='latin1')
        
        chunks = [json.dumps(row) for row in df.to_dict('records')]
        doc_type = "mandate"
        next_node = "embedder" # Ensure it flows to Graph Builder

    elif filename.endswith(".pdf"):
        doc = fitz.open(file_path)
        raw_text = "".join([page.get_text() for page in doc])
        doc.close()
        chunks = semantic_chunker(raw_text)

        if "handbook" in filename:
            next_node = "policy_specialist"
            doc_type = "policy"
        elif "sop" in filename:
            next_node = "process_specialist"
            doc_type = "process"
        else:
            next_node = END
            doc_type = "unstructured"
    else:
        chunks = []
        doc_type = "unknown"
        next_node = END

    return {
        **state,
        "raw_text": raw_text,
        "chunks": chunks,
        "next_node": next_node,
        "doc_type": doc_type,
        "metadata": extract_metadata(file_path)
    }

def policy_specialist(state: HRProjectState) -> HRProjectState:
    """Extracts regional overrides and versioning."""
    print(f"[*] Node: Agent B (Policy Specialist) - Running LLM extraction")
    context = "\n---\n".join(state['chunks'][:5]) # Increased context for regional discovery
    prompt = f"""Extract HR Policy data as JSON.
    Identify: 1. Applicable Regions (GA, NY, VA). 2. Version ID. 3. Effective Date. 4. One-sentence summary.
    Context: {context}"""

    try:
        response = llm.invoke(prompt)
        state['metadata']['summary'] = response.content
        return {**state, "entities": [{"type": "Policy", "data": response.content}], "next_node": "embedder"}
    except Exception as e:
        print(f"[!] LLM Error: {e}")
        return {**state, "errors": [str(e)], "next_node": "embedder"}

def process_specialist(state: HRProjectState) -> HRProjectState:
    """Extracts sequential logic and form requirements."""
    print(f"[*] Node: Agent A (Process Specialist) - Running LLM extraction")
    context = "\n---\n".join(state['chunks'][:5])
    prompt = f"Identify the sequential steps, a one-sentence summary, and any required forms (e.g., Form 102-B) for this process:\n{context}"

    try:
        response = llm.invoke(prompt)
        state['metadata']['summary'] = response.content
        return {**state, "entities": [{"type": "Process", "data": response.content}], "next_node": "embedder"}
    except Exception as e:
        print(f"[!] LLM Error: {e}")
        return {**state, "errors": [str(e)], "next_node": "embedder"}

def contact_specialist(state: HRProjectState) -> HRProjectState:
    """Processes organizational directory data for employee extraction."""
    print(f"[*] Node: Agent C (Contact Specialist) - Normalizing Directory Data")
    return {**state, "entities": [{"type": "Directory", "status": "processed"}], "next_node": "embedder"}

def embedder(state: HRProjectState) -> HRProjectState:
    """Generates vector embeddings for document chunks with task-specific prefixes."""
    if not state.get('chunks'): return {**state, "next_node": END}
    print(f"[*] Node: Embedder - Generating vectors for {len(state['chunks'])} chunks")
    try:
        # Nomic-embed-text performs better with task-specific prefixes
        prefixed_chunks = [f"search_document: {c}" for c in state['chunks']]
        embeddings = embeddings_client.embed_documents(prefixed_chunks)
        return {**state, "embeddings": embeddings, "embeddings_generated": True, "next_node": "graph_builder"}
    except Exception as e:
        print(f"[!] Embedding Error: {e}")
        return {**state, "errors": [str(e)], "embeddings_generated": False, "next_node": END}

def graph_builder(state: HRProjectState) -> HRProjectState:
    """Persists document nodes, chunks, and ontology-aligned entities to Neo4j."""
    if not state.get('embeddings'): return {**state, "neo4j_status": "error", "next_node": END}
    print(f"[*] Node: Agent D (Graph Builder) - Building Ontology-Aligned Graph")

    doc_name = state['doc_name']
    meta = state['metadata']
    chunks = state['chunks']
    embeddings = state['embeddings']
    doc_type = state.get('doc_type', 'unstructured')

    try:
        with neo4j_driver.session() as session:
            # 1. Ensure Indexes
            session.run("CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}")
            session.run("CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
            session.run("CREATE CONSTRAINT employee_id_unique IF NOT EXISTS FOR (e:Employee) REQUIRE e.employee_id IS UNIQUE")
            
            # 2. Create/Update Document Node
            session.run("""
                MERGE (d:Document {name: $name})
                SET d.author = $author, d.title = $title, d.date_created = $date_created,
                    d.company = $company, d.ingested_at = $ingested_at,
                    d.summary = $summary, d.type = $type
            """,
            name=doc_name, author=meta['author'], title=meta['title'],
            date_created=meta['date_created'], company=meta['company'],
            ingested_at=meta['ingested_at'], summary=meta.get('summary', 'N/A'),
            type=doc_type
            )

            # 3. Upload Chunks
            for i, chunk_text in enumerate(chunks):
                session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (c:Chunk {index: $idx, text: $text, embedding: $embedding})
                    CREATE (d)-[:HAS_CHUNK]->(c)
                """, doc_name=doc_name, idx=i, text=chunk_text, embedding=embeddings[i])

            # 4. Special Ontology Mapping
            if doc_type == "directory":
                for i, chunk_text in enumerate(chunks):
                    try:
                        emp = json.loads(chunk_text)
                        # Normalization Strength: Line Cook -> Hotline Cook
                        title = "Hotline Cook" if emp['title'] == "Line Cook" else emp['title']
                        
                        # Extract status from resign_date
                        status = "resigned" if emp.get('resign_date') != "working" else "active"
                        
                        loc = emp.get('location', {})
                        store_id = loc.get('store_id') or loc.get('city', 'Unknown')
                        state_code = loc.get('state', 'Unknown')
                        dept = emp['department']

                        session.run("""
                            MERGE (e:Employee {employee_id: $id})
                            SET e.first_name = $first_name,
                                e.last_name = $last_name,
                                e.name = $full_name,
                                e.title = $title,
                                e.department = $department,
                                e.location_city = $city,
                                e.location_state = $state,
                                e.salary_type = $salary_type,
                                e.annual_salary = $annual_salary,
                                e.hourly_rate = $hourly_rate,
                                e.status = $status,
                                e.reports_to_id = $reports_to_id
                            
                            MERGE (loc:Location {id: $store_id})
                            SET loc.city = $city, loc.state = $state
                            
                            MERGE (r:Region {name: $state})
                            
                            MERGE (role:Role {id: $role_id})
                            SET role.title = $title, role.department = $department
                            
                            MERGE (e)-[:locatedAt]->(loc)
                            MERGE (loc)-[:inRegion]->(r)
                            MERGE (e)-[:HAS_ROLE]->(role)
                            
                            WITH e
                            WHERE $reports_to_id IS NOT NULL
                            MERGE (m:Employee {employee_id: $reports_to_id})
                            MERGE (e)-[:reportsTo]->(m)
                        """,
                        id=emp['employee_id'],
                        first_name=emp['first_name'],
                        last_name=emp['last_name'],
                        full_name=f"{emp['first_name']} {emp['last_name']}",
                        title=title,
                        department=dept,
                        city=loc.get('city', 'Unknown'),
                        state=state_code,
                        salary_type=emp.get('salary_type'),
                        annual_salary=emp.get('annual_salary'),
                        hourly_rate=emp.get('hourly_rate'),
                        status=status,
                        reports_to_id=emp.get('reports_to_id'),
                        store_id=store_id,
                        role_id=f"ROLE_{title.replace(' ', '_')}_{dept.replace(' ', '_')}"
                        )
                    except: continue
            
            elif doc_type == "mandate":
                for chunk_text in chunks:
                    try:
                        data = json.loads(chunk_text)
                        detail = data.get('Specific Detail', '')
                        choice = data.get('Recommended Choice for PPB', '')
                        
                        # Logic to parse the "State Specifics (Mandates)" row
                        if "State Specifics" in detail:
                            # Example: "Georgia ($7.25), California ($16.00), and New York ($15.00)"
                            matches = re.findall(r'([a-zA-Z ]+) \(\$([0-9.]+)\)', choice)
                            for state_full_name, wage in matches:
                                state_full_name = state_full_name.strip().replace("and ", "")
                                # Normalize to abbreviation (e.g., Georgia -> GA) to match Directory
                                state_code = STATE_MAP.get(state_full_name, state_full_name)
                                mandate_id = f"MANDATE_{state_code}"
                                
                                session.run("""
                                    MERGE (r:Region {name: $state})
                                    SET r.full_name = $full_name
                                    
                                    MERGE (m:Mandate {id: $m_id})
                                    SET m.minimum_wage = toFloat($min_wage),
                                        m.state = $state
                                        
                                    MERGE (r)-[:SUBJECT_TO]->(m)
                                """, state=state_code, min_wage=wage, full_name=state_full_name, m_id=mandate_id)
                                print(f"[+] Ontology Link: Region {state_code} -> Mandate {mandate_id} ($${wage})")
                    except: continue

        return {"neo4j_status": "success", "next_node": END}
    except Exception as e:
        print(f"[!] Neo4j Builder Error: {e}")
        return {"neo4j_status": "error", "errors": [str(e)], "next_node": END}

# --- WORKFLOW CONSTRUCTION ---
def build_dais_pipeline():
    workflow = StateGraph(HRProjectState)
    workflow.add_node("router", router)
    workflow.add_node("policy_specialist", policy_specialist)
    workflow.add_node("process_specialist", process_specialist)
    workflow.add_node("contact_specialist", contact_specialist)
    workflow.add_node("embedder", embedder)
    workflow.add_node("graph_builder", graph_builder)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", lambda x: x["next_node"], {
        "policy_specialist": "policy_specialist",
        "process_specialist": "process_specialist",
        "contact_specialist": "contact_specialist",
        "embedder": "embedder",
        END: END
    })
    workflow.add_edge("policy_specialist", "embedder")
    workflow.add_edge("process_specialist", "embedder")
    workflow.add_edge("contact_specialist", "embedder")
    workflow.add_edge("embedder", "graph_builder")
    workflow.add_edge("graph_builder", END)
    return workflow.compile()

def process_corpus():
    app = build_dais_pipeline()
    print(f"[*] Starting Agentic Ingestion Pipeline...")

    if not os.path.exists(RAW_DATA_PATH):
        print(f"[!] Error: {RAW_DATA_PATH} not found.")
        return

    for filename in os.listdir(RAW_DATA_PATH):
        file_path = os.path.join(RAW_DATA_PATH, filename)
        if os.path.isfile(file_path):
            print(f"\n>>> PROCESSING: {filename}")
            initial_state = {
                "doc_name": filename,
                "doc_path": file_path,
                "entities": [],
                "relationships": [],
                "embeddings": [],
                "errors": [],
                "metadata": {},
                "raw_text": "",
                "chunks": [],
                "doc_type": "unstructured",
                "region": "Unknown",
                "embeddings_generated": False,
                "neo4j_node_count": 0,
                "neo4j_status": "pending",
                "next_node": ""
            }
            try:
                app.invoke(initial_state)
                print(f"[+] SUCCESS: {filename} stored in graph.")
            except Exception as e:
                print(f"[!] FAILED: {filename} - Error: {e}")

    neo4j_driver.close()

if __name__ == "__main__":
    process_corpus()
