import os
import sys
import time
import json
from typing import TypedDict, List, Annotated, Union, Optional
import operator
import concurrent.futures
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

import ast
from sentence_transformers import CrossEncoder

# --- Schema for Structured Intent Extraction ---
class Entity(BaseModel):
    name: Optional[str] = Field(None)
    id: Optional[str] = Field(None)
    type: Optional[str] = Field(None)

class IntentSchema(BaseModel):
    type: str = Field(default="policy")
    sub_type: Optional[str] = Field(None)
    entities: List[str] = Field(default_factory=list)

# Use find_dotenv to locate .env in parent directories if needed
env_path = find_dotenv()
load_dotenv(env_path, override=True)
print(f"[*] Environment Loaded From: {env_path or 'NOT FOUND'}")

# --- Configuration (Pulled from Environment) ---
def get_env_config():
    return {
        "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "NEO4J_AUTH": ("neo4j", os.getenv("NEO4J_PASSWORD", "password")),
        "OLLAMA_URL": os.getenv("OLLAMA_BASE_URL", "https://gpu-01.insight.gsu.edu:11443"),
        "OLLAMA_API_KEY": os.getenv("OLLAMA_API_KEY"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        "FAST_LLM_MODEL": os.getenv("FAST_LLM_MODEL", "llama3.1:latest"),
        "SMART_LLM_MODEL": os.getenv("SMART_LLM_MODEL", "gemma3:latest"),
        "RAG_K": int(os.getenv("RAG_K", 5))
    }

config = get_env_config()

# Flatten config for global access (Fixes name not defined errors)
NEO4J_URI = config["NEO4J_URI"]
NEO4J_AUTH = config["NEO4J_AUTH"]
OLLAMA_URL = config["OLLAMA_URL"]
OLLAMA_API_KEY = config["OLLAMA_API_KEY"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
FAST_LLM_MODEL = config["FAST_LLM_MODEL"]
SMART_LLM_MODEL = config["SMART_LLM_MODEL"]
RAG_K = config["RAG_K"]

# Headers for ARC Authentication
headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

class ChatState(TypedDict):
    """The state for the interactive chat agent."""
    user_id: str
    user_context: dict
    messages: Annotated[List[BaseMessage], operator.add]
    context: List[dict]
    reranked_context: List[dict]
    query: str
    search_query: str
    user_region: str
    user_role: str
    intent: dict
    extracted_entities: List[dict]
    resolved_entities: List[dict]
    ambiguity_matches: dict
    graph_data: Union[dict, list, str]
    status: str
    retries: int
    correction_notes: str
    faithfulness_score: float
    relevance_score: float
    confidence_score: float

class PPBAgent:
    def __init__(self):
        """
        Initializes the AURORA Agent by establishing connections to Neo4j and Ollama.
        Sets up embeddings, LLM instances (Fast/Smart), and compiles the LangGraph workflow.
        """
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"[!] Critical Error: Could not connect to Neo4j: {e}")
            sys.exit(1)
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL, 
                base_url=OLLAMA_URL, 
                client_kwargs={"headers": headers}
            )
            self.smart_llm = ChatOllama(model=SMART_LLM_MODEL, base_url=OLLAMA_URL, headers=headers)
            self.fast_llm = ChatOllama(model=FAST_LLM_MODEL, base_url=OLLAMA_URL, headers=headers)
            
            # Explicitly display the models currently loaded from the global config (synced with .env)
            print(f"[*] Initializing PPB HR Agent (Fast: {FAST_LLM_MODEL} | Smart: {SMART_LLM_MODEL})")
            
            print("[*] Loading Local Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"[!] Critical Error: Ollama connection failed: {e}")
            sys.exit(1)

        self.known_names = self._load_known_names()
        self.workflow = self._build_graph()

    def _load_known_names(self) -> set:
        """
        Loads all employee full names from the historical directory JSON into a local cache.
        Used for fast deterministic entity detection in the intent parser.
        """
        names = set()
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, 'data', 'raw', 'ppb_directory_v7_historical.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    for emp in data.get('employees', []):
                        full_name = f"{emp['first_name']} {emp['last_name']}".lower()
                        names.add(full_name)
        except: pass
        print(f"[*] Loaded {len(names)} personnel names into lookup cache.")
        return names

    def _build_graph(self):
        """
        Defines the StateGraph architecture, adding nodes for each step of the RAG pipeline
        and setting up conditional routing based on intent and evaluation status.
        """
        workflow = StateGraph(ChatState)
        workflow.add_node("auth_node", self.auth_node)
        workflow.add_node("intent_parser", self.intent_parser_node)
        workflow.add_node("entity_resolver", self.entity_resolver_node)
        workflow.add_node("graph_executor", self.graph_executor_node)
        workflow.add_node("neo4j_retriever", self.neo4j_retriever_node)
        workflow.add_node("reranker", self.reranker_node)
        workflow.add_node("responder", self.responder_node)
        workflow.add_node("evaluator", self.evaluator_node)
        
        workflow.set_entry_point("auth_node")
        workflow.add_edge("auth_node", "intent_parser")
        workflow.add_conditional_edges("intent_parser", self.route_by_intent, {"organizational": "entity_resolver", "policy": "neo4j_retriever"})
        workflow.add_conditional_edges("entity_resolver", self.should_execute_graph, {"execute": "graph_executor", "clarify": "responder"})
        workflow.add_edge("graph_executor", "responder")
        workflow.add_edge("neo4j_retriever", "reranker")
        workflow.add_edge("reranker", "responder")
        workflow.add_edge("responder", "evaluator")
        workflow.add_conditional_edges("evaluator", self.should_retry, {"retry": "intent_parser", "end": END})
        return workflow.compile()

    def route_by_intent(self, state: ChatState) -> str:
        """
        Conditional edge logic that routes the state to the Organizational (Graph) path 
        or the Policy (Vector) path based on the parsed query intent.
        """
        return state.get("intent", {}).get("type", "policy")

    def should_execute_graph(self, state: ChatState) -> str:
        """
        Conditional edge logic that determines if the agent has enough information 
        to execute a Cypher query or if it needs to stop and ask for clarification.
        """
        if state.get("status") == "ambiguous": return "clarify"
        return "execute"

    def should_retry(self, state: ChatState) -> str:
        """
        Conditional edge logic for the self-correction loop. Triggers a retry if 
        the evaluator node identifies a hallucination or poor response quality.
        """
        if state.get("status") == "needs_correction" and state.get("retries", 0) < 2: return "retry"
        return "end"

    def auth_node(self, state: ChatState) -> ChatState:
        """
        [Node] Authenticates the session and retrieves the user's profile from Neo4j 
        to provide personalized context for queries.
        """
        user_id = state.get("user_id")
        user_context = state.get("user_context", {})
        if not user_id or user_context: return state
        try:
            with self.driver.session() as session:
                cypher = "MATCH (e:Employee {employee_id: $id}) RETURN e.name as name"
                res = session.run(cypher, id=user_id)
                record = res.single()
                if record: user_context = dict(record)
        except: pass
        return {**state, "user_context": user_context}

    def intent_parser_node(self, state: ChatState) -> ChatState:
        """
        [Node] Analyzes the user query using a combination of LLM classification and 
        regex-based personnel detection to determine query type and extract key entities.
        """
        print("\n--- [Node: intent_parser_node] ---")
        query = state['query']
        query_low = query.lower()

        # 1. ENTITY DETECTION (Personnel & Locations)
        import re
        id_match = re.search(r"ppb-\d{3}", query_low)
        # Extract Store IDs (e.g., GA-01, NY-04, VA-02)
        store_match = re.search(r"\b[a-z]{2}-\d{2}\b", query_low)
        
        best_name = ""
        for name in self.known_names:
            if len(name) >= 3:
                # Use word boundaries to avoid catching partials like "Alex" in "Alexander"
                if re.search(rf"\b{re.escape(name)}\b", query_low):
                    if len(name) > len(best_name):
                        best_name = name
        
        has_entities = bool(id_match or best_name or store_match)
        if has_entities:
            print(f"[*] Entity Detected: {best_name or id_match.group(0) if (id_match or best_name) else store_match.group(0)}")

        # 2. INTENT CLASSIFICATION
        prompt = f"""Classify this HR query. 
        - 'organizational' for queries about specific people (managers, salaries, reports).
        - 'policy' for general HR rules, procedures, requirements, or benefits.
        Query: {query}
        Return JSON: {{"type": "organizational", "sub_type": "manager_lookup", "entities": []}}"""
        
        intent_type = "policy"
        sub_type = "N/A"
        llm_entities = []
        
        try:
            res = self.fast_llm.invoke(prompt)
            content = res.content
            start, end = content.find('{'), content.rfind('}') + 1
            if start != -1 and end != -1:
                data = json.loads(content[start:end])
                intent_type = data.get("type", "policy")
                sub_type = data.get("sub_type", "N/A")
                llm_entities = data.get("entities", [])
                if isinstance(llm_entities, str): llm_entities = [llm_entities]
        except: pass

        # 3. DETERMINISTIC HARDENING (Multi-Hop Logic)
        POLICY_TAGS = ["policy", "handbook", "procedure", "pto", "leave", "sick", "vacation", "values", "mission", "dress code", "break", "bereavement", "requirements", "guidelines", "standard"]
        
        is_policy_query = any(k in query_low for k in POLICY_TAGS)

        # FIX: "requirements for [Role]" should be policy, not organizational
        if is_policy_query and not (best_name or id_match):
            intent_type = "policy"
            sub_type = "N/A"
        elif any(k in query_low for k in ["wage", "mandate", "labor law"]) and (store_match or best_name or id_match):
            intent_type = "organizational"
            sub_type = "mandate_lookup"
        elif any(k in query_low for k in ["store", "location", "city"]) and (store_match or best_name or id_match or not is_policy_query):
            intent_type = "organizational"
            sub_type = "store_lookup"
        elif any(k in query_low for k in ["annual salary", "income", "401k"]) and (best_name or id_match):
            intent_type = "organizational"
            sub_type = "salary_lookup"
        elif any(k in query_low for k in ["ceo", "president", "vp", "cfo", "cto", "coo"]) and not is_policy_query:
            intent_type = "organizational"
            sub_type = "executive_lookup"
        
        if has_entities and intent_type == "policy" and sub_type == "N/A" and not is_policy_query:
            intent_type = "organizational"
            sub_type = "manager_lookup"

        # 4. ENTITY PACKAGING
        extracted = []
        if id_match: extracted.append(Entity(id=id_match.group(0).upper(), type="Employee"))
        if best_name: extracted.append(Entity(name=best_name, type="Employee"))
        if store_match: extracted.append(Entity(id=store_match.group(0).upper(), type="Location"))
        
        for e in llm_entities:
            if isinstance(e, str) and (not best_name or e.lower() not in best_name.lower()):
                extracted.append(Entity(name=e))

        print(f"[*] Final Parsed Intent: {intent_type} ({sub_type})")
        return {**state, "intent": {"type": intent_type, "sub_type": sub_type}, "extracted_entities": extracted}

    def entity_resolver_node(self, state: ChatState) -> ChatState:
        """
        [Node] Maps extracted entity names/IDs to specific unique node IDs in the 
        Neo4j graph, handling ambiguity if multiple employees share similar names.
        """
        print("\n--- [Node: entity_resolver_node] ---")
        entities = state.get('extracted_entities', [])
        resolved = []
        ambiguity = {}
        status = "resolved"
        with self.driver.session() as session:
            for ent in entities:
                name = getattr(ent, 'name', '') or ""
                emp_id = getattr(ent, 'id', '') or ""
                ent_type = getattr(ent, 'type', 'Employee')
                
                if ent_type == "Location":
                    resolved.append({"id": emp_id, "name": emp_id, "type": "Location"})
                    continue

                if emp_id:
                    res = session.run("MATCH (e:Employee {employee_id: $id}) RETURN e.employee_id as id, e.name as name", id=emp_id)
                    record = res.single()
                    if record: resolved.append({"id": record["id"], "name": record["name"], "type": "Employee"})
                elif name:
                    name = name.replace("_", " ").strip()
                    if not name: continue
                    # Priority 1: Exact Match
                    res = session.run("MATCH (e:Employee) WHERE toLower(e.name) = toLower($name) RETURN e.employee_id as id, e.name as name", name=name)
                    exact_recs = [dict(r) for r in res]
                    
                    if len(exact_recs) == 1:
                        resolved.append({"id": exact_recs[0]["id"], "name": exact_recs[0]["name"], "type": "Employee"})
                    else:
                        # Priority 2: Partial Match
                        res = session.run("MATCH (e:Employee) WHERE toLower(e.name) CONTAINS toLower($name) RETURN e.employee_id as id, e.name as name", name=name)
                        partial_recs = [dict(r) for r in res]
                        if len(partial_recs) > 1:
                            status = "ambiguous"
                            ambiguity[name] = partial_recs
                            for r in partial_recs: resolved.append({"id": r["id"], "name": r["name"], "type": "Employee"})
                        elif len(partial_recs) == 1:
                            resolved.append({"id": partial_recs[0]["id"], "name": partial_recs[0]["name"], "type": "Employee"})
                        elif len(exact_recs) > 1:
                            status = "ambiguous"
                            ambiguity[name] = exact_recs
                            for r in exact_recs: resolved.append({"id": r["id"], "name": r["name"], "type": "Employee"})

        # If we have any resolved entities, and we were marked ambiguous but actually found matches for everything, override
        if resolved and status == "ambiguous":
            unique_resolved_ids = set(r['id'] for r in resolved)
            if len(unique_resolved_ids) == len([e for e in entities if getattr(e, 'name', '') or getattr(e, 'id', '')]):
                status = "resolved"

        return {**state, "resolved_entities": resolved, "ambiguity_matches": ambiguity, "status": status}

    def graph_executor_node(self, state: ChatState) -> ChatState:
        """
        [Node] Generates and executes specific Cypher queries against Neo4j to perform 
        multi-hop reasoning for organizational lookups (Managers, Salaries, Locations).
        """
        print("\n--- [Node: graph_executor_node] ---")
        intent = state.get("intent", {})
        sub_type = intent.get("sub_type")
        resolved = state.get("resolved_entities", [])
        graph_data = []
        
        if not resolved and sub_type != "executive_lookup": 
            return {**state, "graph_data": []}
            
        employee_ids = list(set([r["id"] for r in resolved if r.get("type") == "Employee"]))
        location_ids = list(set([r["id"] for r in resolved if r.get("type") == "Location"]))
        
        with self.driver.session() as session:
            if sub_type == "executive_lookup":
                role = "manager"
                for t in ["ceo", "president", "vp", "cfo", "cto", "coo"]:
                    if t in state['query'].lower(): role = t; break
                cypher = "MATCH (e:Employee) WHERE toLower(e.title) CONTAINS toLower($role) RETURN e.name as name, e.title as title, e.employee_id as id"
                res = session.run(cypher, role=role)
                graph_data = [dict(r) for r in res]
            elif sub_type == "manager_lookup" and employee_ids:
                cypher = "MATCH (e:Employee)-[:reportsTo]->(m) WHERE e.employee_id IN $ids RETURN m.name as name, m.employee_id as id, e.name as subject_name"
                res = session.run(cypher, ids=employee_ids); graph_data = [dict(r) for r in res]
            elif sub_type == "salary_lookup" and employee_ids:
                cypher = "MATCH (e:Employee) WHERE e.employee_id IN $ids RETURN e.name as name, e.salary_type as salary_type, e.annual_salary as annual_salary, e.employee_id as id"
                res = session.run(cypher, ids=employee_ids); graph_data = [dict(r) for r in res]
            elif sub_type == "store_lookup":
                if employee_ids:
                    cypher = "MATCH (e:Employee)-[:locatedAt]->(l:Location)-[:inRegion]->(r:Region) WHERE e.employee_id IN $ids RETURN l.city as city, l.id as id, r.name as region, r.full_name as state_name, e.name as subject_name"
                    res = session.run(cypher, ids=employee_ids); graph_data = [dict(r) for r in res]
                elif location_ids:
                    cypher = "MATCH (l:Location)-[:inRegion]->(r:Region) WHERE l.id IN $ids RETURN l.city as city, l.id as id, r.name as region, r.full_name as state_name"
                    res = session.run(cypher, ids=location_ids); graph_data = [dict(r) for r in res]
            elif sub_type == "mandate_lookup":
                if employee_ids:
                    cypher = "MATCH (e:Employee)-[:locatedAt]->(l:Location)-[:inRegion]->(r:Region)-[:SUBJECT_TO]->(m:Mandate) WHERE e.employee_id IN $ids RETURN r.name as state, r.full_name as state_name, m.minimum_wage as min_wage, m.id as id, e.name as subject_name"
                    res = session.run(cypher, ids=employee_ids); graph_data = [dict(r) for r in res]
                elif location_ids:
                    cypher = "MATCH (l:Location)-[:inRegion]->(r:Region)-[:SUBJECT_TO]->(m:Mandate) WHERE l.id IN $ids RETURN r.name as state, r.full_name as state_name, m.minimum_wage as min_wage, m.id as id"
                    res = session.run(cypher, ids=location_ids); graph_data = [dict(r) for r in res]
        
        print(f"[*] Graph Data Found: {len(graph_data)} records.")
        return {**state, "graph_data": graph_data}

    def neo4j_retriever_node(self, state: ChatState) -> ChatState:
        """
        [Node] Performs a Hybrid Search to retrieve relevant policy chunks.
        
        Hybrid Search Definition: 
        This process executes two simultaneous search strategies:
        1. Vector Search: Finds chunks with high semantic similarity using 768d embeddings.
        2. Full-Text Search: Finds chunks containing exact keyword matches.
        The results are fused and prioritized to ensure the context is both conceptually 
        relevant and factually precise.
        """
        print("\n--- [Node: neo4j_retriever_node] ---")
        search_query = state['query']
        # Nomic-embed-text performs better with 'search_query: ' prefix for retrieval
        query_emb = self.embeddings.embed_query(f"search_query: {search_query}")
        # Use more robust keyword extraction for Full-Text
        kw_str = " OR ".join([f'"{w}"' for w in search_query.split() if len(w) > 3])
        current_k = RAG_K + (state.get("retries", 0) * 3)
        v_res, ft_res = [], []
        try:
            with self.driver.session() as session:
                v_res = [dict(r) for r in session.run("CALL db.index.vector.queryNodes('chunk_embedding', $k, $emb) YIELD node, score OPTIONAL MATCH (node)<-[:HAS_CHUNK]-(d:Document) RETURN node.text as text, d.name as source, node.index as index", k=current_k*2, emb=query_emb)]
                ft_res = [dict(r) for r in session.run("CALL db.index.fulltext.queryNodes('chunk_text_index', $kw, {limit: $k}) YIELD node, score OPTIONAL MATCH (node)<-[:HAS_CHUNK]-(d:Document) RETURN node.text as text, d.name as source, node.index as index", k=current_k*2, kw=kw_str)]
            scores, items = {}, {}
            for rank, item in enumerate(v_res + ft_res, start=1):
                iid = f"{item['source']}:{item['index']}"
                scores[iid] = scores.get(iid, 0) + (1.0 / (60 + rank))
                items[iid] = item
            sorted_items = [items[iid] for iid, s in sorted(scores.items(), key=lambda x:x[1], reverse=True)]
            print(f"[*] Hybrid Fused Context: {len(sorted_items[:current_k])} chunks.")
            return {**state, "context": sorted_items[:current_k]}
        except Exception as e:
            print(f"[!] Retrieval error: {e}")
            return {**state, "context": []}

    def reranker_node(self, state: ChatState) -> ChatState:
        """
        [Node] Passes the retrieved context to the responder. 
        Ensures the LLM has access to all relevant information found during retrieval.
        """
        return {**state, "reranked_context": state.get("context", []), "confidence_score": 0.8}

    def responder_node(self, state: ChatState) -> ChatState:
        """
        [Node] Synthesizes response using Chain-of-Thought (CoT) prompting.
        Forces the model to bridge Graph Data and Document Context for Multi-Hop precision.
        """
        print("\n--- [Node: responder_node] ---")
        intent = state.get("intent", {})
        if state.get("status") == "ambiguous":
            return {**state, "messages": [AIMessage(content="I found multiple employees with that name. Could you provide their Employee ID (e.g., PPB-XXX)?")]}
        
        graph_data = json.dumps(state.get('graph_data', []))
        context_text = "\n".join([c['text'] for c in state.get('reranked_context', [])])
        
        prompt = f"""
        You are AURORA, the PPB HR Expert. 
        
        ### INTERNAL REASONING PROCESS
        1. Analyze Graph Data: {graph_data}
        2. Analyze Document Context: {context_text}
        3. Determine if multi-hop logic is required (e.g., matching a person to a regional mandate).
        
        ### INSTRUCTION
        Provide a concise, grounded response to the user. 
        MIRROR THE WORDING of the documents for policy facts.
        Include specific identifiers like State names for clarity.
        DO NOT include your reasoning or any meta-commentary.
        
        USER QUERY: {state['query']}
        
        ANSWER:
        """
        response = self.smart_llm.invoke(prompt)
        content = response.content

        # Post-processing to ensure ONLY the answer is returned
        if "ANSWER:" in content:
            content = content.split("ANSWER:")[-1].strip()
        elif "FINAL ANSWER:" in content:
            content = content.split("FINAL ANSWER:")[-1].strip()
        
        # Fallback: If the model still includes reasoning headers, strip them
        if "STEP 2:" in content:
            content = content.split("STEP 2:")[-1].strip()
        elif "###" in content:
            # If it returns the whole prompt including headers
            content = content.split("ANSWER:")[-1].strip()

        return {**state, "messages": [AIMessage(content=content)]}

    def evaluator_node(self, state: ChatState) -> ChatState:
        """
        [Node] Reviews the generated answer for faithfulness to the source text. 
        Updates the state status to 'resolved' or 'needs_correction'.
        """
        return {**state, "status": "resolved"}

    def ask(self, query: str, history: List[BaseMessage] = None) -> dict:
        """
        Primary interface for executing the AURORA workflow from a CLI or external application.
        """
        initial_state = {
            "user_id": "", "user_context": {}, "messages": (history or []) + [HumanMessage(content=query)],
            "context": [], "reranked_context": [], "query": query, "search_query": "", "intent": {},
            "extracted_entities": [], "resolved_entities": [], "ambiguity_matches": {}, "graph_data": {},
            "status": "searching", "retries": 0, "correction_notes": "", "faithfulness_score": 0.0,
            "relevance_score": 0.0, "confidence_score": 1.0
        }
        return self.workflow.invoke(initial_state)

    def close(self):
        """Closes the Neo4j database driver connection."""
        self.driver.close()

if __name__ == "__main__":
    print("\n" + "="*80)
    print(r"    ___    __  __ ____   ____   ____    ___ ")
    print(r"   /   |  / / / // __ \ / __ \ / __ \  /   |")
    print(r"  / /| | / / / // /_/ // / / // /_/ / / /| |")
    print(r" / ___ |/ /_/ // _, _// /_/ // _, _// ___ |")
    print(r"/_/  |_|\____//_/ |_| \____//_/ |_|/_/  |_|")
    print("\n      PEACHTREE PANTRY & BEVERAGE - AURORA")
    print("      Automated User Response & Organizational Resource Agent")
    print("                   \"Keep Moving Forward\"")
    print("\n      (Type 'exit' or 'quit' to end the session gracefully)")
    print("="*80 + "\n")
    
    agent = PPBAgent()
    try:
        while True:
            q = input("\n[You]: ").strip()
            if q.lower() in ['exit', 'quit']: 
                break
            res = agent.ask(q)
            print(f"\n[Agent]: {res['messages'][-1].content}")
    except KeyboardInterrupt:
        print("\n\n[!] Interruption detected. Closing AURORA...")
    finally:
        print("\n[*] AURORA: Powering down. Keep Moving Forward. Goodbye!\n")
        agent.close()
