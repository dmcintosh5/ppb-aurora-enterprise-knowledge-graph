import json
import os
import math
import numpy as np
from scripts.agent import PPBAgent
from langchain_core.messages import HumanMessage

def recall_at_k(retrieved, relevant_ids, k):
    if not relevant_ids: return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)

def hit_rate_at_k(retrieved, relevant_ids, k):
    if not relevant_ids: return 0.0
    return 1.0 if set(retrieved[:k]) & set(relevant_ids) else 0.0

def sensitivity_analysis():
    gold_set_path = 'tests/fixtures/retrieval_gold_set.json'
    if not os.path.exists(gold_set_path):
        print(f"Error: {gold_set_path} not found.")
        return

    with open(gold_set_path, 'r') as f:
        gold_set = json.load(f)

    # Force a high K for extraction
    os.environ["RAG_K"] = "15"
    
    print(f"[*] Initializing PPB Agent for Sensitivity Analysis (K_max=15)...")
    agent = PPBAgent()
    
    # Store results: query_id -> retrieved_ids
    all_retrieved = {}
    query_targets = {}

    print(f"[*] Extracting retrieval results for {len(gold_set)} queries...")
    for entry in gold_set:
        query = entry['query_text']
        relevance_grades = entry['relevance_grades']
        relevant_ids = [k for k, v in relevance_grades.items() if v > 0]
        query_targets[entry['query_id']] = relevant_ids
        
        state = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "intent": {},
            "extracted_entities": [],
            "resolved_entities": [],
            "context": [],
            "graph_data": {},
            "retries": 0
        }
        
        state = agent.intent_parser_node(state)
        intent_type = state['intent']['type']
        
        retrieved_ids = []
        if intent_type == "organizational":
            state = agent.entity_resolver_node(state)
            state = agent.graph_executor_node(state)
            graph_data = state['graph_data']
            if isinstance(graph_data, list):
                for item in graph_data:
                    if isinstance(item, dict) and 'id' in item:
                        retrieved_ids.append(item['id'])
                    elif isinstance(item, str):
                        retrieved_ids.append(item)
            elif isinstance(graph_data, dict) and 'id' in graph_data:
                retrieved_ids.append(graph_data['id'])
        else:
            state = agent.neo4j_retriever_node(state)
            for item in state['context']:
                retrieved_ids.append(f"{item['source']}:chunk_{item['index']}")
        
        all_retrieved[entry['query_id']] = retrieved_ids

    # Sweep across K
    k_range = [1, 2, 3, 5, 7, 10, 15]
    print("\n" + "="*60)
    print(f"{'K':<5} | {'Mean Recall':<15} | {'Hit Rate':<15}")
    print("-" * 40)

    for k in k_range:
        recalls = []
        hit_rates = []
        for q_id, retrieved in all_retrieved.items():
            relevant = query_targets[q_id]
            recalls.append(recall_at_k(retrieved, relevant, k))
            hit_rates.append(hit_rate_at_k(retrieved, relevant, k))
        
        mean_recall = np.mean(recalls)
        mean_hr = np.mean(hit_rates)
        print(f"{k:<5} | {mean_recall:<15.4f} | {mean_hr:<15.4f}")
    
    print("="*60)
    print("[*] Sensitivity Analysis Complete.")
    agent.close()

if __name__ == "__main__":
    sensitivity_analysis()
