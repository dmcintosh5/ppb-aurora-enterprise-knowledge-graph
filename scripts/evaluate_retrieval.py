import json
import os
import sys
import math
import numpy as np

# Ensure project root is in path to allow direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.agent import PPBAgent
from langchain_core.messages import HumanMessage

def precision_at_k(retrieved, relevance_grades, k):
    """
    Calculates the precision at a specific cutoff K. 
    It measures the fraction of retrieved items in the top K that are relevant.
    """
    if k <= 0: return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if relevance_grades.get(item, 0) > 0)
    return hits / k

def recall_at_k(retrieved, relevant_ids, k):
    """
    Calculates the recall at a specific cutoff K.
    It measures the fraction of all truly relevant items that were successfully retrieved in the top K.
    """
    if not relevant_ids: return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & set(relevant_ids))
    return hits / len(relevant_ids)

def reciprocal_rank(retrieved, relevant_ids):
    """
    Calculates the Reciprocal Rank of the first relevant document found in the results.
    Returns 1/rank, where rank is the position of the first correct match.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0

def dcg_at_k(retrieved, relevance_grades, k):
    """
    Calculates the Discounted Cumulative Gain (DCG) for the top K results.
    DCG accounts for the graded relevance of documents and penalizes them if they appear low in the list.
    """
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        grade = relevance_grades.get(item, 0)
        dcg += grade / math.log2(rank + 1)
    return dcg

def ndcg_at_k(retrieved, relevance_grades, k):
    """
    Calculates the Normalized Discounted Cumulative Gain (nDCG) at K.
    Normalizes the DCG by dividing it by the Ideal DCG (the best possible ranking).
    """
    actual_dcg = dcg_at_k(retrieved, relevance_grades, k)
    # Ideal ranking sorts all possible relevant items by their grade
    ideal_ranking = sorted(relevance_grades.keys(), key=lambda x: relevance_grades[x], reverse=True)
    ideal_dcg = dcg_at_k(ideal_ranking, relevance_grades, k)
    if ideal_dcg == 0: return 0.0
    return actual_dcg / ideal_dcg

def evaluate_retrieval():
    """
    Main execution loop for the retrieval evaluation suite. 
    It loads the gold set, executes queries through AURORA, and generates a stratified 
    report containing Hit Rate, MRR, nDCG, and Recall.
    """
    gold_set_path = 'tests/fixtures/retrieval_gold_set.json'
    if not os.path.exists(gold_set_path):
        print(f"Error: {gold_set_path} not found.")
        return

    with open(gold_set_path, 'r') as f:
        gold_set = json.load(f)

    print(f"[*] Initializing PPB Agent for Evaluation...")
    agent = PPBAgent()
    
    results = []
    
    print(f"[*] Evaluating {len(gold_set)} queries...")
    for entry in gold_set:
        query = entry['query_text']
        category = entry['category']
        relevance_grades = entry['relevance_grades']
        relevant_ids = [k for k, v in relevance_grades.items() if v > 0]
        
        print(f"  > [{category}] {query}")
        
        # 1. Simulate Agent Workflow up to Retrieval
        state = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "intent": {},
            "extracted_entities": [],
            "resolved_entities": [],
            "context": [],
            "graph_data": {}
        }
        
        # Intent & Entity Resolution
        state = agent.intent_parser_node(state)
        intent_type = state['intent']['type']
        
        retrieved_ids = []
        
        if intent_type == "organizational":
            state = agent.entity_resolver_node(state)
            state = agent.graph_executor_node(state)
            graph_data = state['graph_data']
            
            # Extract IDs from graph data
            if isinstance(graph_data, list):
                for item in graph_data:
                    if isinstance(item, dict) and 'id' in item:
                        retrieved_ids.append(item['id'])
                    elif isinstance(item, str): # e.g. reporting chain
                        retrieved_ids.append(item)
            elif isinstance(graph_data, dict) and 'id' in graph_data:
                retrieved_ids.append(graph_data['id'])
        else:
            state = agent.neo4j_retriever_node(state)
            context = state['context']
            # Map chunks to f"{source}:chunk_{index}"
            for item in context:
                retrieved_ids.append(f"{item['source']}:chunk_{item['index']}")

        # 2. Calculate Metrics
        metrics = {
            "query_id": entry['query_id'],
            "category": category,
            "hit_rate_at_3": 1.0 if set(retrieved_ids[:3]) & set(relevant_ids) else 0.0,
            "hit_rate_at_5": 1.0 if set(retrieved_ids[:5]) & set(relevant_ids) else 0.0,
            "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg_at_5": ndcg_at_k(retrieved_ids, relevance_grades, 5),
            "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5)
        }
        results.append(metrics)

    # 3. Stratified Reporting
    print("\n" + "="*60)
    print("  STRATIFIED RETRIEVAL EVALUATION REPORT")
    print("="*60)
    
    categories = sorted(list(set(r['category'] for r in results)))
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        print(f"\n[{cat.upper()}] ({len(cat_results)} queries)")
        print("-" * 30)
        
        for metric in ["hit_rate_at_3", "hit_rate_at_5", "mrr", "ndcg_at_5", "recall_at_5"]:
            scores = [r[metric] for r in cat_results]
            mean_score = np.mean(scores)
            bar = "█" * int(mean_score * 20)
            print(f"  {metric:<15}: {mean_score:.4f}  {bar}")

    # Overall Metrics
    print("\n" + "="*60)
    print(f"[OVERALL] ({len(results)} queries)")
    for metric in ["hit_rate_at_3", "hit_rate_at_5", "mrr", "ndcg_at_5", "recall_at_5"]:
        mean_score = np.mean([r[metric] for r in results])
        print(f"  {metric:<15}: {mean_score:.4f}")
    print("="*60)

    agent.close()
    if not os.path.exists(gold_set_path):
        print(f"Error: {gold_set_path} not found.")
        return

    with open(gold_set_path, 'r') as f:
        gold_set = json.load(f)

    print(f"[*] Initializing PPB Agent for Evaluation...")
    agent = PPBAgent()
    
    results = []
    
    print(f"[*] Evaluating {len(gold_set)} queries...")
    for entry in gold_set:
        query = entry['query_text']
        category = entry['category']
        relevance_grades = entry['relevance_grades']
        relevant_ids = [k for k, v in relevance_grades.items() if v > 0]
        
        print(f"  > [{category}] {query}")
        
        # 1. Simulate Agent Workflow up to Retrieval
        state = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "intent": {},
            "extracted_entities": [],
            "resolved_entities": [],
            "context": [],
            "graph_data": {}
        }
        
        # Intent & Entity Resolution
        state = agent.intent_parser_node(state)
        intent_type = state['intent']['type']
        
        retrieved_ids = []
        
        if intent_type == "organizational":
            state = agent.entity_resolver_node(state)
            state = agent.graph_executor_node(state)
            graph_data = state['graph_data']
            
            # Extract IDs from graph data
            if isinstance(graph_data, list):
                for item in graph_data:
                    if isinstance(item, dict) and 'id' in item:
                        retrieved_ids.append(item['id'])
                    elif isinstance(item, str): # e.g. reporting chain
                        retrieved_ids.append(item)
            elif isinstance(graph_data, dict) and 'id' in graph_data:
                retrieved_ids.append(graph_data['id'])
        else:
            state = agent.neo4j_retriever_node(state)
            context = state['context']
            # Map chunks to f"{source}:chunk_{index}"
            for item in context:
                retrieved_ids.append(f"{item['source']}:chunk_{item['index']}")

        # 2. Calculate Metrics
        metrics = {
            "query_id": entry['query_id'],
            "category": category,
            "hit_rate_at_3": 1.0 if set(retrieved_ids[:3]) & set(relevant_ids) else 0.0,
            "hit_rate_at_5": 1.0 if set(retrieved_ids[:5]) & set(relevant_ids) else 0.0,
            "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg_at_5": ndcg_at_k(retrieved_ids, relevance_grades, 5),
            "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5)
        }
        results.append(metrics)

    # 3. Stratified Reporting
    print("\n" + "="*60)
    print("  STRATIFIED RETRIEVAL EVALUATION REPORT")
    print("="*60)
    
    categories = sorted(list(set(r['category'] for r in results)))
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        print(f"\n[{cat.upper()}] ({len(cat_results)} queries)")
        print("-" * 30)
        
        for metric in ["hit_rate_at_3", "hit_rate_at_5", "mrr", "ndcg_at_5", "recall_at_5"]:
            scores = [r[metric] for r in cat_results]
            mean_score = np.mean(scores)
            bar = "█" * int(mean_score * 20)
            print(f"  {metric:<15}: {mean_score:.4f}  {bar}")

    # Overall Metrics
    print("\n" + "="*60)
    print(f"[OVERALL] ({len(results)} queries)")
    for metric in ["hit_rate_at_3", "hit_rate_at_5", "mrr", "ndcg_at_5", "recall_at_5"]:
        mean_score = np.mean([r[metric] for r in results])
        print(f"  {metric:<15}: {mean_score:.4f}")
    print("="*60)

    agent.close()

if __name__ == "__main__":
    evaluate_retrieval()
