import os
import sys
import json
# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.agent import PPBAgent

def debug_factoid():
    agent = PPBAgent()
    query = "What are the core values of Peachtree Pantry & Beverage Co.?"
    print(f"\n[*] DEBUG QUERY: {query}")
    
    res = agent.ask(query)
    
    print("\n--- RETRIEVAL RESULTS ---")
    for i, chunk in enumerate(res.get('context', [])):
        print(f"[{i}] Source: {chunk.get('source')} | Index: {chunk.get('index')}")
        print(f"    Text: {chunk.get('text')[:200]}...")
    
    # Check what the Gold Set expected
    gold_set_path = 'tests/fixtures/retrieval_gold_set.json'
    with open(gold_set_path, 'r') as f:
        gold_set = json.load(f)
    
    target = [g for r in gold_set for g in [r] if r['query_text'] == query][0]
    print("\n--- GOLD SET EXPECTATION ---")
    print(f"Relevant IDs: {target['relevance_grades']}")
    
    agent.close()

if __name__ == "__main__":
    debug_factoid()
