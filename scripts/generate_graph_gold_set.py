import json
import os
import random

def generate_graph_gold_set():
    input_file = 'data/processed/ppb_graph_adjacency.json'
    output_file = 'tests/fixtures/retrieval_gold_set.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        graph_data = json.load(f)

    nodes = {node['id']: node for node in graph_data['nodes']}
    edges = graph_data['edges']
    
    gold_set = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            gold_set = json.load(f)

    # Helper to find targets of a certain type for a source
    def find_targets(source_id, edge_type):
        return [e['target'] for e in edges if e['source'] == source_id and e['type'] == edge_type]

    # 1. Manager Lookups (Graph-Relational)
    employees = [n for n in nodes.values() if n['label'] == 'Employee']
    sampled_emps = random.sample(employees, min(10, len(employees)))
    
    for emp in sampled_emps:
        managers = find_targets(emp['id'], 'reportsTo')
        if managers:
            manager_id = managers[0]
            manager_name = nodes[manager_id]['properties']['name']
            gold_set.append({
                "query_id": f"graph_mgr_{emp['id']}",
                "category": "Graph-Relational",
                "query_text": f"Who is the manager for {emp['properties']['name']}?",
                "expected_answer": manager_name,
                "relevance_grades": {
                    manager_id: 2,
                    emp['id']: 1
                }
            })

    # 2. Store Lookups (Graph-Relational)
    for emp in sampled_emps:
        locations = find_targets(emp['id'], 'locatedAt')
        if locations:
            loc_id = locations[0]
            city = nodes[loc_id]['properties']['city']
            gold_set.append({
                "query_id": f"graph_store_{emp['id']}",
                "category": "Graph-Relational",
                "query_text": f"Which city does {emp['properties']['name']} work in?",
                "expected_answer": city,
                "relevance_grades": {
                    loc_id: 2,
                    emp['id']: 1
                }
            })

    # 3. Mandate Lookups (Multi-Hop)
    for emp in sampled_emps:
        locations = find_targets(emp['id'], 'locatedAt')
        if locations:
            loc_id = locations[0]
            regions = find_targets(loc_id, 'inRegion')
            if regions:
                region_id = regions[0]
                mandates = find_targets(region_id, 'SUBJECT_TO')
                if mandates:
                    mandate_id = mandates[0]
                    wage = nodes[mandate_id]['properties']['minimum_wage']
                    gold_set.append({
                        "query_id": f"multihop_wage_{emp['id']}",
                        "category": "Multi-Hop",
                        "query_text": f"What is the minimum wage for {emp['properties']['name']}'s location?",
                        "expected_answer": str(wage),
                        "relevance_grades": {
                            mandate_id: 2,
                            region_id: 1,
                            loc_id: 1,
                            emp['id']: 1
                        }
                    })

    with open(output_file, 'w') as f:
        json.dump(gold_set, f, indent=4)

    print(f"Graph gold set generation complete. Total queries: {len(gold_set)}")

if __name__ == "__main__":
    generate_graph_gold_set()
