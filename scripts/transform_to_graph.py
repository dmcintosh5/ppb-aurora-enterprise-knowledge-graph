import json
import os

def transform_to_graph():
    input_file = 'data/raw/ppb_directory_v7_historical.json'
    output_file = 'data/processed/ppb_graph_adjacency.json'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    employees = data.get('employees', [])

    nodes = []
    edges = []

    # Sets to track unique nodes to avoid duplicates
    seen_nodes = set()
    
    # Mandate data
    mandates = {
        "VA": 12.77,
        "NY": 17.00,
        "GA": 7.25
    }

    # Add Mandate Nodes
    for state, wage in mandates.items():
        mandate_id = f"MANDATE_{state}"
        nodes.append({
            "id": mandate_id,
            "label": "Mandate",
            "properties": {
                "state": state,
                "minimum_wage": wage
            }
        })
        seen_nodes.add(mandate_id)

    for emp in employees:
        emp_id = emp['employee_id']
        name = f"{emp['first_name']} {emp['last_name']}"
        status = "resigned" if emp.get('resign_date') != "working" else "active"
        salary_type = emp.get('salary_type')
        
        # 1. Employee Node
        nodes.append({
            "id": emp_id,
            "label": "Employee",
            "properties": {
                "id": emp_id,
                "name": name,
                "title": emp['title'],
                "department": emp['department'],
                "status": status,
                "salary_type": salary_type
            }
        })

        # 2. Role Node
        role_id = f"ROLE_{emp['title'].replace(' ', '_')}_{emp['department'].replace(' ', '_')}"
        if role_id not in seen_nodes:
            nodes.append({
                "id": role_id,
                "label": "Role",
                "properties": {
                    "title": emp['title'],
                    "department": emp['department']
                }
            })
            seen_nodes.add(role_id)
        
        # Edge: HAS_ROLE
        edges.append({
            "source": emp_id,
            "target": role_id,
            "type": "HAS_ROLE"
        })

        # 3. Location Node (formerly Store)
        loc = emp.get('location', {})
        store_id = loc.get('store_id')
        city = loc.get('city')
        state = loc.get('state')

        if store_id:
            if store_id not in seen_nodes:
                nodes.append({
                    "id": store_id,
                    "label": "Location",
                    "properties": {
                        "id": store_id,
                        "city": city,
                        "state": state
                    }
                })
                seen_nodes.add(store_id)
            
            # Edge: locatedAt (formerly WORKS_AT)
            edges.append({
                "source": emp_id,
                "target": store_id,
                "type": "locatedAt"
            })

            # 4. Region Node (formerly Jurisdiction)
            region_id = f"STATE_{state}"
            if region_id not in seen_nodes:
                nodes.append({
                    "id": region_id,
                    "label": "Region",
                    "properties": {
                        "name": state
                    }
                })
                seen_nodes.add(region_id)

                # Edge: SUBJECT_TO (Region -> Mandate)
                mandate_id = f"MANDATE_{state}"
                if mandate_id in seen_nodes:
                    edges.append({
                        "source": region_id,
                        "target": mandate_id,
                        "type": "SUBJECT_TO"
                    })

            # Edge: inRegion (formerly LOCATED_IN)
            edges.append({
                "source": store_id,
                "target": region_id,
                "type": "inRegion"
            })

        # 5. reportsTo Edge (formerly REPORTS_TO)
        manager_id = emp.get('reports_to_id')
        if manager_id:
            edge_status = "historical" if status == "resigned" else "active"
            edges.append({
                "source": emp_id,
                "target": manager_id,
                "type": "reportsTo",
                "properties": {
                    "status": edge_status
                }
            })

    graph_data = {
        "nodes": nodes,
        "edges": edges
    }

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"Graph transformation complete. Saved to {output_file}")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")

if __name__ == "__main__":
    transform_to_graph()
