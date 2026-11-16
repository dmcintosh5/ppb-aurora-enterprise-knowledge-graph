import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import networkx as nx

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

def create_employee_nodes(driver, employees_data):
    """
    Creates 'Employee' nodes in Neo4j from a list of employee dictionaries.
    
    Args:
        driver: The Neo4j GraphDatabase driver instance.
        employees_data (list): A list of dictionaries, where each dictionary
                               represents an employee record.
    """
    print(f"[*] Creating {len(employees_data)} Employee nodes in Neo4j...")
    
    # Cypher query to create an Employee node with all properties
    # MERGE is used to create the node if it doesn't exist (based on employee_id)
    # and SET all properties. This handles potential re-runs without duplicating nodes.
    cypher_create_employee = """
    MERGE (e:Employee {employee_id: $employee_id})
    SET e.first_name = $first_name,
        e.last_name = $last_name,
        e.title = $title,
        e.department = $department,
        e.city = $city,
        e.state = $state,
        e.store_id = $store_id,
        e.employment_type = $employment_type,
        e.salary_type = $salary_type,
        e.annual_salary = $annual_salary,
        e.performance_tier = $performance_tier,
        e.hourly_rate = $hourly_rate // Optional, will be null if not provided
    RETURN e.employee_id
    """
    
    with driver.session() as session:
        for employee in employees_data:
            # Prepare parameters for the Cypher query
            params = {
                "employee_id": employee.get("employee_id"),
                "first_name": employee.get("first_name"),
                "last_name": employee.get("last_name"),
                "title": employee.get("title"),
                "department": employee.get("department"),
                "city": employee["location"].get("city"),
                "state": employee["location"].get("state"),
                "store_id": employee["location"].get("store_id"), # Will be None if not present
                "employment_type": employee.get("employment_type"),
                "salary_type": employee.get("salary_type"),
                "annual_salary": employee.get("annual_salary"),
                "performance_tier": employee.get("performance_tier"),
                "hourly_rate": employee.get("hourly_rate") # Will be None if not present
            }
            
            session.run(cypher_create_employee, params)
    print("[+] Employee nodes creation complete.")

def create_store_nodes_and_relationships(driver, employees_data):
    """
    Creates 'Store' and 'State' nodes and 'LOCATED_IN' relationships in Neo4j.
    """
    print("[*] Creating Store and State nodes and LOCATED_IN relationships...")
    unique_stores = {} # To store unique store_id -> {city, state}
    unique_states = set()

    for employee in employees_data:
        location = employee.get("location", {})
        store_id = location.get("store_id")
        city = location.get("city")
        state = location.get("state")

        if store_id:
            unique_stores[store_id] = {"city": city, "state": state}
        if state:
            unique_states.add(state)
    
    with driver.session() as session:
        # Create State nodes
        for state_name in unique_states:
            session.run("MERGE (s:State {name: $state_name})", state_name=state_name)
        
        # Create Store nodes and LOCATED_IN relationships
        for store_id, details in unique_stores.items():
            session.run("""
                MERGE (s:Store {store_id: $store_id})
                SET s.city = $city, s.state = $state
                WITH s
                MATCH (st:State {name: $state})
                MERGE (s)-[:LOCATED_IN]->(st)
            """, store_id=store_id, city=details["city"], state=details["state"])
    print("[+] Store and State nodes and LOCATED_IN relationships creation complete.")

def create_role_nodes(driver, employees_data):
    """
    Creates 'Role' nodes in Neo4j.
    """
    print("[*] Creating Role nodes...")
    unique_roles = set() # To store unique (title, department) tuples

    for employee in employees_data:
        title = employee.get("title")
        department = employee.get("department")
        if title and department:
            unique_roles.add((title, department))
    
    with driver.session() as session:
        for title, department in unique_roles:
            session.run("""
                MERGE (r:Role {title: $title, department: $department})
            """, title=title, department=department)
    print("[+] Role nodes creation complete.")

def create_employee_relationships(driver, employees_data):
    """
    Creates REPORTS_TO, WORKS_AT, and HAS_ROLE relationships in Neo4j.
    Includes join_date and resign_date as edge properties.
    """
    print("[*] Creating Employee relationships (REPORTS_TO, WORKS_AT, HAS_ROLE)...")
    with driver.session() as session:
        for employee in employees_data:
            employee_id = employee.get("employee_id")
            reports_to_id = employee.get("reports_to_id") # Assuming this field exists
            store_id = employee.get("location", {}).get("store_id")
            title = employee.get("title")
            department = employee.get("department")
            join_date = employee.get("join_date")
            resign_date = employee.get("resign_date")

            # REPORTS_TO relationship
            if reports_to_id:
                session.run("""
                    MATCH (e:Employee {employee_id: $employee_id})
                    MATCH (m:Employee {employee_id: $reports_to_id})
                    MERGE (e)-[:REPORTS_TO]->(m)
                """, employee_id=employee_id, reports_to_id=reports_to_id)
            
            # WORKS_AT relationship
            if store_id:
                session.run("""
                    MATCH (e:Employee {employee_id: $employee_id})
                    MATCH (s:Store {store_id: $store_id})
                    MERGE (e)-[:WORKS_AT {join_date: $join_date, resign_date: $resign_date}]->(s)
                """, employee_id=employee_id, store_id=store_id, join_date=join_date, resign_date=resign_date)
            
            # HAS_ROLE relationship
            if title and department:
                session.run("""
                    MATCH (e:Employee {employee_id: $employee_id})
                    MATCH (r:Role {title: $title, department: $department})
                    MERGE (e)-[:HAS_ROLE {join_date: $join_date, resign_date: $resign_date}]->(r)
                """, employee_id=employee_id, title=title, department=department, join_date=join_date, resign_date=resign_date)
    print("[+] Employee relationships creation complete.")

def export_neo4j_to_networkx(driver):
    """
    Exports the entire Neo4j graph into a NetworkX DiGraph.
    """
    print("[*] Exporting Neo4j graph to NetworkX DiGraph...")
    G = nx.DiGraph()
    
    with driver.session() as session:
        # Fetch all nodes
        nodes_result = session.run("MATCH (n) RETURN n, labels(n) as labels")
        for record in nodes_result:
            node = record["n"]
            labels = record["labels"]
            node_id = node.id # Use internal Neo4j ID for NetworkX node ID
            properties = dict(node)
            properties["labels"] = labels # Store Neo4j labels as a property
            G.add_node(node_id, **properties)
            
        # Fetch all relationships
        relationships_result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m, type(r) as type")
        for record in relationships_result:
            start_node_id = record["n"].id
            end_node_id = record["m"].id
            rel_type = record["type"]
            rel_properties = dict(record["r"])
            G.add_edge(start_node_id, end_node_id, type=rel_type, **rel_properties)
            
    print(f"[+] Export complete. NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    # Example usage:
    json_file_path = "c:\\Users\\minhp\\Downloads\\Projects\\GSU_projects\\msa8700-cu-50-zelkova\\data\\raw\\ppb_directory_v7_historical.json"
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        employees_data = data["employees"]

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    # Clear existing data for a clean run (optional, use with caution)
    # print("[*] Clearing existing Neo4j data...")
    # with driver.session() as session:
    #     session.run("MATCH (n) DETACH DELETE n")
    # print("[+] Existing Neo4j data cleared.")

    create_employee_nodes(driver, employees_data) # Creates Employee nodes
    create_store_nodes_and_relationships(driver, employees_data) # Creates Store and State nodes and LOCATED_IN relationships
    create_role_nodes(driver, employees_data) # Creates Role nodes
    create_employee_relationships(driver, employees_data) # Creates relationships between Employee, Store, and Role
    
    # Export to NetworkX (example usage)
    # network_graph = export_neo4j_to_networkx(driver)
    # print(f"NetworkX graph nodes: {network_graph.nodes(data=True)}")
    # print(f"NetworkX graph edges: {network_graph.edges(data=True)}")

    driver.close()