from scripts.agent import PPBAgent
from langchain_core.messages import HumanMessage

def run_tests():
    print("Initializing Agent...")
    agent = PPBAgent()
    
    def test_query(user_id, query_text):
        initial_state = {
            "user_id": user_id,
            "user_context": {},
            "messages": [HumanMessage(content=query_text)],
            "query": query_text,
            "search_query": "",
            "intent": {},
            "extracted_entities": [],
            "resolved_entities": [],
            "ambiguity_matches": {},
            "graph_data": {},
            "status": "searching",
            "retries": 0,
            "correction_notes": "",
            "faithfulness_score": 0.0,
            "relevance_score": 0.0
        }
        
        print(f"\n--- Testing Query: '{query_text}' as User: {user_id} ---")
        final_state = agent.workflow.invoke(initial_state)
        
        user_ctx = final_state.get("user_context", {})
        print(f"[*] Authenticated User: {user_ctx.get('name')} ({user_ctx.get('title')})")
        print(f"[*] Parsed Intent: {final_state.get('intent')}")
        print(f"[*] Search Keywords: {final_state.get('search_keywords')}")
        print(f"[*] Extracted Entities: {final_state.get('extracted_entities')}")
        print(f"[*] Final Response:\n{final_state['messages'][-1].content}")

    # Test 1: Organizational query relying on implicit "my" pronoun
    # PPB-002 is Jules Finn (COO), who reports to PPB-001 (Terry Chen, CEO)
    test_query("PPB-002", "Who is my manager?")
    
    # Test 2: Policy query with exact keywords
    test_query("PPB-002", "What is our policy on bereavement leave?")

if __name__ == "__main__":
    run_tests()
