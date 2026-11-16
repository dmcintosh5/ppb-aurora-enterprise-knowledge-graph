import os
import sys
import json
import time
import pandas as pd

# Ensure project root is in path to allow direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from scripts.agent import PPBAgent
from scripts.evaluator import TokenEvaluator, RagasEvaluator
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
GOLD_SET_PATH = os.path.join('tests', 'fixtures', 'retrieval_gold_set.json')
OUTPUT_JSON_REPORT = os.path.join('tests', 'final_evaluation_report.json')
OUTPUT_CSV_REPORT = os.path.join('tests', 'final_evaluation_report.csv')
SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", 30))

def run_comprehensive_evaluation():
    """
    Phase 4: Comparative Reporting ("The Evaluator")
    Consolidates Token and RAGAS metrics into a single unified pipeline.
    Identifies discrepancies between surface-level similarity and semantic faithfulness.
    """
    if not os.path.exists(GOLD_SET_PATH):
        print(f"[!] Gold set not found at {GOLD_SET_PATH}")
        return

    print("[*] Initializing Comprehensive Response Evaluator...")
    agent = PPBAgent()
    token_eval = TokenEvaluator()
    ragas_eval = RagasEvaluator(llm=agent.smart_llm)
    
    with open(GOLD_SET_PATH, 'r') as f:
        gold_set = json.load(f)

    # Use the sample size from environment or default to full set
    sample_size = min(len(gold_set), SAMPLE_SIZE)
    # Use a fixed seed for deterministic sampling during debugging if needed, 
    # but here we'll just take the top N.
    eval_set = gold_set[:sample_size]
    
    full_results = []
    
    print(f"\n[*] Starting Comprehensive Evaluation ({len(eval_set)} queries)\n")
    print(f"{'#' :<3} | {'Category' :<15} | {'Query' :<50}")
    print("-" * 75)
    
    for i, item in enumerate(eval_set):
        query = item['query_text']
        category = item.get('category', 'Unknown')
        print(f"{i+1:<3} | {category:<15} | {query[:50]}...")
        
        start_time = time.time()
        
        try:
            # 1. Run Agent
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "context": [],
                "reranked_context": [],
                "query": query,
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
            
            final_state = agent.workflow.invoke(initial_state)
            latency = time.time() - start_time
            ai_response = final_state['messages'][-1].content
            
            # 2. Extract Context
            context_list = [c.get('text', '') for c in final_state.get('reranked_context', [])]
            graph_data = final_state.get('graph_data', [])
            graph_context = f"Graph Data: {json.dumps(graph_data)}" if graph_data else ""
            full_context = "\n".join(context_list) + "\n" + graph_context
            
            # 3. Token Evaluation
            em = token_eval.calculate_em(item['expected_answer'], ai_response)
            f1 = token_eval.calculate_f1(item['expected_answer'], ai_response)
            rouge_l = token_eval.calculate_rouge_l(item['expected_answer'], ai_response)
            
            # 4. Semantic (RAGAS) Evaluation
            judgment = ragas_eval.evaluate_trinity(query, full_context, ai_response)
            
            # 5. Compile Result
            res_item = {
                "query_id": item.get('query_id'),
                "category": category,
                "question": query,
                "expected": item['expected_answer'],
                "actual": ai_response,
                "latency": round(latency, 2),
                "em": em,
                "f1": round(f1, 4),
                "rouge_l": round(rouge_l, 4),
                "faithfulness": judgment.get('faithfulness', 0.0),
                "relevancy": judgment.get('relevancy', 0.0),
                "utilization": judgment.get('utilization', 0.0),
                "reasoning": judgment.get('reasoning', 'N/A')
            }
            
            # Identify "Mimicry Hallucination" flag
            # (High ROUGE-L but Low Faithfulness)
            if rouge_l > 0.7 and res_item['faithfulness'] < 0.5:
                res_item['warning'] = "HIGH_MIMICRY_LOW_GROUNDING"
            else:
                res_item['warning'] = "NONE"
                
            full_results.append(res_item)
            
        except Exception as e:
            print(f"    [!] Error processing query {i+1}: {e}")
            continue

    # --- Reporting ---
    df = pd.DataFrame(full_results)
    
    # Save Reports
    df.to_csv(OUTPUT_CSV_REPORT, index=False)
    
    summary = {
        "avg_f1": round(df['f1'].mean(), 4),
        "avg_rouge_l": round(df['rouge_l'].mean(), 4),
        "avg_faithfulness": round(df['faithfulness'].mean(), 4),
        "avg_relevancy": round(df['relevancy'].mean(), 4),
        "avg_latency": round(df['latency'].mean(), 2),
        "mimicry_warnings": int((df['warning'] == "HIGH_MIMICRY_LOW_GROUNDING").sum()),
        "total_queries": len(df)
    }
    
    cat_summary = df.groupby('category')[['f1', 'rouge_l', 'faithfulness', 'relevancy']].mean().round(4).to_dict('index')
    
    report_data = {
        "summary": summary,
        "category_summary": cat_summary,
        "details": full_results
    }
    
    with open(OUTPUT_JSON_REPORT, 'w') as f:
        json.dump(report_data, f, indent=4)

    # --- Final Console Output ---
    print("\n" + "="*70)
    print("      DAIS COMPREHENSIVE RESPONSE EVALUATION: THE EVALUATOR      ")
    print("="*70)
    print(f"Total Queries:         {summary['total_queries']}")
    print(f"Avg Latency:           {summary['avg_latency']}s")
    print(f"Mimicry Hallucinations: {summary['mimicry_warnings']} (High ROUGE / Low Faith)")
    print("-" * 70)
    print(f"{'Category' :<15} | {'F1' :<8} | {'ROUGE-L' :<8} | {'Faith' :<8} | {'Rel' :<8}")
    print("-" * 70)
    for cat, scores in cat_summary.items():
        print(f"{cat:<15} | {scores['f1']:<8.4f} | {scores['rouge_l']:<8.4f} | {scores['faithfulness']:<8.4f} | {scores['relevancy']:<8.4f}")
    print("="*70)
    print(f"\n[+] JSON Report: {OUTPUT_JSON_REPORT}")
    print(f"[+] CSV Report:  {OUTPUT_CSV_REPORT}")

    agent.close()

if __name__ == "__main__":
    run_comprehensive_evaluation()
