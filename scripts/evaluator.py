import os
import json
import nltk
import re
from typing import List, Dict, Any
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class TokenEvaluator:
    """
    Phase 2: The Token Metric Suite
    Calculates ROUGE-L, F1, and Exact Match (EM) similarity metrics.
    """
    
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization using NLTK."""
        return nltk.word_tokenize(text.lower())

    def calculate_em(self, expected: str, actual: str) -> float:
        """Calculates Exact Match (EM)."""
        return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0

    def calculate_f1(self, expected: str, actual: str) -> float:
        """Calculates Token-level F1 score."""
        exp_tokens = self.tokenize(expected)
        act_tokens = self.tokenize(actual)
        
        common = set(exp_tokens) & set(act_tokens)
        if not act_tokens or not exp_tokens:
            return 0.0
            
        precision = len(common) / len(act_tokens)
        recall = len(common) / len(exp_tokens)
        
        if (precision + recall) == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)

    def calculate_rouge_l(self, expected: str, actual: str) -> float:
        """Calculates ROUGE-L (Longest Common Subsequence) recall."""
        exp_tokens = self.tokenize(expected)
        act_tokens = self.tokenize(actual)
        
        if not exp_tokens:
            return 0.0
            
        m, n = len(exp_tokens), len(act_tokens)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif exp_tokens[i-1] == act_tokens[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
                    
        lcs_val = L[m][n]
        return lcs_val / m

    def evaluate_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of results for Token Metrics.
        """
        results = []
        for item in data:
            expected = item.get('expected_answer', '')
            actual = item.get('rag_response', '')
            category = item.get('category', 'Unknown')
            
            metrics = {
                "query_id": item.get('query_id'),
                "category": category,
                "em": self.calculate_em(expected, actual),
                "f1": self.calculate_f1(expected, actual),
                "rouge_l": self.calculate_rouge_l(expected, actual)
            }
            results.append(metrics)
            
        df = pd.DataFrame(results)
        
        summary = {
            "avg_em": round(df['em'].mean(), 4) if not df.empty else 0.0,
            "avg_f1": round(df['f1'].mean(), 4) if not df.empty else 0.0,
            "avg_rouge_l": round(df['rouge_l'].mean(), 4) if not df.empty else 0.0,
            "total_queries": len(df)
        }
        
        cat_summary = {}
        if not df.empty:
            cat_summary = df.groupby('category')[['em', 'f1', 'rouge_l']].mean().round(4).to_dict('index')
            
        return {
            "summary": summary,
            "category_summary": cat_summary,
            "details": results
        }

class RagasEvaluator:
    """
    Phase 3: RAGAS Integration (The "Semantic" Judge)
    Focuses on Faithfulness, Relevancy, and Context Utilization.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _get_json_response(self, prompt: str) -> Dict[str, Any]:
        """Helper to get and parse JSON from LLM with robust fallbacks."""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 1. Try finding JSON within curly braces (Standard)
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            
            # 2. Try cleaning common Markdown code blocks
            clean_content = content.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                pass
                
            # 3. Last resort: Try to extract numbers manually if parsing fails
            faith = re.search(r'"faithfulness":\s*([\d\.]+)', content)
            rel = re.search(r'"relevancy":\s*([\d\.]+)', content)
            util = re.search(r'"utilization":\s*([\d\.]+)', content)
            if faith and rel:
                return {
                    "faithfulness": float(faith.group(1)),
                    "relevancy": float(rel.group(1)),
                    "utilization": float(util.group(1)) if util else 0.5,
                    "reasoning": "Manually extracted from malformed JSON."
                }
                
        except Exception as e:
            print(f"      [!] Semantic judgment error: {e}")
        return {"faithfulness": 0.0, "relevancy": 0.0, "utilization": 0.0, "error": "Parsing failed"}

    def evaluate_trinity(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """Calculates the Trinity of RAG metrics in a single pass."""
        prompt = f"""
        You are a strict HR Auditor. Evaluate the RAG response based ONLY on the provided context.
        
        CONTEXT NOTE: The context may include unstructured PDF text AND structured 'Graph Data' in JSON format. 
        Treat BOTH as valid sources of truth. If the answer is present in the Graph Data JSON, it is FAITHFUL.
        
        QUESTION: {question}
        CONTEXT: {context}
        RESPONSE: {answer}
        
        Provide scores from 0.0 to 1.0:
        1. FAITHFULNESS: Are all facts in the response directly supported by the context (including Graph Data)?
        2. RELEVANCY: Does the response address the specific question asked?
        3. UTILIZATION: Did the response use the context instead of internal knowledge?
        
        Return ONLY a raw JSON object. No preamble, no markdown, no explanation outside the JSON.
        {{
          "faithfulness": 0.0,
          "relevancy": 0.0,
          "utilization": 0.0,
          "reasoning": "one sentence"
        }}
        """
        return self._get_json_response(prompt)

    def evaluate_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of results for RAGAS Metrics.
        Expected keys: 'question', 'retrieved_context', 'rag_response', 'category'
        """
        results = []
        for item in data:
            question = item.get('question', '')
            context = item.get('retrieved_context', '')
            answer = item.get('rag_response', '')
            category = item.get('category', 'Unknown')
            
            print(f"[*] Semantic Judging: '{question[:50]}...'")
            judgment = self.evaluate_trinity(question, context, answer)
            
            metrics = {
                "query_id": item.get('query_id'),
                "category": category,
                "faithfulness": judgment.get('faithfulness', 0.0),
                "relevancy": judgment.get('relevancy', 0.0),
                "utilization": judgment.get('utilization', 0.0),
                "reasoning": judgment.get('reasoning', 'N/A')
            }
            results.append(metrics)
            
        df = pd.DataFrame(results)
        
        summary = {
            "avg_faithfulness": round(df['faithfulness'].mean(), 4) if not df.empty else 0.0,
            "avg_relevancy": round(df['relevancy'].mean(), 4) if not df.empty else 0.0,
            "avg_utilization": round(df['utilization'].mean(), 4) if not df.empty else 0.0,
            "total_queries": len(df)
        }
        
        cat_summary = {}
        if not df.empty:
            cat_summary = df.groupby('category')[['faithfulness', 'relevancy', 'utilization']].mean().round(4).to_dict('index')
            
        return {
            "summary": summary,
            "category_summary": cat_summary,
            "details": results
        }

if __name__ == "__main__":
    # Test with sample data if executed directly
    print("\n--- Testing TokenEvaluator ---")
    token_eval = TokenEvaluator()
    sample_token_data = [
        {
            "query_id": "test_1",
            "category": "Factoid",
            "expected_answer": "The core values are Integrity, Community, and Safety.",
            "rag_response": "Our core values include integrity, community, and safety."
        }
    ]
    token_results = token_eval.evaluate_batch(sample_token_data)
    print(json.dumps(token_results, indent=2))

    print("\n--- Testing RagasEvaluator (Requires LLM) ---")
    try:
        from scripts.agent import PPBAgent
        agent = PPBAgent()
        ragas_eval = RagasEvaluator(llm=agent.smart_llm)
        sample_ragas_data = [
            {
                "query_id": "test_semantic",
                "category": "Policy",
                "question": "What are the core values?",
                "retrieved_context": "The core values are Integrity, Community, and Safety.",
                "rag_response": "The core values are Integrity, Community, and Safety."
            }
        ]
        ragas_results = ragas_eval.evaluate_batch(sample_ragas_data)
        print(json.dumps(ragas_results, indent=2))
        agent.close()
    except Exception as e:
        print(f"[!] RagasEvaluator test skipped or failed: {e}")
