# M04 Milestone - Baseline Evaluation Framework

## Project Overview
The **Document-Driven Agentic Intelligence System (DAIS)** is an Enterprise HR Knowledge Graph & Multi-Agent System developed for Peachtree Pantry & Beverage Co. (PPB). It integrates unstructured HR policies (PDFs) and structured employee data into a Neo4j Knowledge Graph, enabling a RAG (Retrieval-Augmented Generation) assistant named **AURORA** to answer complex HR inquiries.

### Core Components
- **Data Pipeline:** Ingests PDFs and CSVs, performs semantic chunking, generates embeddings (Ollama), and transforms structured data into a graph.
- **Knowledge Graph:** Neo4j database storing Documents, Chunks, Employees, Roles, Stores, and Jurisdictions.
- **AURORA Agent:** A LangGraph-based multi-agent system that performs intent parsing, entity resolution, graph traversal, and hybrid vector search.
- **Evaluator:** A comprehensive suite calculating token-level metrics (F1, ROUGE-L) and semantic "Trinity" metrics (Faithfulness, Relevancy, Utilization).

---

## Baseline Evaluation Results

The evaluation was performed against a gold set of **63 queries** covering Factoid, Graph-Relational, and Multi-Hop categories.

### Summary Metrics
| Metric | Value |
| :--- | :--- |
| **Total Queries** | 63 |
| **Avg Latency** | 1.84s |
| **Avg F1 Score** | 0.3949 |
| **Avg ROUGE-L** | 0.4464 |
| **Avg Faithfulness** | 0.5651 |
| **Avg Relevancy** | 0.5786 |
| **Mimicry Hallucinations** | 2 |

### Category Breakdown
| Category | F1 Score | ROUGE-L | Faithfulness | Relevancy |
| :--- | :--- | :--- | :--- | :--- |
| **Factoid** | 0.3413 | 0.4319 | 0.5297 | 0.5527 |
| **Graph-Relational** | 0.6667 | 0.6667 | 0.6111 | 0.6111 |
| **Multi-Hop** | 0.0312 | 0.0179 | 0.6250 | 0.6250 |

---

## Error Analysis

1.  **Ambiguity Handling:** Queries for common names (e.g., "Jesse Reed") trigger the agent's disambiguation logic ("Multiple matches found"). While this is correct behavior for a production assistant, it results in a score of 0 in the evaluation since the "expected answer" is a specific individual's manager.
2.  **Specific vs. Concise Answers:** In Multi-Hop queries, the agent often provides the correct data (e.g., "7.25") but fails to match the highly descriptive ground truth format (e.g., "$7.25 (Georgia minimum wage)"), leading to low F1/ROUGE scores despite high semantic faithfulness.
3.  **Entity Resolution Mismatches:** Some graph lookups failed when the LLM-extracted entity didn't perfectly match the Neo4j node (e.g., extracting "Logan Lee" but failing to navigate to the "Role" property correctly in the final response).
4.  **Judge Inconsistency:** In some cases, the LLM judge (Gemma 3) provided 0.0 scores for perfectly valid responses, suggesting a need for more robust evaluation prompting or a larger model for judging.

---

## Concrete Improvement Ideas

1.  **Enhanced Multi-Step Reasoning (Chain-of-Thought):**
    Improve the `responder_node` by implementing a "reasoning" step where the LLM explicitly links the retrieved graph data (Minimum Wage) with the retrieved store location before generating the final answer. This will bridge the semantic gap in Multi-Hop queries.

2.  **Context Reranking with Cross-Encoders:**
    Currently, the system uses hybrid search (Vector + Full-text). Implementing a secondary reranking step using a local cross-encoder model would ensure that the most semantically relevant policy chunks are at the top of the context, improving Factoid F1 and ROUGE-L scores by reducing noise.

3.  **Fuzzy Entity Resolution & Alias Support:**
    Upgrade the `entity_resolver_node` to use Levenshtein distance or a fuzzy matching library (like `thefuzz`) when querying Neo4j. Additionally, adding an `alias` property to Employee nodes (e.g., "AJ" for "Andrew") would significantly improve Graph-Relational retrieval success.
