# GEMINI.md - Peachtree Pantry & Beverage Co. (PPB) DAIS Project

---
project_id: msa8700-zelkova-final
---

## Project Overview
The **Document-Driven Agentic Intelligence System (DAIS)** is an Enterprise HR Knowledge Graph & Multi-Agent System developed for Peachtree Pantry & Beverage Co. (PPB). Its primary goal is to provide a robust, containerized data pipeline that transforms unstructured HR policies (PDFs) and structured employee data into a searchable Neo4j Knowledge Graph with vector support, enabling an AI HR Assistant.

### Core Architecture
- **Data Ingestion:** `scripts/ingest.py` performs semantic chunking of PDFs and JSON data, adding metadata for regional and document type filtering.
- **Graph Transformation:** `scripts/transform_to_graph.py` converts flat HR data (`ppb_directory_v7_historical.json`) into a Graph-JSON structure supporting multi-hop reasoning.
- **Vector Embeddings:** `scripts/embed.py` generates high-dimensional vectors (768d) using the `nomic-embed-text` model via Ollama.
- **Knowledge Graph:** `scripts/db_upload.py` and `scripts/load_ppb_graph.py` persist document, employee, and jurisdictional nodes in Neo4j.
- **AURORA Agent:** `scripts/agent.py` implements the **Automated User Response & Organizational Resource Agent**, a RAG system that performs multi-hop reasoning over the Neo4j graph.

### Agent Operational Mandates
- **GIT CONTROL:** NEVER execute `git push` or stage/commit changes without explicit instructions from the user.
- **BUILD CONTROL:** DO NOT execute `docker-compose up --build`. Use `docker-compose run app python scripts/run_data_pipeline.py` for all pipeline executions and data rebuilds.

### Technical Stack
- **Language:** Python 3.12
- **Orchestration:** Docker Compose
- **Database:** Neo4j (Graph + Vector)
- **LLM/Embeddings:** Ollama (`llama3.1`, `nomic-embed-text-v1.5`)
- **Frameworks:** LangChain, LangGraph, SentenceTransformers, PyMuPDF, Neo4j Python Driver

---

## Building and Running

### Prerequisites
- Docker Desktop must be installed and running.
- A `.env` file should be present in the root directory (configured with `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, etc.).

### Key Commands
| Action | Command |
| :--- | :--- |
| **Start Services & Run Pipeline** | `docker-compose up --build` |
| **Stop All Services** | `docker-compose down` |
| **Transform HR Data to Graph** | `docker-compose run app python scripts/transform_to_graph.py` |
| **Load Graph to Neo4j** | `docker-compose run app python scripts/load_ppb_graph.py` |
| **Run AURORA Agent** | `docker-compose run app python scripts/agent.py` |
| **Run Data Pipeline Manually** | `docker-compose run app python scripts/run_data_pipeline.py` |
| **Execute Validation Tests** | `docker-compose run app python tests/test_data_pipeline.py` |
| **Access Neo4j Browser** | http://localhost:7474 (User: `neo4j`, Pass: `password`) |

---

## Development Conventions

### Data Pipeline Flow
1. **Ingest (`ingest.py`):** Raw files in `data/raw/` are processed into `data/processed/*.json`. Paragraph-level atomic chunking is used to preserve context.
2. **Transform (`transform_to_graph.py`):** Structured employee data is converted into a nodes-and-edges JSON (`data/processed/ppb_graph_adjacency.json`).
3. **Embed (`embed.py`):** Chunks are processed in batches (size 5) to generate embeddings, which are appended to the JSON files in `data/processed/`.
4. **Upload (`db_upload.py` & `load_ppb_graph.py`):** Data is uploaded to Neo4j. It creates Document, Chunk, Employee, Store, and Mandate nodes.

### Neo4j Schema
- **Nodes:**
  - `(:Document {name: string, author: string, type: string})`
  - `(:Chunk {text: string, embedding: float[], index: int})`
  - `(:Employee {id: string, name: string, status: string, salary_type: string})`
  - `(:Role {title: string, department: string})`
  - `(:Store {id: string, city: string, state: string})`
  - `(:Jurisdiction {name: string})`
  - `(:Mandate {state: string, minimum_wage: float})`
- **Relationships:**
  - `(Document)-[:HAS_CHUNK]->(Chunk)`
  - `(Employee)-[:REPORTS_TO {status: string}]->(Employee)`
  - `(Employee)-[:WORKS_AT]->(Store)`
  - `(Store)-[:LOCATED_IN]->(Jurisdiction)`
  - `(Jurisdiction)-[:SUBJECT_TO]->(Mandate)`
  - `(Employee)-[:HAS_ROLE]->(Role)`
- **Indexes:** 
  - A vector index `chunk_embedding` is created on `(Chunk.embedding)`.
  - Full-text indexes on `Chunk.text` and `Employee.name`.

### Testing & Validation
- **Acceptance Criteria:** Tests in `tests/test_data_pipeline.py` verify document counts, metadata accuracy, and ensure no empty chunks are ingested.
- **RAG Evaluation Suite ("The Evaluator"):**
    - **Retrieval Eval:** `docker-compose run app python scripts/evaluate_retrieval.py`
        - Metrics: Hit Rate@K, Mean Reciprocal Rank (MRR), nDCG.
    - **Response Eval:** `docker-compose run app python scripts/evaluate_responses.py`
        - Metrics: Token (F1, ROUGE-L) + Semantic (Faithfulness, Relevancy).
- **Implementation Roadmap:**
    - **Phase 2 (Completed):** Token Metric Suite (EM, F1, ROUGE-L).
    - **Phase 3 (Completed):** RAGAS Semantic Integration (Faithfulness, Relevancy).
    - **Phase 4 (Completed):** Unified Comparative Reporting & Mimicry Hallucination Detection.

### Environment Configuration
- The system relies on several environment variables:
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`