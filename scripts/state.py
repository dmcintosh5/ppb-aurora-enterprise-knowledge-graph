from typing import TypedDict, List, Annotated, Optional
import operator

class HRProjectState(TypedDict):
    """
    The 'Shared Notebook' for the PPB DAIS Agents.
    This state tracks the lifecycle of a single document as it moves 
    through the multi-agent pipeline.
    """
    # 1. Source Information
    doc_name: str
    doc_path: str
    doc_type: str  # 'policy', 'process', or 'directory'
    region: str    # GA, NY, etc.
    metadata: dict # Internal document properties (author, title, etc.)

    # 2. Raw & Intermediate Data
    raw_text: str
    chunks: List[str]
    embeddings: List[List[float]]

    # 3. Agent Extractions (Accumulated by Specialists)
    # Annotated with operator.add so multiple agents can contribute to the lists
    entities: Annotated[List[dict], operator.add]
    relationships: Annotated[List[dict], operator.add]

    # 4. Pipeline Infrastructure Status
    embeddings_generated: bool
    neo4j_node_count: int
    neo4j_status: str  # 'pending', 'success', or 'error'

    # 5. Error Tracking
    errors: List[str]

    # 6. Next Step (For the Router/Orchestrator)
    next_node: str
