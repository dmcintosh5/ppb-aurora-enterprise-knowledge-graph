import streamlit as st
from scripts.agent import PPBAgent
import time
import pandas as pd
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page Configuration
st.set_page_config(
    page_title="PPB HR Knowledge Portal", 
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main Background - Deep Charcoal/Black */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0F172A !important; 
        color: #F8FAFC !important;
    }

    /* Universal Text Visibility */
    .stMarkdown, p, span, label, h1, h2, h3, li {
        color: #F1F5F9 !important; 
    }

    /* Header Section - The "Yellow" Accent */
    .main-header-container {
        background: #1E293B;
        padding: 2.5rem;
        border-radius: 1rem;
        border-left: 10px solid #FACC15; /* Bright Yellow Border */
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: #FACC15 !important; /* Primary Yellow */
        letter-spacing: -0.01em;
        text-transform: uppercase;
    }

    /* User Message Bubbles - Industrial Grey */
    .chat-bubble-user {
        background: #334155; 
        color: #FFFFFF !important;
        border-radius: 1rem 1rem 0 1rem;
        padding: 1.2rem;
        max-width: 85%;
        margin-left: auto;
        margin-bottom: 1.2rem;
        border: 1px solid #475569;
    }

    /* Assistant Message Bubbles - Deep Black with Yellow Border */
    .chat-bubble-assistant {
        background: #020617;
        color: #F1F5F9 !important;
        border-radius: 1rem 1rem 1rem 0;
        padding: 1.5rem;
        max-width: 85%;
        border: 1px solid #FACC15; /* Yellow Accent Border */
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }

    /* Sidebar - Grey/Black mix */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 2px solid #334155;
    }
    
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #94A3B8 !important;
    }

    /* Input Box - Dark Grey with Yellow Focus */
    .stTextInput>div>div>input {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 2px solid #334155 !important;
        border-radius: 8px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #FACC15 !important;
        box-shadow: 0 0 10px rgba(250, 204, 21, 0.3);
    }

    /* Tabs - Yellow selection */
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FACC15 !important;
        border-bottom: 3px solid #FACC15 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FACC15 !important;
        color: #020617 !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for database stats
def get_db_stats(agent):
    if not agent:
        return None
    cypher = """
    MATCH (d:Document)
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c)
    RETURN d.name as Document, count(c) as Chunks
    """
    try:
        with agent.driver.session() as session:
            result = session.run(cypher)
            return pd.DataFrame([r.data() for r in result])
    except:
        return None

# Main Layout
st.markdown('<div class="main-header-container">', unsafe_allow_html=True)
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown('<div style="font-size: 4rem;">🏢</div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<h1 class="main-header">Peachtree Pantry & Beverage</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.2rem; font-weight: 500;">DAIS v2.0 • Enterprise HR Intelligence Graph</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_region" not in st.session_state:
    st.session_state.user_region = "GA"
if "user_role" not in st.session_state:
    st.session_state.user_role = "Hotline Cook"

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #1e3a8a;'>🎛️ Command Center</h2>", unsafe_allow_html=True)

    if "agent" not in st.session_state:
        st.warning("⚠️ Connection Required")
        if st.button("🚀 Initialize Neural Engine", type="primary", use_container_width=True):
            with st.spinner("Connecting to Neo4j and Ollama..."):
                try:
                    st.session_state.agent = PPBAgent()
                    st.success("Connected successfully!")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
    else:
        st.success("🟢 System Synchronized")
        st.caption("Neo4j Vector Index: READY")

        st.markdown("---")
        st.markdown("### 🌍 User Context")
        st.session_state.user_region = st.selectbox("Current Region", ["GA", "NY", "VA"], index=["GA", "NY", "VA"].index(st.session_state.user_region))
        st.session_state.user_role = st.text_input("User Role", value=st.session_state.user_role)

        st.markdown("---")
        st.markdown("### ⚡ Rapid Fire")
        suggestions = [
            "Who manages Terry Chen?",
            "What is the PTO policy?",
            "Show IT Department",
            "SOP for hiring process"
        ]
        for query in suggestions:
            if st.button(f"🔍 {query}", use_container_width=True, key=f"sug_{query}"):
                st.session_state.pending_query = query

        st.markdown("---")
        if st.button("🧹 Flush Buffer (Reset)", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main Content Area
if "agent" not in st.session_state:
    st.info("👋 Initialize the system from the sidebar to begin secure session.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>🧠 Latent Space</h4><p>Multi-dimensional vector embeddings for semantic policy retrieval.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>🔗 Knowledge Graph</h4><p>High-fidelity relationship mapping of employee entities.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>🚀 Inference Engine</h4><p>Localized Llama 3 models for private data processing.</p></div>', unsafe_allow_html=True)
else:
    tab_chat, tab_directory, tab_library, tab_system = st.tabs([
        "💬 Intelligence Hub",
        "🗂️ Entity Registry",
        "📚 Knowledge Assets",
        "⚙️ Core Diagnostics"
    ])

    with tab_chat:
        if not st.session_state.messages:
            st.session_state.messages = [
                {"role": "assistant", "content": "Security handshake complete. I am the PPB AI Assistant. How can I assist you with HR intelligence today?", "time": datetime.now().strftime("%H:%M")}
            ]

        # Handle pending query
        if "pending_query" in st.session_state:
            prompt = st.session_state.pop("pending_query")
        else:
            prompt = st.chat_input("Input command or question...")

        # Chat Display
        for idx, message in enumerate(st.session_state.messages):
            is_user = message["role"] == "user"
            bubble_class = "chat-bubble-user" if is_user else "chat-bubble-assistant"
            timestamp = message.get("time", datetime.now().strftime("%H:%M"))

            st.markdown(f"""
            <div class="message-meta" style="{'justify-content: flex-end;' if is_user else ''}">
                <span>{'👤 YOU' if is_user else '🏢 PPB AI'}</span>
                <span>•</span>
                <span>{timestamp}</span>
            </div>
            <div class="{bubble_class}">
            """, unsafe_allow_html=True)
            
            st.markdown(message["content"])
            st.markdown("</div>", unsafe_allow_html=True)

            if not is_user and idx == len(st.session_state.messages) - 1 and idx > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                feedback = st.feedback("stars", key=f"feedback_{idx}")
                if feedback is not None:
                    st.session_state.agent.save_feedback(
                        st.session_state.messages[idx-1]["content"],
                        message["content"],
                        feedback + 1,
                        "Modern UI"
                    )
                    st.toast("Feedback logged to Graph.")

        if prompt:
            t = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({"role": "user", "content": prompt, "time": t})
            
            # Prepare state for agent
            # Note: We need to adapt st.session_state.messages (list of dicts) to ChatState.messages (list of BaseMessage)
            history = []
            for msg in st.session_state.messages[:-1]: # exclude the current prompt which we'll pass explicitly
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))
            
            # Use the 'ask' method or invoke workflow directly to handle the new state fields
            # Since 'ask' creates its own initial state, we'll bypass it to inject region/role
            initial_state = {
                "user_id": "", 
                "user_context": {}, 
                "messages": history + [HumanMessage(content=prompt)], 
                "context": [], 
                "reranked_context": [], 
                "query": prompt, 
                "search_query": "", 
                "user_region": st.session_state.user_region,
                "user_role": st.session_state.user_role,
                "intent": {}, 
                "extracted_entities": [], 
                "resolved_entities": [], 
                "ambiguity_matches": {}, 
                "graph_data": {}, 
                "status": "searching", 
                "retries": 0, 
                "correction_notes": "", 
                "faithfulness_score": 0.0, 
                "relevance_score": 0.0, 
                "confidence_score": 1.0
            }
            
            with st.spinner("Processing through Neural Workflow..."):
                try:
                    final_state = st.session_state.agent.workflow.invoke(initial_state)
                    response_content = final_state["messages"][-1].content
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "time": datetime.now().strftime("%H:%M")
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing query: {e}")

    with tab_directory:
        st.subheader("Graph Entity Search")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            emp_name = st.text_input("Entity Lookup", placeholder="e.g. Sidney Heller")
        with search_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("Query Graph", type="primary", use_container_width=True)

        if search_btn or emp_name:
            with st.spinner("Traversing Nodes..."):
                # Use a simplified state for search
                initial_state = {
                    "messages": [HumanMessage(content=f"Tell me about {emp_name}")],
                    "query": f"Tell me about {emp_name}",
                    "user_region": st.session_state.user_region,
                    "user_role": st.session_state.user_role,
                    # ... other required fields
                    "context": [], "reranked_context": [], "search_query": "", "intent": {},
                    "extracted_entities": [], "resolved_entities": [], "ambiguity_matches": {},
                    "graph_data": {}, "status": "searching", "retries": 0, "correction_notes": "",
                    "user_id": "", "user_context": {}, "faithfulness_score": 0.0, "relevance_score": 0.0, "confidence_score": 1.0
                }
                res = st.session_state.agent.workflow.invoke(initial_state)
                ans = res["messages"][-1].content

                st.markdown("### Search Results")
                st.markdown(f'<div class="chat-bubble-assistant">{ans}</div>', unsafe_allow_html=True)

    with tab_library:
        st.subheader("Ingested Data Lake")
        stats_df = get_db_stats(st.session_state.agent)
        if stats_df is not None:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            with col_b:
                total_chunks = stats_df['Chunks'].sum()
                st.metric("Total Chunks", f"{total_chunks:,}")
                st.metric("Documents", len(stats_df))
            st.bar_chart(stats_df.set_index('Document')['Chunks'])

    with tab_system:
        st.subheader("Core Architecture Diagnostics")
        st.code(f"""
        [SYSTEM STATUS]
        - NEURAL_ENGINE: ONLINE
        - GRAPH_DATABASE: CONNECTED (bolt://neo4j-db:7687)
        - VECTOR_STORE: READY (768d Cosine)
        - PRIMARY_LLM: {st.session_state.agent.smart_llm.model}
        - SECONDARY_LLM: {st.session_state.agent.fast_llm.model}
        - CONTEXT: Region={st.session_state.user_region}, Role={st.session_state.user_role}
        """, language="text")
        st.markdown("#### Performance Trace")
        st.image("https://via.placeholder.com/1000x300.png?text=Agent+Execution+Trace+Visualization", use_column_width=True)
