import streamlit as st
import os
import json
import re
import pandas as pd
import difflib
from typing import List, Dict, Any, Optional, Set, Union
from neo4j import GraphDatabase, Driver, exceptions as neo4j_exceptions
import uuid

# --- LangChain/Mistral Imports ---
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# ---Visualization and url parsing ---
import plotly.express as px
import urllib.parse


# --- CRITICAL: CONFIGURE LANGSMITH BEFORE DEFINING CLASSES ---
# This block must sit here, at the global level, right after imports.
# It ensures the library picks up the config before the @traceable decorators run.

if "LANGSMITH_API_KEY" in st.secrets:
    # 1. Set the API Key
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    
    # 2. Set the Project Name
    os.environ["LANGCHAIN_PROJECT"] = "Testing_analysis_tool"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # 3. FORCE THE EU ENDPOINT (Set both variables for safety)
    # The error logs showed the app was defaulting to US. This forces EU.
    os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"

# -------------------------------------------------------------

def init_app():
    """
    Runs once on app startup. 
    Now simplified because LangSmith setup is handled globally above.
    """
    # Initialize Session ID
    if "app_session_id" not in st.session_state:
        st.session_state["app_session_id"] = str(uuid.uuid4())

    if st.session_state.get("has_tried_login", False):
        return

    # Existing credential logic...
    m_key = get_config("MISTRAL_API_KEY")
    n_uri = get_config("NEO4J_URI")
    n_user = get_config("NEO4J_USER", "neo4j")
    n_pass = get_config("NEO4J_PASSWORD")

    if n_uri and n_pass and m_key:
        success, msg = attempt_connection(n_uri, n_user, n_pass, m_key)
        if not success:
            st.toast(f"⚠️ Auto-login failed: {msg}", icon="⚠️")
    
    st.session_state.has_tried_login = True
# ==========================================
### NEW DATABOOK ###
# ==========================================
# 1. FALLBACK DB FUNCTIONS
# ==========================================

def get_db_driver():
    """Retrieves the cached driver from the main app state."""
    if "app_state" in st.session_state and "neo4j_creds" in st.session_state.app_state:
        creds = st.session_state.app_state["neo4j_creds"]
        uri = creds.get("uri")
        auth = creds.get("auth")
        if uri and auth:
            # We assume the main app has already validated this driver
            # Ideally, we reuse the exact object, but creating a lightweight driver here is safe
            # if we trust the credentials.
            return GraphDatabase.driver(uri, auth=auth)
    return None

def fetch_inventory_from_db():
    """Fallback: Generates the inventory dict by querying the live DB."""
    inventory = {"Object": {}, "Verb": {}, "Lexical": {}}
    driver = get_db_driver()
    
    if not driver:
        return {}
        
    try:
        with driver.session() as session:
            # 1. POPULATE OBJECTS (Nodes)
            labels_result = session.run("CALL db.labels()")
            labels = [r[0] for r in labels_result]
            
            for label in labels:
                q = f"MATCH (n:`{label}`) WHERE n.name IS NOT NULL RETURN n.name as name"
                names = [r["name"] for r in session.run(q)]
                if names:
                    inventory["Object"][label] = sorted(names)

            # 2. POPULATE VERBS (Relationships)
            # We list the Relationship Types as the "Labels"
            rels_result = session.run("CALL db.relationshipTypes()")
            rels = [r[0] for r in rels_result]
            
            for r_type in rels:
                # For relationships, we might not have a "name", so we leave the list empty 
                # or we could fetch distinct properties if your schema supports it.
                # This ensures the 'Verb' menu at least shows the types.
                inventory["Verb"][r_type] = [] 

            # 3. POPULATE LEXICAL
            # Assuming 'MENTIONED_IN' or similar for lexical graph
            # We can just initialize it or check if specific nodes exist
            inventory["Lexical"]["Document"] = []

    except Exception as e:
        st.warning(f"DB Fallback failed: {e}")
    finally:
        driver.close()
        
    return inventory


def fetch_sunburst_from_db(selector_type: str, label: str, names: list[str]) -> pd.DataFrame:
    """
    Fallback: Generates DataFrame via Cypher.
    Handles 'Object' (Node-Centric) and 'Verb' (Relationship-Centric) logic.
    """
    driver = get_db_driver()
    if not driver or not names:
        return pd.DataFrame()

    try:
        with driver.session() as session:
            # --- CASE A: RELATIONSHIP CENTRIC (VERB) ---
            if label == "Verb":
                # Names list contains Relationship Types (e.g. ['COMMUNICATION', 'PAID'])
                # We want to see: Edge Type -> Source Label -> Target Label
                # We use string manipulation to inject types safely because Cypher params can't handle dynamic types easily in this specific aggregation way
                # But safer is to use WHERE type(r) IN $names
                
                query = """
                MATCH (n)-[r]->(m)
                WHERE type(r) IN $names
                RETURN 
                    type(r) as edge, 
                    labels(n)[0] as source_node_label, 
                    labels(m)[0] as connected_node_label, 
                    count(*) as count,
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                LIMIT 2000
                """
                result = session.run(query, names=names)
            
            # --- CASE B: NODE CENTRIC (OBJECT) ---
            else:
                # Standard Logic
                query = f"""
                MATCH (n:`{label}`)-[r]->(m)
                WHERE n.name IN $names
                RETURN 
                    type(r) as edge, 
                    labels(n)[0] as node, 
                    coalesce(n.name, 'Unknown') as node_name, 
                    count(m) as count,
                    labels(m)[0] as connected_node_label, 
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                """
                result = session.run(query, names=names)

            data = [r.data() for r in result]
            return pd.DataFrame(data)
            
    except Exception as e:
        st.error(f"Live Query failed: {e}")
        return pd.DataFrame()
    finally:
        driver.close()
        
# ==========================================
# 2. HYBRID FETCHERS (GITHUB -> DB)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_inventory() -> dict:
    """
    Hybrid Fetcher:
    1. Try GitHub (Fast, Static)
    2. Fallback to DB (Slower, Fresh)
    """
    # 1. Try GitHub
    url = "https://raw.githubusercontent.com/ssopic/testing_app/main/sunburst_jsons/inventory.json"
    try:
        response = pd.read_json(url)
        return response.to_dict()
    except Exception:
        # 2. Fallback to DB
        return fetch_inventory_from_db()

@st.cache_data(ttl=3600)
def fetch_sunburst_data(selector_type: str, items: list[dict]) -> pd.DataFrame:
    """
    Hybrid Fetcher for Multiple Nodes (Mixed Labels):
    1. Iterates through the list of selected items (label, name).
    2. Tries to fetch GitHub JSON for each.
    3. Groups failures by label and falls back to DB in batches.
    4. Returns a concatenated DataFrame.
    
    items structure: [{'label': 'PERSON', 'name': 'Jeff'}, ...]
    """
    if not items:
        return pd.DataFrame()

    base_url = "https://raw.githubusercontent.com/ssopic/testing_app/main/sunburst_jsons/"
    prefix_map = {"Object": "node", "Verb": "relationship", "Lexical": "lexical"}
    file_prefix = prefix_map.get(selector_type, "node")

    dfs = []
    missing_by_label = {} # Group missing items by label for efficient DB query

    # 1. Try GitHub for each item individually
    for item in items:
        label = item['label']
        name = item['name']
        
        safe_label = label.lower().strip()
        safe_name = urllib.parse.quote(name.strip())
        filename = f"{file_prefix}_{safe_label}_{safe_name}.json"
        full_url = f"{base_url}{filename}"
        
        try:
            df_part = pd.read_json(full_url)
            if not df_part.empty:
                dfs.append(df_part)
            else:
                if label not in missing_by_label: missing_by_label[label] = []
                missing_by_label[label].append(name)
        except:
            if label not in missing_by_label: missing_by_label[label] = []
            missing_by_label[label].append(name)

    # 2. Fallback to DB for missing items (Batched by Label)
    for label, names in missing_by_label.items():
        if names:
            df_db = fetch_sunburst_from_db(selector_type, label, names)
            if not df_db.empty:
                dfs.append(df_db)

    # 3. Combine results
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

# ==========================================
# 3. FRAGMENT: WORKSPACE
# ==========================================

#@st.fragment
# ==============================================================================
# UPDATED BACKEND: VERB SUPPORT
# ==============================================================================

def fetch_sunburst_from_db(selector_type: str, label: str, names: list[str]) -> pd.DataFrame:
    """
    Fallback: Generates DataFrame via Cypher.
    Handles 'Object' (Node-Centric) and 'Verb' (Relationship-Centric) logic.
    """
    driver = get_db_driver()
    if not driver or not names:
        return pd.DataFrame()

    try:
        with driver.session() as session:
            # --- CASE A: RELATIONSHIP CENTRIC (VERB) ---
            if label == "Verb":
                # Names list contains Relationship Types (e.g. ['COMMUNICATION', 'PAID'])
                query = """
                MATCH (n)-[r]->(m)
                WHERE type(r) IN $names
                RETURN 
                    type(r) as edge, 
                    labels(n)[0] as source_node_label, 
                    labels(m)[0] as connected_node_label, 
                    count(*) as count,
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                LIMIT 2000
                """
                result = session.run(query, names=names)
            
            # --- CASE B: NODE CENTRIC (OBJECT) ---
            else:
                query = f"""
                MATCH (n:`{label}`)-[r]->(m)
                WHERE n.name IN $names
                RETURN 
                    type(r) as edge, 
                    labels(n)[0] as node, 
                    coalesce(n.name, 'Unknown') as node_name, 
                    count(m) as count,
                    labels(m)[0] as connected_node_label, 
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                """
                result = session.run(query, names=names)

            data = [r.data() for r in result]
            return pd.DataFrame(data)
            
    except Exception as e:
        st.error(f"Live Query failed: {e}")
        return pd.DataFrame()
    finally:
        driver.close()

@st.fragment
def render_explorer_workspace(selector_type, selected_items):
    c_mid, c_right = st.columns([2, 1])
    
    with c_mid:
        if not selected_items:
            st.info("👈 Select entities from the left and click 'Visualize'.")
            return

        names = [item['name'] for item in selected_items]
        st.subheader(f"Analysis: {len(names)} Items")

        # Fetch Data
        df = fetch_sunburst_data(selector_type, selected_items)

        if df.empty:
            st.warning("No data found.")
            return

        # Prepare Plotly Data
        if 'id_list' in df.columns:
            df['id_list_str'] = df['id_list'].astype(str)
            hover_cols = ['id_list_str']
        else:
            hover_cols = None

        # --- DYNAMIC HIERARCHY BASED ON TYPE ---
        if selector_type == "Verb":
            # Hierarchy: Edge Type -> Source Label -> Target Label
            path = ['edge', 'source_node_label', 'connected_node_label']
        else:
            # Hierarchy: Node Name -> Edge Type -> Target Label
            path = ['node_name', 'edge', 'connected_node_label']

        # Ensure columns exist before plotting
        valid_path = [col for col in path if col in df.columns]
        
        if not valid_path:
             st.error("Data columns missing for visualization.")
             return

        fig = px.sunburst(
            df, 
            path=valid_path, 
            values='count',
            color='edge' if 'edge' in df.columns else None,
            hover_data=hover_cols
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("Extraction")
        st.caption("Select data to add to Evidence Locker")

        # --- Cascading Filters ---
        edge_options = sorted(df['edge'].unique()) if 'edge' in df.columns else []
        selected_edge_filter = st.selectbox(
            "Filter by Relationship:",
            ["All"] + edge_options,
            key="filter_edge"
        )

        # 2. Target Label Filter
        if selected_edge_filter == "All":
            filtered_df_step1 = df
            target_options = sorted(df['connected_node_label'].unique()) if 'connected_node_label' in df.columns else []
        else:
            filtered_df_step1 = df[df['edge'] == selected_edge_filter]
            target_options = sorted(filtered_df_step1['connected_node_label'].unique()) if 'connected_node_label' in filtered_df_step1.columns else []

        selected_target_filter = st.selectbox(
            "Filter by Target Type:",
            ["All"] + target_options,
            key="filter_target"
        )

        # 3. Apply Final Filter
        if selected_target_filter == "All":
            final_filtered_df = filtered_df_step1
        else:
            final_filtered_df = filtered_df_step1[filtered_df_step1['connected_node_label'] == selected_target_filter]

        # 4. Flatten IDs
        def deep_flatten(container):
            for i in container:
                if isinstance(i, list):
                    yield from deep_flatten(i)
                elif pd.notna(i) and i != "":
                    yield str(i)

        all_ids = []
        if 'id_list' in final_filtered_df.columns:
            all_ids = list(deep_flatten(final_filtered_df['id_list']))
        
        unique_ids = list(set(all_ids))
        
        st.metric("Documents Found", len(unique_ids))
        with st.expander("Preview ID List", expanded=False):
            st.write(unique_ids)

        if st.button("Add to Locker", type="primary", use_container_width=True):
            if not unique_ids:
                st.error("No documents to add.")
            else:
                # Construct query description
                if len(names) > 1:
                    # FIX: Distinctly list verbs for relationship analysis so user knows what they selected
                    if selector_type == "Verb":
                        name_str = f"Verbs: {', '.join(names)}"
                    else:
                        name_str = f"Entities: {', '.join(names)}"
                else:
                    name_str = names[0]
                    
                query_desc = f"Manual Explorer: {name_str}"
                filters = []
                if selected_edge_filter != "All":
                    filters.append(f"Edge: {selected_edge_filter}")
                if selected_target_filter != "All":
                    filters.append(f"Target: {selected_target_filter}")
                
                if filters:
                    query_desc += f" ({', '.join(filters)})"

                payload = {
                    "query": query_desc,
                    "answer": f"Visual discovery found {len(unique_ids)} related documents.",
                    "ids": [str(uid) for uid in unique_ids]
                }
                
                if "evidence_locker" not in st.session_state.app_state:
                    st.session_state.app_state["evidence_locker"] = []
                    
                st.session_state.app_state["evidence_locker"].append(payload)
                st.toast(f"✅ Added {len(unique_ids)} docs to Locker!")
                
        current_count = len(st.session_state.app_state.get("evidence_locker", []))
        st.caption(f"Total items in Locker: {current_count}")
        
# ==========================================
# 4. MAIN SCREEN CONTROLLER
# ==========================================

def screen_databook():
    st.title("🧭 The Databook Explorer")
    
    inventory = fetch_inventory()
    
    # Initialize persistent selection
    if "databook_selections" not in st.session_state:
        st.session_state.databook_selections = set()

    if "active_explorer_items" not in st.session_state:
        st.session_state.active_explorer_items = []

    # Initialize last selector type to detect tab switches
    if "last_selector_type" not in st.session_state:
        st.session_state.last_selector_type = "Object"

    if not inventory:
        st.warning("⚠️ Could not load Inventory (GitHub or DB). check connection.")
    
    c_left, c_workspace = st.columns([1, 3])

    with c_left:
        with st.container(border=True):
            st.subheader("Selector")
            
            # 1. Mode Selection
            selector_type = st.radio(
                "Analysis Mode", 
                ["Object", "Verb", "Lexical"], 
                captions=["Node-Centric", "Relationship-Centric", "Text-Mentions"],
                horizontal=True
            )
            
            # Auto-Clean on Tab Switch
            if selector_type != st.session_state.last_selector_type:
                st.session_state.databook_selections = set()
                st.session_state.active_explorer_items = []
                st.session_state.last_selector_type = selector_type
                st.rerun()

            st.divider()

            # 2. Controls: Visualize & Clear (Only show if NOT Lexical)
            if selector_type != "Lexical":
                selection_count = len(st.session_state.databook_selections)
                
                c_vis, c_clear = st.columns([2, 1])
                with c_vis:
                    if st.button(f"Visualize ({selection_count})", type="primary", use_container_width=True):
                        st.session_state.active_explorer_items = [
                            {'label': l, 'name': n} for l, n in st.session_state.databook_selections
                        ]
                with c_clear:
                    if st.button("Clear", use_container_width=True):
                        st.session_state.databook_selections = set()
                        st.session_state.active_explorer_items = []
                        st.rerun()

                st.divider()

            # 3. Scrollable List Container
            with st.container(height=600, border=False):
                # --- LOGIC FOR LEXICAL (Placeholder) ---
                if selector_type == "Lexical":
                     st.info("Lexical Analysis (Text-Mentions) will be added in a future update.")
                     # No further logic executes for this branch

                # --- LOGIC FOR OBJECT & VERB ---
                else:
                    available_data = inventory.get(selector_type, {})
                    
                    if not available_data:
                        st.caption(f"No inventory for {selector_type}.")
                    else:
                        if isinstance(available_data, dict):
                            # --- OBJECT MODE ---
                            if selector_type == "Object":
                                labels = sorted(list(available_data.keys()))
                                for label in labels:
                                    search_key = f"search_{selector_type}_{label}"
                                    is_expanded = bool(st.session_state.get(search_key, ""))

                                    with st.expander(f"{label}", expanded=is_expanded):
                                        # Clean data
                                        raw_vals = available_data[label]
                                        clean_names = []
                                        if isinstance(raw_vals, dict):
                                            clean_names = [v for v in raw_vals.values() if v and pd.notna(v)]
                                        elif isinstance(raw_vals, list):
                                            clean_names = [v for v in raw_vals if v and pd.notna(v)]
                                        names = sorted(list(set(str(n) for n in clean_names)))
                                        
                                        if names:
                                            search_term = st.text_input(f"Search {label}", placeholder="Filter...", key=search_key)
                                            filtered_names = [n for n in names if search_term.lower() in n.lower()] if search_term else names
                                            
                                            if not filtered_names:
                                                st.caption("No matches.")
                                            else:
                                                # Truncate large lists
                                                display_names = filtered_names[:50] if (len(filtered_names) > 50 and not search_term) else filtered_names
                                                if len(filtered_names) > 50 and not search_term:
                                                    st.info(f"Showing 50 of {len(filtered_names)}.")

                                                for name in display_names:
                                                    is_selected = (label, name) in st.session_state.databook_selections
                                                    chk_key = f"chk_{selector_type}_{label}_{name}"
                                                    
                                                    def update_selection(l=label, n=name, k=chk_key):
                                                        if st.session_state[k]:
                                                            st.session_state.databook_selections.add((l, n))
                                                        else:
                                                            st.session_state.databook_selections.discard((l, n))

                                                    st.checkbox(name, value=is_selected, key=chk_key, on_change=update_selection)
                                        else:
                                            st.caption("No names.")

                            # --- VERB MODE ---
                            elif selector_type == "Verb":
                                rel_types = sorted(list(available_data.keys()))
                                if rel_types:
                                    search_key = f"search_{selector_type}"
                                    search_term = st.text_input("Search Relationships", placeholder="Filter...", key=search_key)
                                    filtered_rels = [r for r in rel_types if search_term.lower() in r.lower()] if search_term else rel_types
                                    
                                    for r_type in filtered_rels:
                                        # Use "Verb" as the label for backend logic
                                        is_selected = ("Verb", r_type) in st.session_state.databook_selections
                                        chk_key = f"chk_verb_{r_type}"
                                        
                                        def update_verb_selection(t=r_type, k=chk_key):
                                            if st.session_state[k]:
                                                st.session_state.databook_selections.add(("Verb", t))
                                            else:
                                                st.session_state.databook_selections.discard(("Verb", t))
                                        
                                        st.checkbox(r_type, value=is_selected, key=chk_key, on_change=update_verb_selection)
                                else:
                                    st.caption("No relationship types found.")
                        else:
                            st.error("Invalid inventory format.")

    with c_workspace:
        # Don't try to render workspace for Lexical placeholder
        if selector_type != "Lexical":
            render_explorer_workspace(
                selector_type, 
                st.session_state.active_explorer_items
            )
        
# ==========================================
# 0. NON UPDATED PARTS
# ==========================================

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Graph Analyst", layout="wide", page_icon="🕸️")

# --- CONSTANTS & CONFIG ---
LLM_MODEL = "mistral-medium"
SAFETY_REGEX = re.compile(r"(?i)\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|INSERT|ALTER|GRANT|REVOKE)\b")

# --- OPTIMIZATION: CACHED RESOURCES ---
# These functions prevent the app from reconnecting to DB/API on every rerun.

@st.cache_resource
def get_cached_driver(uri, auth):
    """Maintains a persistent Neo4j connection pool."""
    cleaned_uri = uri.strip()
    if not any(cleaned_uri.startswith(p) for p in ["neo4j+s://", "bolt://", "neo4j://"]):
        cleaned_uri = f"neo4j+s://{cleaned_uri}"
    driver = GraphDatabase.driver(cleaned_uri, auth=auth)
    driver.verify_connectivity()
    return driver

@st.cache_resource
def get_cached_llm(api_key):
    """Maintains a persistent LLM client."""
    return ChatMistralAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.0
    )

@st.cache_data
def load_github_data():
    """Loads external context data efficiently."""
    url = "https://raw.githubusercontent.com/ssopic/some_data/main/sum_data.csv"
    try:
        df = pd.read_csv(url)
        # Robust cleaning: convert to string, remove potential '.0' from floats, and strip whitespace
        df['PK'] = df['PK'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return df
    except Exception as e:
        return pd.DataFrame(columns=['PK', 'details'])

# --- SCHEMA DEFINITIONS (Preserved from Old Version) ---

class StructuredBlueprint(BaseModel):
    intent: str = Field(description="The primary action: FindEntity, FindPath, MultiHopAnalysis, etc.")
    target_entity_nl: str = Field(description="Natural language term for the primary entity.")
    source_entity_nl: str = Field(description="Natural language term for the starting entity or context.")
    complexity: str = Field(description="Simple, MultiHop, or Aggregation.")
    proposed_relationships: List[str] = Field(default_factory=list, description="List of sequential relationships or verbs.")
    filter_on_verbs: List[str] = Field(default_factory=list, description="Specific raw verbs to filter by.")
    constraints: List[str] = Field(description="Temporal or attribute constraints.")
    properties_to_return: List[str] = Field(description="Properties the user wants to see.")

class GroundingOutput(BaseModel):
    thought_process: str = Field(description="Step-by-step reasoning.")
    source_node_label: str = Field(description="Primary source node label.")
    target_node_label: str = Field(description="Primary target node label.")
    relationship_paths: List[str] = Field(description="Ordered list of specific relationship types.")
    target_node_variable: str = Field(description="Cypher var for target.")
    source_node_variable: str = Field(description="Cypher var for source.")
    cypher_constraint_fragment: str = Field(description="WHERE clause fragment.")
    cypher_return_fragment: str = Field(description="RETURN statement fragment.")

class GroundedComponent(BaseModel):
    source_node_label: str
    target_node_label: str
    relationship_paths: List[str]
    cypher_constraint_clause: str
    return_variables: str
    target_node_variable: str
    source_node_variable: str
    filter_on_verbs: List[str] = Field(default_factory=list)

class CorrectionReport(BaseModel):
    error_source_agent: str
    correction_needed: str
    fixed_cypher: Optional[str] = Field(None)
    new_blueprint_fragment: Optional[Dict[str, Any]] = Field(None)

class PrunedSchema(BaseModel):
    NodeLabels: List[str]
    RelationshipTypes: List[str]
    NodeProperties: Dict[str, List[str]]
    RelationshipProperties: Dict[str, List[str]]

class SynthesisOutput(BaseModel):
    final_answer: str
    source_document_ids: List[str] = Field(default_factory=list)

class CypherWrapper(BaseModel):
    query: str

# --- SYSTEM PROMPTS (Preserved) ---

SYSTEM_PROMPTS = {
    "Intent Planner": (
        "You are a Cypher query planning expert. Analyze the user's natural language query. "
        "The user will almost always input their question in English but if the question is in another language, you must translate the users question to English prior to continuing with the following steps:"
        "Determine if this is a simple lookup or a 'MultiHop' query.\n"
        "RELATIONSHIPS: Extract the sequence of actions or verbs into 'proposed_relationships' as a list. "
        "Example: 'Who paid Epstein and visited?' -> ['paid', 'visited'].\n"
        "VERB FILTERS: If the user specifically asks for exact phrases (e.g., 'relationships with the verb \"stocks of\"'), "
        "extract those exact strings into 'filter_on_verbs'."),
    "Schema Selector": (
        "You are a Schema Context manager. Your goal is to select the relevant graph schema elements for the user's query.\n"
        "INPUTS:\n"
        "1. BLUEPRINT: The user's intent and proposed relationships.\n"
        "2. FULL_SCHEMA: The actual Node Labels, Relationship Types, AND their Properties.\n"
        "TASK:\n"
        "Select ONLY the Node Labels and Relationship Types relevant to the blueprint. "
        "Crucially, populate 'NodeProperties' and 'RelationshipProperties' with the valid keys for the selected types.\n"
        "CRITICAL RULES:\n"
        "1. MAPPING: You MUST map the user's natural language verbs (e.g. 'paid') to the closest semantic equivalent in the FULL_SCHEMA (e.g. 'FINANCIAL_TRANSACTION'). Do NOT return empty lists if a plausible match exists.\n"
        "2. EXACT RETURN: Once you identify the match, return the string EXACTLY as it appears in FULL_SCHEMA.\n"
        "3. PROPERTY AWARENESS: Pass the valid property lists. This prevents the generator from inventing properties like '.type' or '.status' if they don't exist."
    ),
    "Grounding Agent": (
        "You are a Graph Grounding expert. Map the blueprint to the specific Schema provided.\n"
        "CHAIN OF THOUGHT (Required):\n"
        "1. Analyze Entities: Is 'Island' a PERSON or a LOCATION? Check the Schema labels.\n"
        "2. If you are sure the Node is a PERSON feel free to use the .name property. If the node has any other label(ie. ORGANIZATION, ISLAND ), do not use any filters on it. \n"
        "3. Define Path: If the destination is a Location, ensure both the right relationship type is being used and that there are no labels in the pattern (e.g., `(p)-[:MOVED]->(l:)`).\n\n"
        "TASK: Create a blueprint where 'relationship_paths' uses the EXACT relationship types from the SCHEMA.\n"
        "MULTI-HOP: The 'proposed_relationships' list (e.g., ['paid', 'visited']) must be mapped to the valid schema types provided in the SCHEMA list.\n"
        "PROVENANCE FOR RELATIONSHIPS: Return provenance from the RELATIONSHIPS. Use `coalesce(r.source_pks)` to make sure the user can properly analyze the results. .\n"
        "PROVENANCE FOR Documents: If the relationship is 'MENTIONED_IN'. Use `coalesce(d.doc_id)` for nodes labeled as 'document'.\n"        
        "CONSTRAINT RULE: Do NOT use properties in the WHERE clause that are not listed in the Schema's NodeProperties."
    ),
    "Cypher Generator": (
        "You are an expert Cypher Generator. Convert the Grounded Component into a VALID, READ-ONLY Cypher query. "
        "RULES:\n"
        "1. PATHS: Iterate through the 'relationship_paths' list to build the pattern. Assign variables to ALL relationships.\n"
        "   Example 2 steps: (a:LABEL)-[r1:REL_TYPE_1]->(b)-[r2:REL_TYPE_2]->(c:LABEL).\n"
        "   CRITICAL 1: Do NOT create self-loops like `(b)--(b)`. Ensure the path is continuous: `(a)-[r1]->(b)-[r2]->(c)`.\n"
        "   CRITICAL 2: All labels are CAPITALIZED.\n"
        "2. PROPERTIES: Only use properties explicitly listed in the Schema. Do NOT invent properties like `.type`, `.category`, etc.\n"
        "   **CRITICAL EXCEPTION**: For (n:Person), ONLY use `n.name`. NEVER use `n.id` or `n.entity_id`.\n"
        "3. FUZZY MATCHING: For names/strings, always use `toLower(n.name) CONTAINS 'Johnny'` over strict equality `=` to handle messy data.\n"
        "4. VERB FILTERS: If 'filter_on_verbs' is provided, add a WHERE clause to check `raw_verbs` on the relationships.\n"
        "   Example: `WHERE ANY(v IN r1.raw_verbs WHERE v CONTAINS 'stocks of')`.\n"
        "5. PROVENANCE: Return provenance from the relationship variables using `coalesce(r.source_pks, r.doc_id)`. "
        "Do NOT query 'target_pks'. If multi-hop, return a list (e.g. `[coalesce(r1.source_pks, r1.doc_id), ...]`).\n"
        "6. DISTINCT: Always use `RETURN DISTINCT` to avoid duplicate result rows.\n"
        "7. Do not include semicolons at the end."
    ),
    "Query Debugger": (
        "You are an expert Neo4j Debugger. Analyze the error, warnings, and the failed query. "
        "1. If the error mentions missing properties (e.g. 'properties does not exist'), change `r.properties` to `properties(r)`.\n"
        "2. If fixing a path syntax error, ensure you do not create self-loops like `(n)--(n)`. This returns no data. Ensure nodes are distinct.\n"
        "3. SCHEMA CHECK: You MUST NOT invent new relationship types or properties. You must check the `schema` provided in the context and only use elements listed there.\n"
        "Provide a `fixed_cypher` string with the correction."
    ),
    "Synthesizer": (
        "You are a conversational AI. Explain the cypher query, explaining it compared to the users question mentioning which data might be captured with it." 
        "Make sure the user knows which other details might be caught that might have been unintended, focusing on the fact that we use only relationships and do not filter nodes other than Person nodes" 
        "Make sure to mention that this is a wide search and furhter questions can be asked at the analysis section which currently contains only the emails from the oversight comittee released in the November of 2025. After that summarize what you see in the database preview specifically mentioning that you do not see all of the documents. "
        "If the result is empty, state clearly that no information was found."
    )
}

# --- UTILITIES ---

def fetch_schema_statistics(uri: str, auth: tuple) -> Dict[str, Any]:
    """
    Connects to DB and creates comprehensive schema stats.
    Optimization: Uses a one-off connection here, but logic is robust.
    """
    stats = {
        "status": "INIT", "NodeCounts": {}, "RelationshipVerbs": {},
        "NodeLabels": [], "RelationshipTypes": [], 
        "NodeProperties": {}, "RelationshipProperties": {}
    }
    driver = None
    try:
        # We don't use the cached driver here to allow for connection testing with new creds
        cleaned_uri = uri.strip()
        if not any(cleaned_uri.startswith(p) for p in ["neo4j+s://", "bolt://", "neo4j://"]):
            cleaned_uri = f"neo4j+s://{cleaned_uri}"
            
        driver = GraphDatabase.driver(cleaned_uri, auth=auth)
        driver.verify_connectivity()

        with driver.session() as session:
            stats["NodeLabels"] = session.run("CALL db.labels()").value()
            stats["RelationshipTypes"] = session.run("CALL db.relationshipTypes()").value()

            for label in stats["NodeLabels"]:
                count = session.run(f"MATCH (n:`{label}`) RETURN count(n) as c").single()["c"]
                stats["NodeCounts"][label] = count

            for r_type in stats["RelationshipTypes"]:
                try:
                    verb_q = f"MATCH ()-[r:`{r_type}`]->() WHERE r.raw_verbs IS NOT NULL UNWIND r.raw_verbs as v RETURN DISTINCT v LIMIT 10"
                    verbs = [record["v"] for record in session.run(verb_q)]
                    stats["RelationshipVerbs"][r_type] = verbs
                except: stats["RelationshipVerbs"][r_type] = []

            # Optimized Property Fetching
            node_props_q = "CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName RETURN nodeType, collect(propertyName) as props"
            for record in session.run(node_props_q):
                raw_type = record["nodeType"]
                if raw_type.startswith(":"):
                    for label in raw_type[1:].split(":"):
                        current_props = set(stats["NodeProperties"].get(label, []))
                        current_props.update(record["props"])
                        stats["NodeProperties"][label] = list(current_props)

            rel_props_q = "CALL db.schema.relTypeProperties() YIELD relType, propertyName RETURN relType, collect(propertyName) as props"
            for record in session.run(rel_props_q):
                raw_type = record["relType"]
                if raw_type.startswith(":"):
                    stats["RelationshipProperties"][raw_type[1:]] = record["props"]

            stats["status"] = "SUCCESS"
    except Exception as e:
        stats["status"] = "ERROR"
        stats["error"] = str(e)
    finally:
        if driver: driver.close()
    return stats

def extract_provenance_from_result(result: Union[Dict, List]) -> List[str]:
    """Scans JSON recursively for keys indicating ID storage (provenance, pks, doc_id)."""
    ids = []
    # Broader set of keywords to catch un-aliased returns like 'r.source_pks'
    target_keys = ["provenance", "source_pks", "doc_id", "id_list", "pk"]
    
    def recurse(data):
        if isinstance(data, dict):
            for k, v in data.items():
                # Check if ANY target key matches the result key (case-insensitive)
                if any(t in k.lower() for t in target_keys):
                    if isinstance(v, list): ids.extend([str(i) for i in v])
                    else: ids.append(str(v))
                else: 
                    recurse(v)
        elif isinstance(data, list):
            for item in data: recurse(item)
            
    recurse(result)
    return list(set(ids))

# --- PIPELINE CLASS (Optimized) ---

class GraphRAGPipeline:
    def __init__(self, driver: Driver, llm: ChatMistralAI, schema_stats: Dict[str, Any]):
        self.driver = driver
        self.llm = llm
        self.schema_stats = schema_stats or {}

    def _get_full_schema(self) -> Dict[str, Any]:
        return {
            "NodeLabels": self.schema_stats.get("NodeLabels", []),
            "RelationshipTypes": self.schema_stats.get("RelationshipTypes", []),
            "NodeProperties": self.schema_stats.get("NodeProperties", {}),
            "RelationshipProperties": self.schema_stats.get("RelationshipProperties", {})
        }

    def _check_query_safety(self, cypher: str) -> bool:
        return not bool(SAFETY_REGEX.search(cypher))

    @traceable 
    def _run_agent(self, agent_name: str, output_model: BaseModel, context: Dict) -> Any:
        
        # ---  Dynamic Renaming Logic ---
        rt = get_current_run_tree()
        if rt:
            rt.name = agent_name
        # ---------------------------------

        sys_prompt = SYSTEM_PROMPTS.get(agent_name, "")
        
        formatted_context = {}
        for k, v in context.items():
            if isinstance(v, BaseModel): 
                formatted_context[k] = v.model_dump_json(indent=2)
            elif isinstance(v, (dict, list)): 
                formatted_context[k] = json.dumps(v, indent=2)
            else: 
                formatted_context[k] = str(v)

        msgs = [("system", sys_prompt)]
        
        # Prompt Mapping
        if agent_name == "Intent Planner": 
            msgs.append(("human", "QUERY: {user_query}"))
        elif agent_name == "Schema Selector": 
            msgs.append(("human", "BLUEPRINT: {blueprint}\nFULL_SCHEMA: {full_schema}"))
        elif agent_name == "Grounding Agent": 
            msgs.append(("human", "BLUEPRINT: {blueprint}\nSCHEMA: {schema}\nQUERY: {user_query}"))
        elif agent_name == "Cypher Generator": 
            msgs.append(("human", "GROUNDED_COMPONENT: {blueprint}"))
        elif agent_name == "Query Debugger": 
            msgs.append(("human", "ERROR: {error}\nQUERY: {failed_query}\nBLUEPRINT: {blueprint}\nWARNINGS: {warnings}"))
        elif agent_name == "Synthesizer": 
            msgs.append(("human", "QUERY: {user_query}\nCYPHER: {final_cypher}\nRESULTS: {db_result}"))
        
        prompt = ChatPromptTemplate.from_messages(msgs)
        return (prompt | self.llm.with_structured_output(output_model)).invoke(formatted_context)
        
    @traceable(name="GraphRAG Main Pipeline")
    def run(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        rt = get_current_run_tree()
        if rt:
            rt.add_metadata({"session_id": session_id})
        pipeline_object = {"user_query": user_query, "status": "INIT", "execution_history": [], "proof_ids": []}
        
        try:
            # 1. Intent
            pipeline_object["status"] = "PLANNING"
            blueprint = self._run_agent("Intent Planner", StructuredBlueprint, {"user_query": user_query})

            pipeline_object["intent_data"] = blueprint.model_dump() # Save for debugging
            
            # 2. Schema
            full_schema = self._get_full_schema()
            pruned_schema = self._run_agent("Schema Selector", PrunedSchema, {"blueprint": blueprint, "full_schema": full_schema})
            
            # 3. Grounding
            grounding = self._run_agent("Grounding Agent", GroundingOutput, {"blueprint": blueprint, "schema": pruned_schema.model_dump(), "user_query": user_query})
            pipeline_object["grounding_data"] = grounding.model_dump() # Save for debugging
            
            grounded_comp = GroundedComponent(
                source_node_label=grounding.source_node_label,
                target_node_label=grounding.target_node_label,
                relationship_paths=grounding.relationship_paths,
                cypher_constraint_clause=grounding.cypher_constraint_fragment,
                return_variables=grounding.cypher_return_fragment,
                target_node_variable=grounding.target_node_variable,
                source_node_variable=grounding.source_node_variable,
                filter_on_verbs=blueprint.filter_on_verbs
            )

            # 4. Generation
            cypher_resp = self._run_agent("Cypher Generator", CypherWrapper, {"blueprint": grounded_comp})
            raw_cypher = cypher_resp.query.replace("\n", " ")
            if not self._check_query_safety(raw_cypher): raise ValueError("Unsafe Cypher detected.")
            pipeline_object["cypher_query"] = raw_cypher

            # 5. Execution & Retry Loop
            pipeline_object["status"] = "EXECUTING"
            results = []
            
            for attempt in range(1, 4):
                attempt_log = {"attempt": attempt, "cypher": raw_cypher, "status": "PENDING", "warnings": []}
                try:
                    with self.driver.session() as session:
                        res_obj = session.run(raw_cypher)
                        records = [r.data() for r in res_obj]
                        
                        # Capture notifications/warnings if available
                        summary = res_obj.consume()
                        if summary.notifications:
                             # FIX: Robust warning handling (Dict vs Object)
                             warnings = []
                             for n in summary.notifications:
                                 # Handle Neo4j driver returning dicts instead of objects
                                 if isinstance(n, dict):
                                     code = n.get("code", "UNKNOWN")
                                     msg = n.get("description", "No description")
                                 else:
                                     code = getattr(n, "code", "UNKNOWN")
                                     msg = getattr(n, "description", "No description")
                                 warnings.append({"code": code, "message": msg})
                             attempt_log["warnings"] = warnings
                        
                        results = records
                        attempt_log["status"] = "SUCCESS"
                        pipeline_object["execution_history"].append(attempt_log)
                        pipeline_object["cypher_query"] = raw_cypher # Update in case of fix
                        break
                except Exception as e:
                    attempt_log["status"] = "FAILED"
                    attempt_log["error"] = str(e)
                    pipeline_object["execution_history"].append(attempt_log)
                    
                    if attempt < 3:
                        correction = self._run_agent("Query Debugger", CorrectionReport, {
                            "error": str(e), "failed_query": raw_cypher, 
                            "blueprint": grounded_comp, "schema": full_schema, 
                            "warnings": attempt_log["warnings"]
                        })
                        if correction.fixed_cypher: raw_cypher = correction.fixed_cypher.replace("\n", " ")
                        else: break
                    else: raise e

            # 6. Synthesis
            pipeline_object["raw_results"] = results
            pipeline_object["proof_ids"] = extract_provenance_from_result(results) # Internal utility
            
            if not results:
                pipeline_object["final_answer"] = "No information found."
            else:
                synth = self._run_agent("Synthesizer", SynthesisOutput, {
                    "user_query": user_query, "final_cypher": raw_cypher, "db_result": results[:50]
                })
                pipeline_object["final_answer"] = synth.final_answer
            
            pipeline_object["status"] = "SUCCESS"

        except Exception as e:
            pipeline_object["status"] = "ERROR"
            pipeline_object["error"] = str(e)
            
        return pipeline_object

# --- INITIALIZATION ---

if 'github_data' not in st.session_state:
    st.session_state.github_data = load_github_data()

if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "connected": False, "mistral_key": "", "neo4j_creds": {}, 
        "schema_stats": {}, "evidence_locker": [], "selected_ids": set(), "chat_history": []
    }

# --- SCREENS ---

# ==========================================
# AUTHENTICATION & SETTINGS LOGIC
# ==========================================

def get_config(key, default=""):
    """Helper to get credentials from Secrets (Cloud) or Env (Local)."""
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

def attempt_connection(uri, username, password, api_key):
    """
    Attempts to connect to Neo4j and validate the Mistral Key.
    Returns (Success: bool, Message: str)
    """
    try:
        # Validate Neo4j connection using your existing function
        stats = fetch_schema_statistics(uri, (username, password))
        
        if stats["status"] == "SUCCESS":
            # Update Session State on success
            st.session_state.app_state.update({
                "connected": True,
                "mistral_key": api_key,
                "neo4j_creds": {
                    "uri": uri, 
                    "user": username, 
                    "pass": password, 
                    "auth": (username, password)
                },
                "schema_stats": stats
            })
            return True, "✅ Successfully connected to Neo4j & Mistral!"
        else:
            return False, f"Neo4j Error: {stats.get('error')}"

    except Exception as e:
        return False, f"Connection Failed: {str(e)}"

@st.dialog("⚙️ Settings")
def show_settings_dialog():
    """
    Pop-up modal for manually configuring credentials.
    Masks secrets so they are not exposed in the UI source.
    """
    st.write("Configure your database and API connections.")
    
    # Load current values from session or environment defaults
    current_creds = st.session_state.app_state.get("neo4j_creds", {})
    
    # Get existing values (from session or secrets) to determine STATUS (not value)
    existing_mistral = st.session_state.app_state.get("mistral_key") or get_config("MISTRAL_API_KEY")
    existing_uri = current_creds.get("uri") or get_config("NEO4J_URI")
    existing_user = current_creds.get("user") or get_config("NEO4J_USER", "neo4j")
    existing_pass = current_creds.get("pass") or get_config("NEO4J_PASSWORD")

    # --- SECRETS MASKING LOGIC ---
    # We do NOT put the actual key in 'value'. We only show a placeholder if it exists.
    
    mistral_placeholder = "******** (Stored)" if existing_mistral else "Enter Mistral API Key"
    uri_placeholder = "******** (Stored)" if existing_uri else "Enter Neo4j URI"
    user_placeholder = "******** (Stored)" if existing_user else "Enter Neo4j User"
    pass_placeholder = "******** (Stored)" if existing_pass else "Enter Neo4j Password"
    
    st.caption("Leave fields blank to keep the currently stored values.")
    
    # Inputs start empty to protect secrets
    m_key_input = st.text_input("Mistral API Key", value="", type="password", placeholder=mistral_placeholder)
    n_uri_input = st.text_input("Neo4j URI", value="", placeholder=uri_placeholder)
    n_user_input = st.text_input("Neo4j User", value="", placeholder=user_placeholder)
    n_pass_input = st.text_input("Neo4j Password", value="", type="password", placeholder=pass_placeholder)
    
    if st.button("Save & Reconnect", type="primary"):
        # LOGIC: Use new input if provided, otherwise fall back to existing value
        
        # Resolve inputs (ternary operator handles 'None' or empty string)
        final_mistral = m_key_input if m_key_input else existing_mistral
        final_uri = n_uri_input if n_uri_input else existing_uri
        final_user = n_user_input if n_user_input else existing_user
        final_pass = n_pass_input if n_pass_input else existing_pass

        with st.spinner("Testing connection..."):
            success, msg = attempt_connection(final_uri, final_user, final_pass, final_mistral)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

#old version. probabbly going to delete
# def init_app():
#     """
#     Runs once on app startup to try auto-login using Secrets/Env Vars.
#     """
#     if st.session_state.get("has_tried_login", False):
#         return

#     # 1. NEW: Setup LangSmith 
#     # This MUST happen here so the library finds the key when the pipeline runs later
#     ls_key = get_config("LANGCHAIN_API_KEY") 
#     if ls_key:
#         os.environ["LANGCHAIN_API_KEY"] = ls_key
#         os.environ["LANGCHAIN_TRACING_V2"] = "true"
#         os.environ["LANGCHAIN_PROJECT"] = "Testing_analysis_tool"
#         os.environ["LANGCHAIN_ENDPOINT"] = "[https://eu.api.smith.langchain.com](https://eu.api.smith.langchain.com)"
#     if "app_session_id" not in st.session_state:
#         st.session_state["app_session_id"] = str(uuid.uuid4())

#     # 2. Get DATABASE and mistral keys
#     m_key = get_config("MISTRAL_API_KEY")
#     n_uri = get_config("NEO4J_URI")
#     n_user = get_config("NEO4J_USER", "neo4j")
#     n_pass = get_config("NEO4J_PASSWORD")

#     # Only attempt if we actually have credentials
#     if n_uri and n_pass and m_key:
#         success, msg = attempt_connection(n_uri, n_user, n_pass, m_key)
#         if not success:
#             # Pop-up toast notification of failure (non-intrusive)
#             st.toast(f"⚠️ Auto-login failed: {msg}", icon="⚠️")
    
#     st.session_state.has_tried_login = True
# FRAGMENT: Updates only the chat area when interacting
@st.fragment
def screen_extraction():
    st.title("🔍 Extraction & Cypher Sandbox")
    
    # 1. Define Tabs
    tab_chat, tab_cypher = st.tabs(["💬 Agent Chat", "🛠️ Raw Cypher"])
    
    # --- TAB 1: EXISTING AGENT CHAT (Preserved) ---
    with tab_chat:
        # Check Connections using credentials from app_state
        creds = st.session_state.app_state["neo4j_creds"]
        driver = get_cached_driver(creds["uri"], creds["auth"])
        llm = get_cached_llm(st.session_state.app_state["mistral_key"])
        
        if not driver or not llm:
            st.warning("System unavailable. Please check secrets.")
            return
            
        # Pipeline: Pass schema_stats from app_state
        pipeline = GraphRAGPipeline(driver, llm, st.session_state.app_state["schema_stats"])

        # Chat UI: Iterate over chat_history in app_state
        for chat in st.session_state.app_state["chat_history"]:
            with st.chat_message(chat["role"]): st.write(chat["content"])

        user_msg = st.chat_input("Ask about the graph...")
        if user_msg:
            # Update app_state chat history
            st.session_state.app_state["chat_history"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"): st.write(user_msg)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing public dataset..."):
                    result = pipeline.run(user_msg, session_id=st.session_state["app_session_id"])
                    
                    if result.get("status") == "ERROR":
                        err_msg = result.get("error", "Unknown Error")
                        st.error(f"Pipeline Error: {err_msg}")
                        with st.expander("Technical Details"):
                            st.write(result)
                    else:
                        ans = result.get("final_answer", "No answer generated.")
                        st.write(ans)
                        
                        if result.get("cypher_query"):
                            with st.expander("🕵️ Analysis Details (Cypher & Trace)"):
                                if result.get("cypher_query"):
                                    st.caption("Generated Cypher:")
                                    st.code(result["cypher_query"], language="cypher")
                                
                                st.caption("Full Pipeline Trace Data:")
                                st.json(result) # This shows the Intent/Grounding data we saved!

                        st.session_state.app_state["chat_history"].append({"role": "assistant", "content": ans})
                        
                        if result.get("proof_ids"):
                            st.session_state.app_state["evidence_locker"].append({
                                "query": user_msg, "answer": ans, "ids": result["proof_ids"]
                            })
                            st.toast("Evidence saved to locker")

    # --- TAB 2: RAW CYPHER INPUT (Updated with Save Logic) ---
    with tab_cypher:
        st.markdown("### Safe Cypher Execution")
        st.caption("Read-Only mode active. Modifications (CREATE, SET, DELETE) are blocked.")
        
        # State container for manual query results
        if "manual_results" not in st.session_state:
            st.session_state.manual_results = None
        if "manual_query_text" not in st.session_state:
            st.session_state.manual_query_text = ""

        cypher_input = st.text_area("Enter Cypher Query", height=150, value="MATCH (n) RETURN n LIMIT 5")
        
        if st.button("Run Query"):
            if SAFETY_REGEX.search(cypher_input):
                st.error("🚨 SECURITY ALERT: destructive commands are not allowed.")
            else:
                creds = st.session_state.app_state["neo4j_creds"]
                driver = get_cached_driver(creds["uri"], creds["auth"])
                
                if driver:
                    try:
                        with driver.session() as session:
                            res = session.run(cypher_input)
                            data = [r.data() for r in res]
                            
                            # Save to temporary state for the 'Save' button to access
                            st.session_state.manual_results = data
                            st.session_state.manual_query_text = cypher_input
                            
                            if not data:
                                st.warning("Query returned no results.")
                    except Exception as e:
                        st.error(f"Cypher Syntax Error: {e}")

        # Display Results & Save Button (Persistent across reruns in fragment)
        if st.session_state.manual_results:
            st.divider()
            st.subheader("Results")
            st.dataframe(pd.DataFrame(st.session_state.manual_results), use_container_width=True)
            
            # Helper to count IDs found
            found_ids = extract_provenance_from_result(st.session_state.manual_results)
            st.caption(f"Found {len(found_ids)} potential document IDs.")
            
            if st.button("💾 Add Results to Locker"):
                if found_ids:
                    payload = {
                        "query": f"Manual Cypher: {st.session_state.manual_query_text[:30]}...",
                        "answer": "Manually executed Cypher query results.",
                        "ids": found_ids
                    }
                    st.session_state.app_state["evidence_locker"].append(payload)
                    st.toast(f"Saved {len(found_ids)} IDs to Locker!")
                else:
                    st.warning("No IDs (provenance/source_pks) found in these results to save.")

@st.fragment
def screen_locker():
    st.title("🗄️ Evidence Locker")
    locker = st.session_state.app_state["evidence_locker"]
    
    if not locker:
        st.info("Locker is empty.")
        return

    # 1. Initialize local selection set
    current_selection = set()
    
    # 2. Retrieve previously selected IDs to restore checkbox state
    global_selected = st.session_state.app_state.get("selected_ids", set())

    for i, entry in enumerate(locker):
        with st.container(border=True):
            c1, c2 = st.columns([0.1, 0.9])
            with c1:
                # 3. Logic: If the entry's IDs are already in the global set, check the box.
                # We convert entry IDs to a set of strings to compare.
                entry_ids_str = {str(pid) for pid in entry["ids"]}
                
                # Check if this batch is already selected (subset of global selection)
                is_checked_default = entry_ids_str.issubset(global_selected) if entry_ids_str else False

                # 4. Render Checkbox with `value=` set to restored state
                is_sel = st.checkbox("Select", key=f"sel_{i}", value=is_checked_default)
                
                if is_sel: 
                    # If checked (either by user or restored state), add to current set
                    for pid in entry["ids"]: 
                        current_selection.add(str(pid))
            with c2:
                st.write(f"**Query:** {entry['query']}")
                st.caption(f"Found IDs: {', '.join(entry['ids'])}")

    # 5. Commit to Global State
    st.session_state.app_state["selected_ids"] = current_selection

@st.fragment
def screen_analysis():
    st.title("🔬 Analysis Pane")
    ids = list(st.session_state.app_state["selected_ids"])
    
    if not ids:
        st.warning("No documents selected.")
        return
        
    df = st.session_state.github_data
    matched = df[df['PK'].isin(ids)]
    
    st.subheader(f"Analyzing {len(matched)} Documents")
    if not matched.empty:
        context = ""
        for _, row in matched.iterrows():
            context += f"ID: {row['PK']}\nContent: {row.get('email_content', 'No Content')}\n---\n"
            
        with st.expander("Raw Content"):
            st.text_area("Context", context, height=200)
            
        q = st.chat_input("Ask about this evidence:")
        if q:
            llm = get_cached_llm(st.session_state.app_state["mistral_key"])
            resp = llm.invoke(f"Context:\n{context}\n\nQuestion: {q}")
            st.info(resp.content)

# --- CSS ---
# 1. CSS Injection
def inject_custom_css():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        header { visibility: hidden; }
        [data-testid="column"]:nth-of-type(1), [data-testid="column"]:nth-of-type(3) {
            background-color: #0f1116; 
            border: 1px solid #1f2937; 
            padding: 1rem; 
            border-radius: 8px;
        }
        .stButton button { width: 100%; height: 3rem; font-weight: bold; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
# --- MAIN NAVIGATION ---

# 1. Try to initialize (Auto-connect on startup)
init_app()
inject_custom_css()

# 2. Sidebar Navigation
with st.sidebar:
    st.header("Graph Analyst")
    
    # Navigation Options
    nav_options = ["Databook", "Search", "Locker", "Analysis"]
    nav = st.radio("Navigation", nav_options)
    
    st.divider()
    
    # NEW: Settings Button (Opens the pop-up dialog)
    if st.button("⚙️ Settings"):
        show_settings_dialog()
    
    # NEW: Connection Status & Logout
    if st.session_state.app_state["connected"]:
        st.caption("🟢 Connected")
        if st.button("Logout"):
            st.session_state.app_state["connected"] = False
            st.session_state.has_tried_login = False # Reset so it doesn't auto-login immediately
            st.rerun()
    else:
        st.caption("🔴 Disconnected")

# 3. Main Content Router
if not st.session_state.app_state["connected"]:
    # Landing message instead of the old 'screen_connection()'
    st.info("👋 Welcome! The app is disconnected.\n\nIf you set up your Secrets correctly, this should not appear.\n\nOtherwise, click **⚙️ Settings** in the sidebar to connect manually.")
else:
    # Router to your screens
    if nav == "Databook": 
        screen_databook()
    elif nav == "Search": 
        screen_extraction()
    elif nav == "Locker": 
        screen_locker()
    elif nav == "Analysis": 
        screen_analysis()
