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

def setup_langsmith():
    """
    Sets up LangSmith environment variables.
    NOTE: For decorators like @traceable to work reliably, these variables 
    should ideally be set at the very top of your script, before other imports/definitions.
    """
    if "LANGSMITH_API_KEY" in st.secrets:
        # 1. Set the API Key
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
        
        # 2. Set the Project Name
        os.environ["LANGCHAIN_PROJECT"] = "graph_rag_analysis"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # 3. FORCE THE EU ENDPOINT
        # Ensure your API Key was actually created in the EU region!
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

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
    """
    Refactored Fallback: Generates the inventory dict with strict segregation.
    Uses EXISTS subqueries for robust filtering of Semantic vs Lexical nodes.
    """
    inventory = {"Entities": {}, "Connections": {}, "Text Mentions": {}}
    driver = get_db_driver()
    
    if not driver:
        return {}
        
    try:
        with driver.session() as session:
            # 1. POPULATE OBJECTS (Nodes)
            labels_result = session.run("CALL db.labels()")
            labels = [r[0] for r in labels_result]
            
            for label in labels:
                # --- A. ENTITIES (Semantic Priority) ---
                # Nodes that have AT LEAST ONE outgoing Semantic relationship.
                q_entities = f"""
                MATCH (n:`{label}`)
                WHERE n.name IS NOT NULL
                AND EXISTS {{ (n)-[r]-() WHERE type(r) <> 'MENTIONED_IN' }}
                RETURN DISTINCT n.name as name
                """
                names_entities = [r["name"] for r in session.run(q_entities)]
                if names_entities:
                    inventory["Entities"][label] = sorted(names_entities)

                # --- B. TEXT MENTIONS (Purely Lexical) ---
                # Nodes that have MENTIONED_IN but NO outgoing Semantic relationships.
                q_lexical = f"""
                MATCH (n:`{label}`)
                WHERE n.name IS NOT NULL
                AND EXISTS {{ (n)-[:MENTIONED_IN]->() }}
                AND NOT EXISTS {{ (n)-[r]->() WHERE type(r) <> 'MENTIONED_IN' }}
                RETURN DISTINCT n.name as name
                """
                names_lexical = [r["name"] for r in session.run(q_lexical)]
                if names_lexical:
                    inventory["Text Mentions"][label] = sorted(names_lexical)

            # 2. POPULATE VERBS (Relationships - Semantic Only)
            rels_result = session.run("CALL db.relationshipTypes()")
            # Exclude MENTIONED_IN from the available connections list
            rels = [r[0] for r in rels_result if r[0] != "MENTIONED_IN"]
            
            for r_type in rels:
                inventory["Connections"][r_type] = [] 

    except Exception as e:
        st.warning(f"DB Fallback failed: {e}")
    finally:
        driver.close()
        
    return inventory

def fetch_sunburst_from_db(selector_type: str, label: str, names: list[str]) -> pd.DataFrame:
    """
    Refactored DB Fetcher.
    Handles 'Entities', 'Connections', and 'Text Mentions' with specific logic.
    """
    driver = get_db_driver()
    if not driver or not names:
        return pd.DataFrame()

    try:
        with driver.session() as session:
            # --- CASE A: RELATIONSHIP CENTRIC (CONNECTIONS) ---
            if selector_type == "Connections":
                # Strict Semantic: Exclude MENTIONED_IN entirely
                query = """
                MATCH (n)-[r]-(m)
                WHERE type(r) IN $names 
                  AND type(r) <> 'MENTIONED_IN'
                RETURN 
                    type(r) as edge, 
                    labels(n)[0] as source_node_label, 
                    labels(m)[0] as connected_node_label, 
                    count(*) as count,
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                LIMIT 2000
                """
                result = session.run(query, names=names)
                data = [r.data() for r in result]
                return pd.DataFrame(data)

            # --- CASE B: TEXT MENTIONS (PURE LEXICAL) ---
            elif selector_type == "Text Mentions":
                # Pure Lexical: Fetch only MENTIONED_IN
                # Logic: Fetch MENTIONED_IN edges, but return columns compatible with Entity View
                query = f"""
                MATCH (n:`{label}`)-[r:MENTIONED_IN]-(m:Document)
                WHERE n.name IN $names
                RETURN 
                    type(r) as edge,            // Returns "MENTIONED_IN"
                    labels(n)[0] as node, 
                    coalesce(n.name, 'Unknown') as node_name, 
                    count(m) as count,
                    'Document' as connected_node_label, // Explicit Target Label
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                """
                result = session.run(query, names=names)
                data = [r.data() for r in result]
                return pd.DataFrame(data)
            
            # --- CASE C: NODE CENTRIC (ENTITIES) ---
            else:
                # Hybrid: Fetch EVERYTHING (Semantic + Lexical)
                # We do NOT filter out MENTIONED_IN here because we need it for the subtraction logic.
                query = f"""
                MATCH (n:`{label}`)-[r]-(m)
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
                df = pd.DataFrame(data)
                
                # Apply Semantic Priority Logic (Subtract Semantic IDs from Lexical IDs)
                return process_entity_sunburst_logic(df)
            
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
    prefix_map = {"Entities": "node", "Connections": "relationship", "Text Mentions": "lexical"}
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
# 3. Processing entities 
# ==========================================
def flatten_ids(container):
    """Recursively flattens a container of IDs (strings/ints/nested lists) into a set."""
    ids = set()
    if isinstance(container, (list, tuple, set)):
        for item in container:
            ids.update(flatten_ids(item))
    elif pd.notna(container):
        ids.add(container)
    return ids

def process_entity_sunburst_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the "Semantic Priority" logic.
    1. Identify Semantic IDs (from non-MENTIONED_IN edges).
    2. Subtract Semantic IDs from MENTIONED_IN edges.
    3. Remove MENTIONED_IN rows that become empty.
    """
    if df.empty or 'edge' not in df.columns or 'id_list' not in df.columns:
        return df
        
    # Check if MENTIONED_IN even exists in this slice
    if 'MENTIONED_IN' not in df['edge'].values:
        return df

    # 1. Separate Semantic vs Lexical
    semantic_df = df[df['edge'] != 'MENTIONED_IN'].copy()
    lexical_df = df[df['edge'] == 'MENTIONED_IN'].copy()
    
    # 2. Collect all Semantic IDs into a set (FLATTENED)
    # This prevents "unhashable type: list" errors if id_list contains nested lists
    semantic_ids = set()
    for ids in semantic_df['id_list']:
        semantic_ids.update(flatten_ids(ids))
            
    # 3. Filter Lexical IDs
    def filter_ids(row_ids):
        # Flatten row_ids first
        row_set = flatten_ids(row_ids)
        # Subtract semantic IDs
        return list(row_set - semantic_ids)
        
    lexical_df['id_list'] = lexical_df['id_list'].apply(filter_ids)
    lexical_df['count'] = lexical_df['id_list'].apply(len)
    
    # 4. Remove empty lexical rows
    lexical_df = lexical_df[lexical_df['count'] > 0]
    
    # 5. Recombine
    return pd.concat([semantic_df, lexical_df], ignore_index=True)
# ==============================================================================
# UPDATED BACKEND: VERB SUPPORT
# ==============================================================================



@st.fragment
def render_explorer_workspace(selector_type, selected_items):
    accent_line = "<hr style='border: 2px solid #00ADB5; opacity: 0.5; margin-top: 15px; margin-bottom: 15px;'>"

    # --- Accessible Hacker Palette ---
    COLOR_ROOT = "#FF8C00"          # Amber (Subject)
    COLOR_RELATIONSHIP = "#4E545C"  # Gunmetal Gray (Relationship)
    COLOR_TARGET = "#00ADB5"        # Teal (Object)
    COLOR_BORDER = "#FFFFFF"        # White borders

    c_mid, c_right = st.columns([2, 1])
    
    with c_mid:
        if not selected_items:
            st.info("👈 Select entities from the left and click 'Show Data'.")
            return

        names = [item['name'] for item in selected_items]
        
        # --- Dynamic Legend ---
        if selector_type == "Connections":
            # Hierarchy: Edge (Gray) -> Source (Amber) -> Target (Teal)
            legend_items = [
                (COLOR_RELATIONSHIP, "Relationship", "Layer 1: The Connection (Root)", "border: 1px solid #666;"),
                (COLOR_ROOT, "Subject (🟠)", "Layer 2: The Source Entity", "box-shadow: 0 0 5px " + COLOR_ROOT + ";"),
                (COLOR_TARGET, "Object Type (🟦)", "Layer 3: The Target Entity", "box-shadow: 0 0 5px " + COLOR_TARGET + ";")
            ]
        else:
            # Hierarchy: Name (Amber) -> Edge (Gray) -> Target (Teal)
            legend_items = [
                (COLOR_ROOT, "Subject", "Layer 1: The Source Entity (Root)", "box-shadow: 0 0 5px " + COLOR_ROOT + ";"),
                (COLOR_RELATIONSHIP, "Relationship (⬜)", "Layer 2: The Action/Connection", "border: 1px solid #666;"),
                (COLOR_TARGET, "Object Type (🟦)", "Layer 3: The Target Entity", "box-shadow: 0 0 5px " + COLOR_TARGET + ";")
            ]
            
        legend_html = '<div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; justify-content: center;">'
        for col, label, title, style_extra in legend_items:
             legend_html += f'<span style="display: flex; align-items: center;" title="{title}"><span style="width: 12px; height: 12px; background: {col}; border-radius: 50%; display: inline-block; margin-right: 5px; {style_extra}"></span>{label}</span>'
        legend_html += '</div>'
        
        st.markdown(legend_html, unsafe_allow_html=True)

        # 1. Fetch Data
        df = fetch_sunburst_data(selector_type, selected_items)

        if df.empty:
            st.warning("No data found.")
            return

        # 2. Prepare Path & Columns based on Selector
        if 'id_list' in df.columns:
            df['id_list_str'] = df['id_list'].astype(str)
            hover_cols = ['id_list_str']
        else:
            hover_cols = None

        if selector_type == "Connections":
            # Hierarchy: Edge -> Source -> Target
            path = ['edge', 'source_node_label', 'connected_node_label']
        else:
            # Hierarchy: Name -> Edge -> Target
            path = ['node_name', 'edge', 'connected_node_label']

        valid_path = [col for col in path if col in df.columns]
        
        if not valid_path:
             st.error("Data columns missing for visualization.")
             return

        # 3. Plot Generation
        fig = px.sunburst(
            df, 
            path=valid_path, 
            values='count',
            hover_data=hover_cols
        )

        # 4. Post-Process: Colors (Layers) & Tooltips (Sentence Structure)
        try:
            sunburst_ids = fig.data[0]['ids']
            colors = []
            hover_texts = []
            
            # --- Pre-calculate Aggregates for Tooltips ---
            # We need to manually sum counts for parent rings to display accurate "Total" numbers in the sentence.
            id_to_count = {}
            for _, row in df.iterrows():
                # Reconstruct Path IDs used by Plotly (Root, Root/Mid, Root/Mid/Leaf)
                root_val = row[path[0]]
                mid_val = row[path[1]]
                leaf_val = row[path[2]]
                
                # Plotly IDs are joined by '/'
                leaf_id = f"{root_val}/{mid_val}/{leaf_val}"
                mid_id = f"{root_val}/{mid_val}"
                root_id = f"{root_val}"
                
                c = row['count']
                id_to_count[leaf_id] = id_to_count.get(leaf_id, 0) + c
                id_to_count[mid_id] = id_to_count.get(mid_id, 0) + c
                id_to_count[root_id] = id_to_count.get(root_id, 0) + c
            
            for id_str in sunburst_ids:
                depth = id_str.count('/')
                parts = id_str.split('/')
                val = id_to_count.get(id_str, 0)
                
                # --- A. COLOR LOGIC ---
                if selector_type == "Connections":
                    # Structure: Relationship (Root) -> Subject (Mid) -> Object (Leaf)
                    if depth == 0:
                        colors.append(COLOR_RELATIONSHIP)
                    elif depth == 1:
                        colors.append(COLOR_ROOT)
                    elif depth >= 2:
                        colors.append(COLOR_TARGET)
                    else:
                        colors.append('#333333')
                else:
                    # Structure: Subject (Root) -> Relationship (Mid) -> Object (Leaf)
                    if depth == 0:
                        colors.append(COLOR_ROOT)
                    elif depth == 1:
                        colors.append(COLOR_RELATIONSHIP)
                    elif depth >= 2:
                        colors.append(COLOR_TARGET)
                    else:
                        colors.append('#333333')

                # --- B. TOOLTIP LOGIC (Sentence Structure) ---
                tooltip_text = ""
                
                if selector_type == "Connections":
                    # [Edge, Source, Target]
                    edge = parts[0]
                    if depth == 0:
                         tooltip_text = f"Connection Type: <b>{edge}</b><br>Total Occurrences: <b>{val}</b>"
                    elif depth == 1:
                        source = parts[1]
                        tooltip_text = f"<b>{source}</b> has <b>{val}</b> {edge} relationships."
                    elif depth == 2:
                        source = parts[1]
                        target = parts[2]
                        tooltip_text = f"<b>{source}</b> has {edge} <b>{val}</b> {target}s."
                
                else:
                    # [Name, Edge, Target]
                    name = parts[0]
                    if depth == 0:
                        tooltip_text = f"Entity: <b>{name}</b><br>Total Connections: <b>{val}</b>"
                    elif depth == 1:
                        edge = parts[1]
                        tooltip_text = f"<b>{name}</b> has <b>{val}</b> {edge} relationships."
                    elif depth == 2:
                        edge = parts[1]
                        target = parts[2]
                        tooltip_text = f"<b>{name}</b> has {edge} <b>{val}</b> {target}s."
                
                hover_texts.append(tooltip_text)

            # Apply Colors AND Custom Tooltips
            fig.update_traces(
                marker=dict(colors=colors), 
                hovertext=hover_texts, 
                hovertemplate="%{hovertext}<br><br><i>For definition, see Glossary below.</i><extra></extra>"
            )
            
        except Exception as e:
            # Fallback if parsing fails
            pass

        # 5. Styling & UX
        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0), 
            height=500,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',  
            font=dict(color="white")       
        )
        
        # Note: We moved hovertemplate update into the loop block above to use the custom text
        fig.update_traces(marker=dict(line=dict(color=COLOR_BORDER, width=2)))
        
        st.plotly_chart(fig, use_container_width=True)

        # 6. Glossary
        visible_edges = sorted(df['edge'].unique()) if 'edge' in df.columns else []
        
        with st.expander("📖 Relationship Glossary", expanded=False):
            if not visible_edges:
                st.caption("No relationships visible.")
            else:
                for edge in visible_edges:
                    definition = get_rel_definition(edge)
                    st.markdown(f"**{edge}**: {definition}")

    with c_right:
        st.subheader("Filter Data", divider = "gray")

        # --- Conditional Filtering Logic ---
        
        if selector_type == "Text Mentions":
            # CASE C: No Filters for Lexical
            st.caption("Standard Extraction: Entities found in Documents via 'MENTIONED_IN'.")
            st.info("No filters applicable.")
            final_filtered_df = df

        elif selector_type == "Connections":
            # CASE B: Connections Mode (Filter by Nodes, not Relationships)
            st.caption("Filter by Source and Target Nodes")
            
            # Filter 1: Source Node (Subject - Amber)
            if 'source_node_label' in df.columns:
                raw_sources = sorted(df['source_node_label'].unique())
                source_options = [f"🟠 {s}" for s in raw_sources]
                
                selected_sources_fmt = st.multiselect(
                    "Filter by Source Type:",
                    options=source_options,
                    default=[], 
                    placeholder="Select source types...",
                    key="filter_source_multi"
                )
                selected_sources = [s.replace("🟠 ", "") for s in selected_sources_fmt]
                
                if not selected_sources:
                    filtered_df_step1 = df
                else:
                    filtered_df_step1 = df[df['source_node_label'].isin(selected_sources)]
            else:
                filtered_df_step1 = df

            # Filter 2: Target Node (Object - Teal)
            if 'connected_node_label' in filtered_df_step1.columns:
                raw_targets = sorted(filtered_df_step1['connected_node_label'].unique())
                target_options = [f"🟦 {t}" for t in raw_targets]

                selected_targets_fmt = st.multiselect(
                    "Filter by Target Type:",
                    options=target_options,
                    default=[], 
                    placeholder="Select target types...",
                    key="filter_target_multi"
                )
                selected_targets = [t.replace("🟦 ", "") for t in selected_targets_fmt]

                if not selected_targets:
                    final_filtered_df = filtered_df_step1
                else:
                    final_filtered_df = filtered_df_step1[filtered_df_step1['connected_node_label'].isin(selected_targets)]
            else:
                final_filtered_df = filtered_df_step1

        else:
            # CASE A: Entities Mode (Standard Logic)
            st.caption("Filter by Relationships and Target Types")

            # Filter 1: Relationship (Gray)
            raw_edges = sorted(df['edge'].unique()) if 'edge' in df.columns else []
            edge_options = [f"⬜ {e}" for e in raw_edges]
            
            selected_edges_fmt = st.multiselect(
                "Filter by Connection Type:",
                options=edge_options,
                default=[], 
                placeholder="Select connections...",
                key="filter_edge_multi"
            )
            selected_edges = [e.replace("⬜ ", "") for e in selected_edges_fmt]

            if not selected_edges:
                filtered_df_step1 = df
            else:
                filtered_df_step1 = df[df['edge'].isin(selected_edges)]
                
            # Filter 2: Target Node (Object - Teal)
            if 'connected_node_label' in filtered_df_step1.columns:
                raw_targets = sorted(filtered_df_step1['connected_node_label'].unique())
                target_options = [f"🟦 {t}" for t in raw_targets]

                selected_targets_fmt = st.multiselect(
                    "Filter by Target Type:",
                    options=target_options,
                    default=[], 
                    placeholder="Select target types...",
                    key="filter_target_multi"
                )
                selected_targets = [t.replace("🟦 ", "") for t in selected_targets_fmt]

                if not selected_targets:
                    final_filtered_df = filtered_df_step1
                else:
                    final_filtered_df = filtered_df_step1[filtered_df_step1['connected_node_label'].isin(selected_targets)]
            else:
                final_filtered_df = filtered_df_step1


        # --- ID Extraction & Evidence Cart ---
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
        string = "Documents Found: " + str(len(unique_ids)) 
        st.text(string)
        # Uncomment to allow the preview of ID lists in the app. Useful for debugging
        # with st.expander("Preview ID List", expanded=False):
        #     st.write(unique_ids)

        st.markdown(accent_line, unsafe_allow_html=True)
        st.subheader(":arrow_down_small: Add to Evidence Cart :arrow_down_small:", divider="gray")
        
        if st.button("Add to Evidence Cart", type="primary", use_container_width=True):
            if not unique_ids:
                st.error("No documents to add.")
            else:
                if len(names) > 1:
                    if selector_type == "Connections":
                        name_str = f"Connections: {', '.join(names)}"
                    else:
                        name_str = f"Entities: {', '.join(names)}"
                else:
                    name_str = names[0]
                    
                query_desc = f"Manual Explorer: {name_str}"
                
                payload = {
                    "query": query_desc,
                    "answer": f"Visual discovery found {len(unique_ids)} related documents.",
                    "ids": [str(uid) for uid in unique_ids]
                }
                
                if "evidence_locker" not in st.session_state.app_state:
                    st.session_state.app_state["evidence_locker"] = []
                    
                st.session_state.app_state["evidence_locker"].append(payload)
                st.toast(f"✅ Added {len(unique_ids)} docs to Evidence Cart!")


        
# ==========================================
# 4. MAIN SCREEN CONTROLLER
# ==========================================

def screen_databook():
    st.title("Find Evidence Manually")
    
    inventory = fetch_inventory()
    
    if "databook_selections" not in st.session_state:
        st.session_state.databook_selections = set()

    if "active_explorer_items" not in st.session_state:
        st.session_state.active_explorer_items = []

    if "last_selector_type" not in st.session_state:
        st.session_state.last_selector_type = "Entities"

    if "widget_reset_token" not in st.session_state:
        st.session_state.widget_reset_token = 0

    if not inventory:
        st.warning("⚠️ Could not load Inventory (GitHub or DB). check connection.")
    
    c_left, c_workspace = st.columns([1, 3])

    with c_left:
        with st.container(border=True):
            st.subheader("Selector")
            
            selector_type = st.radio(
                "Analysis Mode", 
                ["Entities", "Connections", "Text Mentions"], 
                captions=["Semantic (Node)", "Semantic (Verb)", "Lexical (Node)"],
                horizontal=True
            )
            
            if selector_type != st.session_state.last_selector_type:
                st.session_state.databook_selections = set()
                st.session_state.active_explorer_items = []
                st.session_state.last_selector_type = selector_type
                st.session_state.widget_reset_token += 1
                
                for key in list(st.session_state.keys()):
                    if key.startswith("chk_"):
                        del st.session_state[key]
                        
                st.rerun()

            st.divider()

            selection_count = len(st.session_state.databook_selections)
            
            c_vis, c_clear = st.columns([2, 1])
            with c_vis:
                if st.button(f"Show Data({selection_count})", type="primary", use_container_width=True):
                    st.session_state.active_explorer_items = [
                        {'label': l, 'name': n} for l, n in st.session_state.databook_selections
                    ]
            with c_clear:
                if st.button("Clear", use_container_width=True):
                    st.session_state.databook_selections = set()
                    st.session_state.active_explorer_items = []
                    st.session_state.widget_reset_token += 1
                    
                    for key in list(st.session_state.keys()):
                        if key.startswith("chk_"):
                            del st.session_state[key]
                            
                    st.rerun()

            st.divider()

            with st.container(height=400, border=False):
                available_data = inventory.get(selector_type, {})
                
                if not available_data:
                    st.caption(f"No items found for {selector_type}.")
                else:
                    if isinstance(available_data, dict):
                        token = st.session_state.widget_reset_token

                        if selector_type in ["Entities", "Text Mentions"]:
                            labels = sorted(list(available_data.keys()))
                            for label in labels:
                                search_key = f"search_{selector_type}_{label}"
                                is_expanded = bool(st.session_state.get(search_key, ""))

                                with st.expander(f"{label}", expanded=is_expanded):
                                    raw_vals = available_data[label]
                                    clean_names = []
                                    if isinstance(raw_vals, dict):
                                        clean_names = [v for v in raw_vals.values() if v and pd.notna(v)]
                                    elif isinstance(raw_vals, list):
                                        clean_names = [v for v in raw_vals if v and pd.notna(v)]
                                    names = sorted(list(set(str(n) for n in clean_names)))
                                    
                                    if names:
                                        c_search, c_btn = st.columns([5, 1])
                                        with c_search:
                                            search_term = st.text_input(
                                                f"Search {label}", 
                                                placeholder=f"Filter...", 
                                                key=search_key,
                                                label_visibility="collapsed"
                                            )
                                        with c_btn:
                                            st.button("⏎", key=f"btn_{search_key}",  use_container_width=True)

                                        filtered_names = [n for n in names if search_term.lower() in n.lower()] if search_term else names
                                        
                                        if not filtered_names:
                                            st.caption("No matches.")
                                        else:
                                            display_names = filtered_names[:50] if (len(filtered_names) > 50 and not search_term) else filtered_names
                                            if len(filtered_names) > 50 and not search_term:
                                                st.info(f"Showing 50 of {len(filtered_names)}.")

                                            for name in display_names:
                                                is_selected = (label, name) in st.session_state.databook_selections
                                                chk_key = f"chk_{token}_{selector_type}_{label}_{name}"
                                                
                                                def update_selection(l=label, n=name, k=chk_key):
                                                    if st.session_state[k]:
                                                        st.session_state.databook_selections.add((l, n))
                                                    else:
                                                        st.session_state.databook_selections.discard((l, n))

                                                st.checkbox(name, value=is_selected, key=chk_key, on_change=update_selection)
                                    else:
                                        st.caption("No names.")

                        elif selector_type == "Connections":
                            rel_types = sorted(list(available_data.keys()))
                            if rel_types:
                                search_key = f"search_{selector_type}"
                                
                                c_search, c_btn = st.columns([5, 1])
                                with c_search:
                                    search_term = st.text_input(
                                        "Search Relationships", 
                                        placeholder="Filter...", 
                                        key=search_key,
                                        label_visibility="collapsed"
                                    )
                                with c_btn:
                                    st.button("⏎", key=f"btn_{search_key}", help="Apply Filter", use_container_width=True)
                                
                                filtered_rels = [r for r in rel_types if search_term.lower() in r.lower()] if search_term else rel_types
                                
                                for r_type in filtered_rels:
                                    is_selected = ("Connections", r_type) in st.session_state.databook_selections
                                    chk_key = f"chk_{token}_verb_{r_type}"
                                    
                                    def update_verb_selection(t=r_type, k=chk_key):
                                        if st.session_state.get(k, False):
                                            st.session_state.databook_selections.add(("Connections", t))
                                        else:
                                            st.session_state.databook_selections.discard(("Connections", t))
                                    
                                    st.checkbox(r_type, value=is_selected, key=chk_key, on_change=update_verb_selection)
                            else:
                                st.caption("No Connection types found.")
                    else:
                        st.error("Invalid inventory format.")

    with c_workspace:
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
    """
    Loads external context data efficiently and performs deduplication logic.
    Also synthetically creates 'Bates_Identity' and ensures 'Text Link' exists.
    """
    url = "https://raw.githubusercontent.com/ssopic/some_data/main/sum_data.csv"
    try:
        df = pd.read_csv(url)
        
        # 1. Basic Cleaning
        # Convert PK to string, remove potential '.0' from floats, and strip whitespace
        df['PK'] = df['PK'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # 2. Bates & Link Column Handling
        # The source data might not have these columns, so we initialize them if missing.
        for col in ['Bates Begin', 'Bates End', 'Text Link']:
            if col not in df.columns:
                df[col] = "" # Initialize if missing
            # Clean: to string, remove float decimals, strip whitespace
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
        # 3. Create Bates Identity (User Requirement)
        # Since 'Bates Identity' does not exist in source, we construct it: "Begin End"
        df['Bates_Identity'] = df['Bates Begin'] + " " + df['Bates End']
        # Fallback: If result is just whitespace (missing bates), use "Doc_PK" to ensure UI has a label
        df['Bates_Identity'] = df['Bates_Identity'].apply(lambda x: x if x.strip() else None)
        df['Bates_Identity'] = df['Bates_Identity'].fillna("Doc_" + df['PK'])
        
        # 4. Sequence Logic Processing
        if 'chain_sequence_order' in df.columns:
            # Ensure sequence is numeric for correct sorting
            df['chain_sequence_order'] = pd.to_numeric(df['chain_sequence_order'], errors='coerce')
            
            # Sort by sequence descending (Highest first)
            df = df.sort_values(by="chain_sequence_order", ascending=False)
            
            # Drop duplicates on PK, keeping the 'first' (highest sequence)
            df = df.drop_duplicates(subset=["PK"], keep="first")
            
        return df
    except Exception as e:
        # Fallback empty dataframe structure with all required columns
        return pd.DataFrame(columns=['PK', 'Bates Begin', "Bates End", "Body", "Text Link", "chain_sequence_order", "Bates_Identity"])


# --- SCHEMA DEFINITIONS (Preserved from Old Version) ---

class StructuredBlueprint(BaseModel):
    intent: str = Field(description="The primary action: FindEntity, FindPath, MultiHopAnalysis, etc.")
    target_entity_nl: str = Field(description="Natural language term for the primary entity.")
    source_entity_nl: str = Field(description="Natural language term for the starting entity or context.")
    complexity: str = Field(description="Simple, MultiHop, or Aggregation.")
    proposed_relationships: List[str] = Field(default_factory=list, description="List of sequential relationships or verbs. EXTRACT VERBATIM. Do not normalize. If user says 'bribed', output 'bribed'.")
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
    # NEW: Friction Reducers for when the user specifically asks for verbs and labels.
    verb_mapping: Dict[str, str] = Field(
        default_factory=dict, 
        description="Map from Blueprint Verb (key) to Schema Relationship Type (value). Example: {'paid': 'FINANCIAL_TRANSACTION'}"
    )
    entity_mapping: Dict[str, str] = Field(
        default_factory=dict, 
        description="Map from Blueprint Keyword (key) to Schema Node Label (value). Example: {'island': 'LOCATION'}"
    )

class SynthesisOutput(BaseModel):
    final_answer: str
    source_document_ids: List[str] = Field(default_factory=list)

class CypherWrapper(BaseModel):
    query: str

# --- SYSTEM PROMPTS (Preserved) ---

SYSTEM_PROMPTS = {
  "Intent Planner": """
You are a Cypher query planning expert. Analyze the user's natural language query.
The user will almost always input their question in English but if the question is in another language, you must translate the users question to English prior to continuing with the following steps:

TASK:
1. Extract entities and relationships.
2. CRITICAL - VERB EXTRACTION: You must extract the verbs EXACTLY as the user spoke them. Do NOT normalize them into categories.
   - Example: If user says 'Who visited and paid?', output ['visited', 'paid'], NOT ['travel_event', 'financial_transaction'].

3. Determine the 'complexity' using this strict taxonomy:
   - 'Simple': Direct attribute lookups or single-hop neighbors (e.g., "Find John", "Who paid John?").
   - 'MultiHop': Queries requiring traversal of 2+ edges or finding paths between entities (e.g., "How is John connected to Mary?", "Friends of friends").
   - 'Aggregation': Questions asking for counts, maximums, minimums, or averages (e.g., "How many...", "Who has the most...").

RELATIONSHIPS: Extract the sequence of actions or verbs into 'proposed_relationships' as a list.
VERB FILTERS: If the user specifically asks for exact phrases (e.g., 'relationships with the verb "stocks of"'), extract those exact strings into 'filter_on_verbs'.

EXAMPLES:

Input: "Who paid John Doe?"
Output: {{
  "intent": "FindEntity",
  "target_entity_nl": "John Doe",
  "source_entity_nl": "Who",
  "complexity": "Simple",
  "proposed_relationships": ["paid"],
  "filter_on_verbs": [],
  "constraints": []
}}

Input: "How is 'Project Omega' connected to the 'Oversight Committee'?"
Output: {{
  "intent": "MultiHopAnalysis",
  "target_entity_nl": "Oversight Committee",
  "source_entity_nl": "Project Omega",
  "complexity": "MultiHop",
  "proposed_relationships": ["connected"],
  "filter_on_verbs": [],
  "constraints": []
}}

Input: "Show me all interactions where the verb is explicitly 'bribed'."
Output: {{
  "intent": "FindPath",
  "target_entity_nl": "all",
  "source_entity_nl": "all",
  "complexity": "Simple",
  "proposed_relationships": ["bribed"],
  "filter_on_verbs": ["bribed"],
  "constraints": []
}}

Input: "How much did Mary give Johnny where that Johnny does not have a surname Smith?"
Output: {{
  "intent": "Aggregation",
  "target_entity_nl": "Johnny",
  "source_entity_nl": "Mary",
  "complexity": "Simple",
  "proposed_relationships": ["give"],
  "filter_on_verbs": [],
  "constraints": ["Johnny surname IS NOT 'Smith'"]
}}
""",

    "Schema Selector": """
You are a Schema Context manager. Your goal is to select the relevant graph schema elements for the user's query and CREATE A MAP for the downstream agents.

INPUTS:
1. BLUEPRINT: The user's intent and proposed relationships.
2. FULL_SCHEMA: The actual Node Labels, Relationship Types, AND their Properties.

TASK:
1. Select ONLY the Node Labels and Relationship Types relevant to the blueprint.
2. MAPPING (CRITICAL): You must explicity map the User's terminology to the Schema's terminology.
   - `verb_mapping`: Map each verb in `proposed_relationships` to the Schema Relationship Type.
     * STRICT RULE: The value MUST be the EXACT string from the schema (e.g., "FINANCIAL_TRANSACTION", not "Financial Transaction").
     * Example: `{{"paid": "FINANCIAL_TRANSACTION", "visited": "TRAVEL_EVENT"}}`
   - `entity_mapping`: Map entities/constraints to the Schema Node Label.
     * Example: `{{"island": "LOCATION", "company": "ORGANIZATION"}}`

3. Populate 'NodeProperties' and 'RelationshipProperties' with the valid keys for the selected types.
""",

    "Grounding Agent": """
You are a Graph Grounding expert. Map the blueprint to the specific Schema provided.

INPUTS:
- BLUEPRINT: User's original intent.
- SCHEMA: Pruned schema WITH MAPPING GUIDES (`verb_mapping`, `entity_mapping`).

TASK: Create a blueprint where 'relationship_paths' uses the EXACT relationship types from the SCHEMA.

RULES:
1. USE THE MAP: Do not guess. Look at `schema.verb_mapping`.
   - If the map says "paid" -> "FINANCIAL_TRANSACTION", use "FINANCIAL_TRANSACTION".

2. HANDLING AMBIGUITY (OR Logic): If the user targets multiple types, use pipes `|`.
   - Example Nodes: `Person|Organization`

3. STRICT ATTRIBUTE FILTERING (CRITICAL):
   - **TARGET NODES ONLY:** You may ONLY apply name filters to nodes labeled `PERSON`. Do NOT filter `LOCATION`, `ORGANIZATION`, etc. by name (map them to Labels instead).
   - **FUZZY MATCHING REQUIRED:** When filtering properties, you MUST use `toLower(n.prop) CONTAINS 'value'`.
     * FORBIDDEN: Do NOT use strict equality (`=`).
     * BAD: `WHERE n.name = 'John'`
     * GOOD: `WHERE toLower(n.name) CONTAINS 'john'`
   - **NEGATIVE CONSTRAINTS:** For "is not X", use `NOT toLower(n.name) CONTAINS 'x'`.
     * FORBIDDEN: Do NOT use `<>` or `!=`.
     * BAD: `WHERE n.name <> 'Gates'`
     * GOOD: `WHERE NOT toLower(n.name) CONTAINS 'gates'`
   - **LOWERCASE VALUES:** Always convert the target string to lowercase in the query (e.g. `'john'`, not `'John'`).

4. PATH PARSIMONY: Do not add redundant or mirrored relationships.
   - If the Blueprint asks for 2 steps (e.g., "paid" and "visited"), generate exactly 2 relationships. Do not hallucinate a 3rd step or "double back" unless the logic strictly demands it.

PROVENANCE FOR RELATIONSHIPS: Return provenance from the RELATIONSHIPS. Use `coalesce(r.source_pks)` to make sure the user can properly analyze the results.
PROVENANCE FOR Documents: If the relationship is 'MENTIONED_IN'. Use `coalesce(d.doc_id)` for nodes labeled as 'Document'.
CONSTRAINT RULE: Do NOT use properties in the WHERE clause that are not listed in the Schema's NodeProperties.
""",
       "Cypher Generator": """
You are an expert Cypher Generator. Convert the Grounded Component into a VALID, READ-ONLY Cypher query.

RULES:
1. SAFE RETURN POLICY (STRICT):
   - For NODES: You MUST ONLY return the `.name` property (e.g., `n.name`).
   - DO NOT return generic properties like `.title`, `.age`, `.role` even if the user asks, as they are unreliable.
   - For PROVENANCE: Always return `coalesce(r.source_pks, r.doc_id)`.

2. STRING MATCHING (MANDATORY): For ALL string property filters in WHERE clauses, you MUST use `toLower(n.prop) CONTAINS 'value'`.
   - BAD: `WHERE n.name = 'John Doe'`
   - GOOD: `WHERE toLower(n.name) CONTAINS 'john doe'`
   - **NEGATIVE MATCHING:** For exclusion, use `NOT ... CONTAINS`.
     - BAD: `WHERE n.name <> 'Gates'`
     - GOOD: `WHERE NOT toLower(n.name) CONTAINS 'gates'`

3. PATHS & LOGIC:
   - **Continuous Paths:** Ensure the path is fully connected. `(a)-[r1]->(b)-[r2]->(c)`. NEVER use comma-separated disconnected patterns like `MATCH (a), (b)` (Cartesian Product).
   - **OR Logic:** If the Grounding Agent provided pipes `|` in labels (e.g., `Person|Organization`), write them EXACTLY as provided in the Cypher (e.g., `(n:Person|Organization)`).

4. PROPERTIES IN WHERE CLAUSES: You may use other properties (e.g. `.date`, `.status`) ONLY in the `WHERE` clause to filter data, and ONLY if they are explicitly listed in the Schema.
5. DISTINCT: Always use `RETURN DISTINCT` to avoid duplicates.
6. SYNTAX: Do not include semicolons at the end.
""",
    "Query Debugger": (
        "You are an expert Neo4j Debugger. Analyze the error, warnings, and the failed query. "
        "1. If the error mentions missing properties (e.g. 'properties does not exist'), change `r.properties` to `properties(r)`.\n"
        "2. If fixing a path syntax error, ensure you do not create self-loops like `(n)--(n)`. This returns no data. Ensure nodes are distinct.\n"
        "3. SCHEMA CHECK: You MUST NOT invent new relationship types or properties. You must check the `schema` provided in the context and only use elements listed there.\n"
        "Provide a `fixed_cypher` string with the correction."
    ),
    "Synthesizer": """
You are a helpful Data Analyst / Investigator. Your goal is to answer the user's question directly based on the database results.

GUIDELINES:
1. **ANSWER FIRST**: Start immediately with the findings. Do NOT explain the Cypher query structure (e.g., "I matched a Person node...") unless the results are ambiguous and require technical context.
   - Clearly admit the limitations of the cypher. The database does not allow for filtering on the names of anything other than Persons. If a name of an island or organization is being used, clearly state
   that it might extract non relevant data as a precaution to make sure that the relevant data is extracted. 
   - YES: "I found 36 individuals who fit the criteria, including..."
   - NO: "The query used a MATCH clause to find..."

2. **CATEGORIZE FINDINGS**:
   - **Key Individuals**: List specific, recognizable names (e.g., "Bill Clinton", "Prince Andrew").
   - **Ambiguous Entries**: Group generic terms (e.g., "Sender", "You", "She") separately so they don't clutter the main answer.

3. **CONTEXT**: Briefly mention that these results come from a broad relationship search and may include indirect connections.

4. **PROVENANCE**: Mention that the specific documents linking these people can be found in the evidence locker (referenced by the IDs in the data).

5. **EMPTY RESULTS**: If the result list is empty, state clearly "No matching records found in the current dataset using the current query."
"""
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


@st.fragment
def screen_extraction():
    st.title("Chat with helper or write your own cypher")
    
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
    st.title("Evidence Cart")
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
    
    # Retrieve selected IDs safely
    ids = list(st.session_state.app_state.get("selected_ids", []))
    
    if not ids:
        st.warning("No documents selected.")
        return
    
    # Get the main dataframe (Already cleaned and has Bates_Identity)
    df = st.session_state.github_data

    # Filter for selected IDs using the cleaned dataframe
    matched = df[df['PK'].isin(ids)]
    
    st.subheader(f"Analyzing {len(matched)} Documents")
    
    if not matched.empty:
        # --- 1. Prepare LLM Context (Using Bates Identity) ---
        llm_context = ""
        for _, row in matched.iterrows():
            # Use the synthetic Bates Identity
            bates_id = row.get('Bates_Identity', 'Unknown Doc')
            body_text = row.get('Body') or 'No Content'
            # Update Context to use Bates ID instead of PK
            llm_context += f"Document: {bates_id}\nContent: {body_text}\n---\n"

        # --- 2. Document Reader Interface ---
        # Selectbox allows user to view specific doc details without filtering LLM context
        doc_options = matched['Bates_Identity'].tolist()
        
        # Dropdown to select document
        selected_bates = st.selectbox("Select Document to Read:", options=doc_options)
        
        # Find the row corresponding to selection
        if selected_bates:
            # Safe retrieval of the specific row
            view_row = matched[matched['Bates_Identity'] == selected_bates].iloc[0]
            
            # Metadata Display
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("Bates Begin", view_row['Bates Begin'])
                c2.metric("Bates End", view_row['Bates End'])
                # Text Link display (Read-only or Link if valid URL)
                link_val = view_row['Text Link']
                if link_val and link_val.startswith('http'):
                    c3.link_button("Open Text Link", link_val)
                else:
                    c3.text_input("Text Link", value=link_val if link_val else "N/A", disabled=True)
            
            # Content Display
            st.caption("Document Body")
            st.text_area("Body content", view_row.get('Body', ''), height=300, label_visibility="collapsed", disabled=True)
            
        # --- 3. Chat Interface ---
        # Optional: Show what the LLM sees
        with st.expander("View LLM Context (Bates Format)"):
            st.text(llm_context)
            
        q = st.chat_input("Ask about this evidence:")
        if q:
            # Assuming get_cached_llm is available
            llm = get_cached_llm(st.session_state.app_state["mistral_key"])
            resp = llm.invoke(f"Context:\n{llm_context}\n\nQuestion: {q}")
            st.info(resp.content)
    
# --- MAIN NAVIGATION ---
def inject_custom_css():
    """
    Hides the standard Streamlit sidebar and applies styling for the cockpit layout.
    """
    st.markdown(
        """
        <style>
            /* 1. Hide the default Streamlit Sidebar elements */
            [data-testid="stSidebar"] { display: none; }
            [data-testid="collapsedControl"] { display: none; }
            
            /* 2. Main Layout Adjustment */
            .stApp {
                background-color: #0E1117; /* Deep Slate / Black */
                color: #FFFFFF;
            }

            .block-container {
                padding-top: 4rem; 
                padding-bottom: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
                max_width: 100%;
            }

            /* 3. TECH BUTTON STYLING */
            /* Target ALL buttons. We use 'background' shorthand to nuking any default gradients/images. */
            div.stButton > button {
                width: 100%;
                background: #1F2129 !important; /* Force Dark Background (Shorthand) */
                background-color: #1F2129 !important;
                color: #FFFFFF !important; 
                border: 1px solid #41444C !important; /* Force Dark Border */
                border-radius: 4px;
                
                /* SIZE ADJUSTMENT: Matches standard input height (approx 42px) */
                min-height: 42px !important; 
                height: auto !important;
                padding-top: 0.25rem !important;
                padding-bottom: 0.25rem !important;

                font-family: 'Source Sans Pro', sans-serif;
                font-weight: 700 !important;
                letter-spacing: 0.5px;
                transition: all 0.2s ease-in-out; 
                box-shadow: none !important;
            }
            
            /* Force text inside button to be white */
            div.stButton > button p {
                color: #FFFFFF !important;
            }

            /* Focus/Active States */
            div.stButton > button:focus,
            div.stButton > button:active {
                background: #1F2129 !important;
                color: #FFFFFF !important;
                border-color: #41444C !important;
                box-shadow: none !important;
            }

            /* ACCENT: Cyan Border & Text on Hover */
            div.stButton > button:hover {
                background: #1F2129 !important; 
                border-color: #00ADB5 !important; /* Cyan Accent */     
                color: #00ADB5 !important;        /* Cyan Text */
                box-shadow: 0 0 4px rgba(0, 173, 181, 0.3) !important; /* Subtle Glow */          
            }
            
            /* Hover Text Color */
            div.stButton > button:hover p {
                color: #00ADB5 !important;
            }
            
            /* 4. GLOBAL TEXT STYLING */
            h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stText, .stCaption {
                color: #FFFFFF !important;
                font-weight: 600 !important; 
            }

            /* FIX FOR MULTI-LINE HEADERS (st.subheader with \n) */
            /* Forces <p> tags created by newlines inside headers to match the header size */
            h1 p, h2 p, h3 p, h4 p, h5 p, h6 p,
            h1 span, h2 span, h3 span, h4 span, h5 span, h6 span {
                font-size: inherit !important;
                font-weight: inherit !important;
                color: inherit !important;
                line-height: 1.2 !important; 
                margin-bottom: 0 !important;
            }

            /* 5. Tighter Dividers */
            hr {
                border-color: #41444C;
                margin-top: 0.5em !important;
                margin-bottom: 0.5em !important;
            }

            /* 6. EXPANDER (DATABOOK DROPDOWNS) FIX */
            
            /* Target the HTML <details> and <summary> elements directly */
            
            /* The Container (Closed state usually) */
            div[data-testid="stExpander"] details {
                background-color: #1F2129 !important;
                border-color: #41444C !important;
                border-radius: 4px;
            }

            /* The Clickable Header (Summary) */
            div[data-testid="stExpander"] summary {
                background-color: #1F2129 !important;
                color: #FFFFFF !important;
                border: 1px solid #41444C !important;
                border-radius: 4px;
                transition: border-color 0.2s, color 0.2s;
            }

            /* ACCENT: Hover state for the Expander Header */
            div[data-testid="stExpander"] summary:hover {
                background-color: #1F2129 !important; 
                border-color: #00ADB5 !important; /* Cyan Border */
                color: #00ADB5 !important; /* Cyan Text */
            }

            /* Force the text inside the summary (the label) to be white/cyan */
            div[data-testid="stExpander"] summary span,
            div[data-testid="stExpander"] summary p {
                 color: inherit !important;
            }

            /* The SVG Arrow inside the header */
            div[data-testid="stExpander"] summary svg {
                fill: #FFFFFF !important;
            }
            div[data-testid="stExpander"] summary:hover svg {
                fill: #00ADB5 !important;
            }
            
            /* The Content Box that opens up */
            div[data-testid="stExpander"] div[role="group"] {
                 background-color: #0E1117 !important; /* Match main background */
                 color: #FFFFFF !important;
                 border: 1px solid #41444C;
            }

            /* 7. CHECKBOX & INPUT FIXES INSIDE EXPANDER */
            
            /* Checkbox Label Text */
            label[data-baseweb="checkbox"] span {
                color: #FFFFFF !important;
            }
            
            /* 8. MULTISELECT & DROPDOWN LIST FIXES */

            /* The Selected Tags (Chips) */
            span[data-baseweb="tag"] {
                background-color: #31333F !important; /* Distinct from bg */
                color: #FFFFFF !important;
                border: 1px solid #41444C;
            }
            
            /* The 'X' icon in tags */
            span[data-baseweb="tag"] svg {
                fill: #FFFFFF !important;
            }
            
            /* The Dropdown Menu Container */
            div[data-baseweb="popover"],
            div[data-baseweb="menu"] {
                background-color: #1F2129 !important;
                border: 1px solid #41444C !important;
            }
            
            /* The Options in the Dropdown */
            li[role="option"] {
                background-color: #1F2129 !important;
                color: #FFFFFF !important;
            }
            
            /* Hover/Selected Option */
            li[role="option"]:hover,
            li[role="option"][aria-selected="true"] {
                background-color: #31333F !important;
                color: #00ADB5 !important; /* Cyan Text */
            }
            
            /* Fix for MultiSelect Input Container Background */
            div[data-baseweb="select"] > div {
                background-color: #1F2129 !important;
                color: #FFFFFF !important;
                border-color: #41444C !important;
            }

            /* 9. NOTIFICATIONS & ALERTS (Toasts) */
            
            div[data-testid="stToast"] {
                background-color: #1F2129 !important;
                border: 1px solid #41444C !important;
                color: #FFFFFF !important;
            }
            
            /* Ensure text inside toast is white */
            div[data-testid="stToast"] p, 
            div[data-testid="stToast"] div {
                color: #FFFFFF !important;
            }
            
            /* Alerts (st.success, st.info, etc) */
            div[data-testid="stAlert"] {
                background-color: #1F2129 !important;
                color: #FFFFFF !important;
                border: 1px solid #41444C;
            }
            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] div {
                color: #FFFFFF !important;
            }

            /* 10. CODE BLOCKS & TRACES (Fix for Cypher/Trace visibility) */
            
            /* The code block container */
            div[data-testid="stCodeBlock"] {
                background-color: #0E1117 !important;
                border: 1px solid #41444C;
                border-radius: 4px;
            }
            
            /* The code text itself */
            code {
                color: #00ADB5 !important; /* Cyan for code/traces */
                background-color: transparent !important; 
                font-family: 'Courier New', monospace !important;
            }
            
            /* Preformatted text blocks */
            pre {
                background-color: #0E1117 !important;
                color: #FFFFFF !important;
                border: 1px solid #41444C;
            }

            /* 11. JSON & RAW TEXT FIXES (For Traces/Debug Data) */
            
            /* Target the JSON viewer specifically (st.json) */
            div[data-testid="stJson"],
            .react-json-view {
                background-color: #0E1117 !important;
                color: #FFFFFF !important;
            }
            
            /* Force keys/values in JSON to be visible */
            .react-json-view span {
                color: #00ADB5 !important; /* Cyan for values */
            }
            
            /* st.text() raw output */
            div[data-testid="stText"] {
                background-color: #0E1117 !important;
                color: #00ADB5 !important; /* Cyan for raw text logs */
                font-family: 'Courier New', monospace !important;
            }

            /* 12. SETTINGS DIALOG (MODAL) FIXES */

            /* The Modal Background (The popup box itself) */
            div[role="dialog"][aria-modal="true"] {
                background-color: #1F2129 !important;
                border: 1px solid #41444C !important;
                color: #FFFFFF !important;
            }

            /* Modal Header */
            div[role="dialog"] header {
                background-color: #1F2129 !important;
                color: #FFFFFF !important;
            }

            /* Content Text inside Modal */
            div[role="dialog"] div, 
            div[role="dialog"] label,
            div[role="dialog"] p {
                color: #FFFFFF !important;
            }

            /* Close Button (X) */
            button[aria-label="Close"] {
                color: #FFFFFF !important;
                background-color: transparent !important;
                border: none !important;
            }
            button[aria-label="Close"]:hover {
                color: #00ADB5 !important;
            }

            /* 13. VERTICAL SEPARATION LINES (Column Borders) - DEPTH-BASED SELECTOR */
            
            /* LEFT FRAME (First Column) */
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(1),
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(1),
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1),
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) {
                border-right: 3px solid #00ADB5 !important; 
                background-color: #14171F; 
                padding: 1rem !important;
                min-height: 85vh; 
                box-shadow: 5px 0 15px -5px rgba(0, 173, 181, 0.4); 
            }
            
            /* RIGHT FRAME (Last Column) */
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
                border-left: 3px solid #00ADB5 !important;
                background-color: #14171F;
                padding: 1rem !important;
                min-height: 85vh;
                box-shadow: -5px 0 15px -5px rgba(0, 173, 181, 0.4); 
            }
            
            /* Reset nested columns just in case */
            [data-testid="stColumn"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"],
            [data-testid="column"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
                border: none !important;
                background-color: transparent !important;
                box-shadow: none !important;
                min-height: 0 !important;
                padding: 0 !important;
            }

            /* 14. SEARCH BAR & TEXT INPUT FIXES (FIXING WHITE-ON-WHITE) */
            
            /* The Container */
            div[data-testid="stTextInput"] div[data-baseweb="input"] {
                background-color: #1F2129 !important;
                border-color: #41444C !important;
                border-radius: 4px;
            }
            
            /* The Input Field Itself */
            div[data-testid="stTextInput"] input {
                color: #FFFFFF !important;  
                background-color: #1F2129 !important;
                caret-color: #00ADB5 !important; /* Cyan caret */
            }
            
            /* Placeholder Text */
            div[data-testid="stTextInput"] input::placeholder {
                color: #B0B0B0 !important;
            }

            /* Focus state for text input */
            div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
                border-color: #00ADB5 !important;
                box-shadow: 0 0 2px rgba(0, 173, 181, 0.5);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
accent_line = "<hr style='border: 2px solid #00ADB5; opacity: 0.5; margin-top: 15px; margin-bottom: 15px;'>"

def set_page(page_name):
    """Helper to update the current page in session state."""
    st.session_state.current_page = page_name

def main():
    # 1. Setup & Styling
    st.set_page_config(layout="wide", page_title="Graph Analyst")
    
    # Initialize LangSmith env vars immediately
    setup_langsmith()
    
    inject_custom_css()
    
    # Initialize app (secrets, state)
    init_app()

    # Initialize Page State if not present
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Find Evidence via Chat or Cypher"

    # 2. Connection Gatekeeper
    if not st.session_state.app_state["connected"]:
        st.info("👋 Welcome! The app is disconnected. Please connect below.")
        if st.button("⚙️ Settings"):
            show_settings_dialog()
        return

    # 3. Cockpit Layout (3 Columns)
    # Ratios: [1.2, 8, 1.2]
    c_left, c_main, c_right = st.columns([1.2, 8, 1.2], gap="medium")

    # --- LEFT COLUMN (Input & Config) ---
    with c_left:
        #st.markdown("Find Evidence and add to cart", text_alignment="center")
        st.markdown("<div style='text-align: center; font-size: 1.1em;'><b>Find Evidence and add to cart</b></div>", unsafe_allow_html=True)
        st.markdown(accent_line, unsafe_allow_html=True)
        
        
        # Navigation Buttons (Using callbacks for single-click nav)
        st.button("Find Evidence Manually",  use_container_width=True, 
                  on_click=set_page, args=("Find Evidence Manually",))
            
        st.button("Find Evidence via Chat or Cypher", use_container_width=True,
                  on_click=set_page, args=("Find Evidence via Chat or Cypher",))

        # Vertical Spacer to push Config to bottom
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        
        st.divider()
        
        # Settings at bottom
        if st.button("⚙️ Config", use_container_width=True):
            show_settings_dialog()

    # --- CENTER COLUMN (Main Router) ---
    with c_main:
        with st.container():
            # Router Logic
            current = st.session_state.current_page
            
            if current == "Find Evidence Manually":
                screen_databook()
            elif current == "Find Evidence via Chat or Cypher":
                screen_extraction()
            elif current == "Evidence Cart":
                screen_locker()
            elif current == "Analysis":
                screen_analysis()
            else:
                st.error(f"Unknown page: {current}")

    # --- RIGHT COLUMN (Output & Tools) ---
    with c_right:
        st.markdown("Select from Cart and Analyze", text_alignment="center")
        st.markdown(accent_line, unsafe_allow_html=True)
        
        
        # Locker Badge Calculation
        locker_count = len(st.session_state.app_state["evidence_locker"])
        badge = f" ({locker_count})" if locker_count > 0 else ""
        
        st.button(f"Evidence Cart",  use_container_width=True,
                  on_click=set_page, args=("Evidence Cart",))
            
        st.button("📈 Analysis",  use_container_width=True,
                  on_click=set_page, args=("Analysis",))
            
        # Vertical Spacer to push Logout to bottom
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

        st.divider()
        
        # Logout
        if st.button("Logout", use_container_width=True):
            st.session_state.app_state["connected"] = False
            st.session_state.has_tried_login = False
            st.rerun()


### RELATIONSHIP DEFINITONS ##
RELATIONSHIP_DEFINITIONS= {
    "ABILITY": "Refers to the functional capacity or practical feasibility of an entity to perform a specific task. It highlights the availability of space, time, or technical resources required to meet an objective. (Example: Confirming a building has 'plenty of room' for equipment or determining if a person 'can get there' for a scheduled event.)",
    "ACHIEVEMENT": "Signifies the successful attainment of professional milestones, the receipt of prestigious honors, or significant breakthroughs in a field. (Example: Receiving a humanitarian award or reaching a major financial benchmark like €1 billion in assets under management.)",
    "ACTION": "Captures the execution of logistical tasks, formal procedures, or specific physical movements intended to achieve a result. (Example: Signing a legal release form or arranging for a private aircraft to transport a colleague.)",
    "AFFILIATION": "Denotes an entity's formal associations, institutional ties, or professional status. It identifies organizational memberships and social connections that establish a person's role or identity. (Example: Holding a professorship at a university or being a partner in a specific investment group.)",
    "ANALYSIS": "Describes the systematic evaluation, interpretation, or comparative critique of information and concepts. It involves breaking down complex topics to provide judgment or insight. (Example: Contrasting the political strategies of different candidates or evaluating the impact of economic trends.)",
    "ASSISTANCE": "Refers to the provision of resources, advocacy, or logistical aid to support another entity's goals or resolve their challenges. This includes financial backing, professional endorsements, and personal guidance. (Example: Providing 'wonderful support' to a cultural project like Poetry in America or using political influence to help an associate secure a government appointment.)",
    "ATTENTION_DRAWING": "Involves directing focus toward a specific subject, variable, or event to highlight its importance or potential impact. It often appears in the context of investigations or financial warnings where certain details are singled out for closer scrutiny. (Example: Focusing on an investigation into Saudi royal finances or highlighting the negative return characteristics of a specific investment.)",
    "AUDITION/CASTING": "Relates to the selection and assignment of individuals to specific roles in professional or creative productions. This includes career-defining casting decisions and strategic 'casting' used to facilitate the production of a work. (Example: Being cast in a first movie role or casting a specific person in a play to ensure it gets produced.)",
    "CAUSALITY": "Describes a systemic relationship where one event or policy change triggers a shift in behavior or strategy across a group. It focuses on the 'ripple effect' of broad actions. (Example: How government budget cuts caused scientists to turn to private donors for funding.)",
    "CAUSATION": "Refers to a direct link between an event and an immediate, often personal or punitive, consequence for an individual. It highlights a clear 'action-result' pair in a specific case. (Example: A person's 'unhappy situation' being the direct result of legal or disciplinary proceedings.)",
    "CHALLENGE": "Represents the experience of significant difficulty, formal opposition, or public scrutiny. It often involves legal conflict, ethical reprimands, or being 'in the sights' of an investigator. (Example: Facing a federal judge's 'harsh rebuke' for failing to disclose information or navigating a 'professional misconduct' investigation.)",
    "CHANGE": "Denotes a transition, modification, or reversal in a particular state or perception. This includes the replacement of information, shifts in public reputation, or the withdrawal of a candidate from a process. (Example: Successfully replacing a 'mug shot' on Wikipedia with a different photo or a official withdrawing from a high-profile job race.)",
    "COMMUNICATION": "Encompasses the general exchange of information, inquiries, and the mention of entities within media or correspondence. It covers status updates, networking introductions, and instances where a subject is simply 'addressed' or 'named.' (Example: An email asking to 'call when you get a chance' or a news article that 'mentions' a person's name in relation to a professional topic.)",
    "CONFLICT": "Describes a state of active disagreement, opposition, or a clash between different interests and viewpoints. It often manifests as public policy debates, legal resistance, or the rejection of specific plans. (Example: Opposing a city council's moratorium on development or being involved in a 'raging' debate between critics and supporters of a new law.)",
    "CREATION": "Describes the process of originating, developing, or producing something new. This includes creative works, business ventures, conceptual frameworks, and physical structures. (Example: Producing a television series like 'Poetry in America,' launching a new publication like 'Cavalier' magazine, or establishing a foundation.)",
    "DEVELOPMENT": "Refers to the iterative process of advancing, building, or refining projects, technologies, or products. It emphasizes growth and technical improvement rather than just the initial act of creation. (Example: Spending years 'translating a discovery' into a new medical drug, building a software database, or expanding a business's technology suite.)",
    "DOCUMENT_INTERACTION": "Refers to the creation, handling, or referencing of formal records and published media. This includes writing reports, citing sources, filing legal paperwork, or being the subject of a news article. (Example: Filing a 'HIPAA release,' being 'featured' in a major newspaper, or citations in a formal report.)",
    "EDUCATION": "Describes the formal process of teaching, academic training, and the pursuit of knowledge within institutional settings. It covers being a student or professor, taking specific courses, and the attainment (or abandonment) of degrees. (Example: Teaching a class to 'Japanese Engineers,' studying at a university like Northwestern, or obtaining a Master's degree from Harvard.)",
    "EMOTION": "Captures the expression of internal feelings, subjective reactions, and interpersonal sentiments. It tracks the psychological or emotional state of an entity in response to events, people, or information. (Example: Feeling 'devastated' by an association, being 'excited' to start a project, or expressing 'obsession' with a particular subject.)",
    "EVALUATION": "Relates to the act of assessing or assigning a qualitative or quantitative value to an entity. It focuses on the final 'rating' or 'ranking' given by an authority or expert. (Example: Reporting on how municipal bonds were 'rated' by Moody’s or determining that a country ranks 'at the bottom' for growth outlook.)",
    "EVENT_PARTICIPATION": "Denotes an entity's presence at or active involvement in organized gatherings, social functions, or public performances. This covers attending professional conferences, participating in artistic rehearsals, or being present at social events. (Example: Performing in a community theater production, attending the World Economic Forum in Davos, or participating in a specific summit.)",
    "FINANCIAL_TRANSACTION": "Relates to the movement, exchange, or management of monetary resources and assets. It covers a wide range of fiscal activities, including investments, the purchase of goods or services, charitable donations, and the handling of budgets. (Example: Recommending 'bank stocks' to buy, handling a '$25K donation' that bounced, or managing the 'budget and timing of funding' for a production.)",
    "GAMBLING": "Refers to the act of placing stakes or taking financial risks based on the predicted outcome of a future event, such as an election or a market pivot. It frames speculation as a 'win/loss' scenario rather than traditional long-term investment. (Example: An absolute majority 'betting on HRC to win' the election or making market trades based on high-risk political 'scenarios.')",
    "GIFT": "Identifies the voluntary transfer of assets, property, or services to another party without receiving payment in return. This includes charitable contributions, the gifting of high-value real estate, and offers of professional time. (Example: Giving a '$50 million townhouse' to a friend or offering several hours of professional expertise as a 'birthday present.')",
    "GROWTH": "Describes the scaling up or progression of a project or entity into a more substantial stage of existence. It highlights increasing complexity, the addition of new members, or the move from a planning phase to full execution. (Example: Moving from 'preproduction' to 'production' or expanding a project to include new contributors.)",
    "GUIDANCE": "Involves the sharing of information, news, or legal updates intended to provide advice or direction to a person or group. It focuses on helping others navigate changes in social or political landscapes. (Example: Forwarding a news article about new marriage benefits to help inform 'the gay community.')",
    "INFLUENCE": "Refers to the exertive force an entity has on the thoughts, behaviors, or decision-making of another. This includes applying professional pressure, providing strategic coaching, or leveraging reputation to sway an outcome. (Example: Receiving 'pressure' from senior supporters to take a leave of absence or coaching an associate on how to refute specific public charges.)",
    "KNOWLEDGE/OPINION": "Denotes an entity's subjective beliefs, stances, or conceptual framing of external reality. It involves making predictions, assigning qualitative traits to individuals, and asserting logical parallels between events. (Example: Predicting that a resignation will be a 'game changer' or asserting that a specific political figure is a 'heavyweight.')",
    "LEGAL_ACTION": "Encompasses the formal processes and adversarial interactions associated with the justice system. This includes allegations of crimes, the filing of lawsuits, the execution of search warrants, and the representation of clients by attorneys. (Example: Being 'charged with a sex crime,' filing a civil suit for 'malicious prosecution,' or authorizing a 'search warrant.')",
    "LEGAL_REPRESENTATION": "Refers to the formal professional relationship where an attorney or legal expert acts on behalf of a client. This involves providing defense, handling negotiations, and serving as a legal spokesperson. (Example: Ken Starr and Alan Dershowitz being the 'lawyers for' a client during proceedings or David Boies representing specific accusers.)",
    "MEDIA_BROADCAST": "Refers to the act of airing, broadcasting, or featuring content in high-reach media outlets like television, radio, and major press organizations. It covers the filming of interviews, the dissemination of news shows, and the public release of documentaries or televised series. (Example: Airing a primary debate on Fox News, being the subject of a BBC 'Today' programme interview, or being 'featured' in a major television series.)",
    "MEDICAL": "Relates to biological health, pharmaceuticals, and the study or treatment of physical conditions. This includes clinical trials, the pathology of diseases (such as parasites or viruses), and the regulatory approval of medical drugs. (Example: Conducting a 'clinical trial program' for cholesterol medication or analyzing the effect of 'parasite load' on biological development.)",
    "MONITORING": "Refers to the persistent observation, surveillance, or systematic tracking of entities, behaviors, and communications. This includes following news cycles, tracking digital presence, or using intelligence to keep 'eyes on' a specific situation or individual. (Example: Watching 'phone lines,' tracking the progress of a 60-day clock on a deal, or monitoring the digital reputation of an individual.)",
    "MOVEMENT": "Refers to the physical relocation of entities, the logistics of travel, and the specific geographic positioning of people or objects. It tracks departures, arrivals, visits, and the methods of transportation used to facilitate movement between locations. (Example: Flying from 'DC to Brussels,' being 'on the way to the airport,' or returning to a specific residence.)",
    "NEED": "Describes a requirement, necessity, or strategic imperative that must be fulfilled. It identifies logistical needs, professional mandates, or procedural demands that an entity must address to achieve a goal. (Example: Stating that a person 'needs squashed' for political reasons, 'requiring' a signature on a legal document, or needing 'marching orders' to proceed.)",
    "OTHER": "Serves as a miscellaneous category for interactions, relationships, and actions that do not fit into specific professional or legal labels. It covers incidental associations, general inclusion, broad impacts, and unclassified physical or conceptual links. (Example: Being 'included' in a general list, 'causing damage' to a property, or 'filming with' a colleague outside of a formal event.)",
    "OWNERSHIP": "Refers to the possession, control, or legal title an entity has over assets, property, information, or objects. This includes real estate, financial asset classes, personal belongings, and the possession of sensitive media or data. (Example: Owning a 'palatial home,' having 'video recordings' in one's possession, or the state of 'where she keeps her' specific items.)",
    "PARTICIPATION": "Describes an entity's engagement, role, or active involvement in a broad project, social movement, or ongoing situation. It captures the state of being 'part of' a process or having a stake in a complex unfolding event. (Example: Having a 'peripheral' role in a scandal, 'volunteering' for an international program, or becoming 'involved with' a large-scale real estate development.)",
    "PHYSICAL_ACTION": "Refers to concrete physical events, natural processes, or forceful physical interactions that occur in the material world. It covers natural phenomena, physical violence, and the forced physical removal or protection of entities. (Example: An ice sheet 'melting' into the ocean, an associate who 'stabbed' a government official, or being 'booted off' a team.)",
    "PLANNING": "Refers to the formulation of future intentions, strategies, schedules, and logistical arrangements. It covers high-level strategic maneuvering, professional scheduling, and the consideration of alternatives or contingency plans. (Example: Discussing a 'plot' to have a candidate lose an election, 'scheduling' a meeting for a specific time, or 'planning' a visit to a country.)",
    "POLITICS": "Relates to the exercise of power, institutional governance, and the electoral process. It encompasses campaigning for office, the mechanics of voting and vetoes, party leadership dynamics, and geopolitical maneuvering between states. (Example: Being 'elected' to office, 'campaigning for' a candidate, or 'awaiting a return to power' in a specific country.)",
    "PREPARATION": "Relates to the state of readiness or the immediate logistical coordination required to execute a plan. It captures the 'readying' phase where entities confirm platforms, time zones, or availability to ensure an upcoming interaction can proceed. (Example: Declaring one is 'ready when you are' for a series or checking for the best platform to use for a scheduled call.)",
    "PROFESSIONAL_RELATIONSHIP": "Refers to formal connections and collaborative structures within a business or organizational context. It encompasses employment, management hierarchy, contractual partnerships, and professional hiring. (Example: Being 'managed by' a project lead, 'partnering with' another firm for a drug launch, or 'employing' a specific individual for research.)",
    "PROTECTION": "Refers to actions taken to safeguard an entity, asset, or reputation from harm, interference, or unauthorized access. This includes digital security measures, physical safety in high-risk environments, and the prevention of negative outcomes or professional sabotage. (Example: Securing domains to prevent 'hijacks,' using incognito mode to protect search integrity, or mastering the 'danger' of a high-crime area.)",
    "RELATED_TO": "Describes a broad, contextual, or logical association between entities. It encompasses social ties, topical relevance, and general connectivity that indicates one entity is relevant to another without a specific procedural or professional role. (Example: Having a 'friendship with' a person, being 'at the center of' a news cycle, or being 'tied up in' a specific investigation.)",
    "REQUIREMENT": "Relates to an essential condition, prerequisite, or unavoidable obligation that must be met for an entity to function or a process to move forward. It covers functional dependencies, regulatory mandates, and critical needs like funding. (Example: Institutional needs that 'required' a reduction in spending or an individual who 'really needed' money for basic survival.)",
    "RESEARCH": "Refers to the systematic examination, inquiry, or probing into individuals, entities, or events—often with the intent to uncover hidden, obscured, or previously unknown information. This includes formal investigations and the pursuit of evidence related to legal, financial, or reputational matters. The language used is active and purposeful, emphasizing discovery and the pursuit of facts, frequently in the context of legal or media-driven inquiries. (Example: Launched an investigation of offshore accounts.)",
    "SELECTION": "Refers to the deliberate act of choosing, nominating, or identifying specific individuals, options, or assets from a broader set, often for a particular role, opportunity, or distinction. This includes the process of being selected for awards, appointments, or inclusion in exclusive lists, as well as the strategic choice of investments, candidates, or partners. The language used is decisive and outcome-oriented, emphasizing the act of picking, nominating, or being chosen for a defined purpose.",
    "SHARED_CONTENT": "Refers to the act of distributing, forwarding, or making available information, media, or documents to one or more recipients. This includes sharing articles, links, images, legal documents, or other digital content—often to inform, persuade, or prompt discussion. The language and context suggest a focus on the dissemination of news, analysis, or evidence, frequently in real-time or as part of ongoing dialogue.",
    "SOCIAL_INTERACTION": "Refers to the informal, interpersonal exchanges and relational dynamics between individuals, often characterized by casual conversation, humor, personal updates, and emotional support. This includes discussions about social plans, personal experiences, travel, professional gossip, and the sharing of opinions or advice. The language is typically conversational, sometimes playful or empathetic, and reflects the nuances of personal relationships, social bonding, and the navigation of both public and private spheres. (Example: 'Ask him about a sandwich,' 'I envy you,' 'Don’t let them get you to be emotional. Breathe! Think judicial demeanor.')",
    "STATE_OF_BEING": "Refers to the condition, status, or existence of a person, entity, organization, or situation at a given time. This includes descriptions of legal, financial, or reputational states, as well as assessments of stability, vulnerability, or transformation. The language often reflects evaluations of risk, certainty, or inevitability, and may involve discussions of ongoing investigations, legal exposure, organizational health, or personal circumstances.",
    "SUPPLY": "Refers to the provision, delivery, or facilitation of goods, services, information, or access—often in response to a specific request or need. This includes the arrangement of physical items (such as tickets, books, or technology), the sharing of specialized knowledge or resources, and the coordination of logistical support. The language used is transactional and solution-oriented, emphasizing the ability to source, deliver, or enable access to desired assets or opportunities.",
    "SUPPORT": "Refers to the provision of assistance, encouragement, or backing—whether emotional, strategic, logistical, or professional—to an individual or group. This includes offering advice, sharing resources, advocating for someone’s position, or helping to navigate complex personal, political, or professional challenges. The language used is often empathetic, directive, or collaborative, reflecting a commitment to the recipient’s well-being, success, or resilience.",
    "TIMING": "Refers to the scheduling, coordination, or sequencing of events, meetings, or actions—often with strategic, logistical, or symbolic significance. This includes the arrangement of appointments, the alignment of activities with external events (such as anniversaries, deadlines, or political transitions), and the consideration of timing as a tactical element in negotiations, public relations, or personal interactions. The language used highlights urgency, opportunity, or the importance of synchronization.",
    "USAGE": "Refers to the act of employing, leveraging, or repurposing resources, information, or assets for a specific purpose or goal. This includes the strategic application of media, data, or personal connections to achieve an outcome, as well as the adaptation of content, platforms, or networks for new or expanded uses. The language used is functional and outcome-oriented, emphasizing the practical or tactical deployment of available tools or opportunities."
}
def get_rel_definition(rel_name):
    """Helper to safely get a definition or a default prompt."""
    return RELATIONSHIP_DEFINITIONS.get(rel_name, "Relationship connection between entities.")
    
if __name__ == "__main__":
    main()
    
