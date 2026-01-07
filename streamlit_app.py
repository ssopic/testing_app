import streamlit as st
import os
import json
import re
import pandas as pd
import difflib
from typing import List, Dict, Any, Optional, Set, Union
from neo4j import GraphDatabase, Driver, exceptions as neo4j_exceptions

# --- LangChain/Mistral Imports ---
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError

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
        df['PK'] = df['PK'].astype(str)
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
        "Determine if this is a simple lookup or a 'MultiHop' query.\n"
        "RELATIONSHIPS: Extract the sequence of actions or verbs into 'proposed_relationships' as a list.\n"
        "VERB FILTERS: If the user specifically asks for exact phrases, extract those exact strings into 'filter_on_verbs'."
    ),
    "Schema Selector": (
        "You are a Schema Context manager. Select ONLY the Node Labels and Relationship Types relevant to the blueprint.\n"
        "Crucially, populate 'NodeProperties' and 'RelationshipProperties' with the valid keys for the selected types.\n"
        "CRITICAL RULE: Pass the valid property lists. This prevents the generator from inventing properties."
    ),
    "Grounding Agent": (
        "You are a Graph Grounding expert. Map the blueprint to the specific Schema provided.\n"
        "CHAIN OF THOUGHT (Required):\n"
        "1. Analyze Entities: Is 'Island' a Person or a Location?\n"
        "2. Check Properties: Does the Label have the property?\n"
        "3. Define Path: Ensure the path includes nodes with correct Labels.\n"
        "PROVENANCE: Return provenance from the RELATIONSHIPS. Use `coalesce(r.source_pks, r.doc_id)`."
    ),
    "Cypher Generator": (
        "You are an expert Cypher Generator. Convert the Grounded Component into a VALID, READ-ONLY Cypher query.\n"
        "RULES:\n"
        "1. PATHS: Assign variables to ALL relationships.\n"
        "2. PROPERTIES: Only use properties explicitly listed in the Schema.\n"
        "3. FUZZY MATCHING: Use `toLower(n.name) CONTAINS` for strings.\n"
        "4. VERB FILTERS: If 'filter_on_verbs' is provided, add a WHERE clause checking `r.raw_verbs`.\n"
        "5. PROVENANCE: Return provenance from the relationship variables using `coalesce(r.source_pks, r.doc_id)`.\n"
        "6. DISTINCT: Always use `RETURN DISTINCT`."
    ),
    "Query Debugger": (
        "You are an expert Neo4j Debugger. Analyze the error and warnings. "
        "If missing properties, use `properties(r)`. Do not invent new types. Provide a `fixed_cypher` string."
    ),
    "Synthesizer": (
        "You are a conversational AI. Synthesize a concise answer based on the provided DB preview. "
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

def extract_provenance_from_result(result: Dict) -> List[str]:
    """Scans JSON recursively for any keys containing 'provenance'."""
    ids = []
    def recurse(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if "provenance" in k.lower():
                    if isinstance(v, list): ids.extend([str(i) for i in v])
                    else: ids.append(str(v))
                else: recurse(v)
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

    def _run_agent(self, agent_name: str, output_model: BaseModel, context: Dict) -> Any:
        sys_prompt = SYSTEM_PROMPTS.get(agent_name, "")
        formatted_context = {}
        for k, v in context.items():
            if isinstance(v, BaseModel): formatted_context[k] = v.model_dump_json(indent=2)
            elif isinstance(v, (dict, list)): formatted_context[k] = json.dumps(v, indent=2)
            else: formatted_context[k] = str(v)

        msgs = [("system", sys_prompt)]
        
        # Prompt Mapping
        if agent_name == "Intent Planner": msgs.append(("human", "QUERY: {user_query}"))
        elif agent_name == "Schema Selector": msgs.append(("human", "BLUEPRINT: {blueprint}\nFULL_SCHEMA: {full_schema}"))
        elif agent_name == "Grounding Agent": msgs.append(("human", "BLUEPRINT: {blueprint}\nSCHEMA: {schema}\nQUERY: {user_query}"))
        elif agent_name == "Cypher Generator": msgs.append(("human", "GROUNDED_COMPONENT: {blueprint}"))
        elif agent_name == "Query Debugger": msgs.append(("human", "ERROR: {error}\nQUERY: {failed_query}\nBLUEPRINT: {blueprint}\nWARNINGS: {warnings}"))
        elif agent_name == "Synthesizer": msgs.append(("human", "QUERY: {user_query}\nCYPHER: {final_cypher}\nRESULTS: {db_result}"))
        
        prompt = ChatPromptTemplate.from_messages(msgs)
        return (prompt | self.llm.with_structured_output(output_model)).invoke(formatted_context)

    def run(self, user_query: str) -> Dict[str, Any]:
        pipeline_object = {"user_query": user_query, "status": "INIT", "execution_history": [], "proof_ids": []}
        
        try:
            # 1. Intent
            pipeline_object["status"] = "PLANNING"
            blueprint = self._run_agent("Intent Planner", StructuredBlueprint, {"user_query": user_query})
            
            # 2. Schema
            full_schema = self._get_full_schema()
            pruned_schema = self._run_agent("Schema Selector", PrunedSchema, {"blueprint": blueprint, "full_schema": full_schema})
            
            # 3. Grounding
            grounding = self._run_agent("Grounding Agent", GroundingOutput, {"blueprint": blueprint, "schema": pruned_schema.model_dump(), "user_query": user_query})
            
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
                             attempt_log["warnings"] = [{"code": n.code, "message": n.description} for n in summary.notifications]
                        
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

def screen_connection():
    st.title("🔗 Connection Gatekeeper")
    with st.container(border=True):
        m_key = st.text_input("Mistral API Key", type="password", key="m_key")
        n_uri = st.text_input("Neo4j URI", key="n_uri")
        n_user = st.text_input("Neo4j User", value="neo4j", key="n_user")
        n_pass = st.text_input("Neo4j Password", type="password", key="n_pass")
        
        if st.button("🚀 Connect"):
            with st.spinner("Validating..."):
                stats = fetch_schema_statistics(n_uri, (n_user, n_pass))
                if stats["status"] == "SUCCESS":
                    st.session_state.app_state.update({
                        "connected": True, "mistral_key": m_key,
                        "neo4j_creds": {"uri": n_uri, "user": n_user, "pass": n_pass, "auth": (n_user, n_pass)},
                        "schema_stats": stats
                    })
                    st.rerun()
                else: st.error(stats.get("error"))

# FRAGMENT: Updates only this section when clicking preview
@st.fragment
def screen_databook():
    st.title("📚 The Databook")
    stats = st.session_state.app_state["schema_stats"]
    search = st.text_input("Fuzzy Filter Labels")
    
    col_n, col_r = st.columns(2)
    with col_n:
        st.subheader("Nodes")
        labels = stats.get("NodeLabels", [])
        if search: labels = difflib.get_close_matches(search, labels, n=10, cutoff=0.3)
        
        for label in labels:
            with st.expander(f"{label} ({stats['NodeCounts'].get(label, 0)})"):
                st.write("**Properties:**", stats["NodeProperties"].get(label, []))
                if st.button(f"Preview {label}", key=f"p_{label}"):
                    # Use CACHED driver
                    creds = st.session_state.app_state["neo4j_creds"]
                    driver = get_cached_driver(creds["uri"], creds["auth"])
                    with driver.session() as s:
                        st.json([r.data() for r in s.run(f"MATCH (n:`{label}`) RETURN n LIMIT 3")])

    with col_r:
        st.subheader("Relationships")
        rels = stats.get("RelationshipTypes", [])
        if search: rels = difflib.get_close_matches(search, rels, n=10, cutoff=0.3)
        
        for rtype in rels:
            with st.expander(rtype):
                st.write("**Verbs:**", stats["RelationshipVerbs"].get(rtype, []))
                st.write("**Properties:**", stats["RelationshipProperties"].get(rtype, []))

# FRAGMENT: Updates only the chat area when interacting
@st.fragment
def screen_extraction():
    st.title("🔍 Extraction & Cypher Sandbox")
    
    # 1. Define Tabs
    tab_chat, tab_cypher = st.tabs(["💬 Agent Chat", "🛠️ Raw Cypher"])
    
    # --- TAB 1: EXISTING AGENT CHAT ---
    with tab_chat:
        # Check Connections
        creds = st.session_state.app_state["neo4j_creds"]
        driver = get_cached_driver(creds["uri"], creds["auth"])
        llm = get_cached_llm(st.session_state.app_state["mistral_key"])
        
        if not driver or not llm:
            st.warning("System unavailable. Please check secrets.")
            return
            
        # Pipeline
        # FIX: Access schema_stats from app_state
        pipeline = GraphRAGPipeline(driver, llm, st.session_state.app_state["schema_stats"])

        # Chat UI
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]): st.write(chat["content"])

        user_msg = st.chat_input("Ask about the graph...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"): st.write(user_msg)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing public dataset..."):
                    result = pipeline.run(user_msg)
                    ans = result.get("final_answer", "")
                    st.write(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    
                    if result.get("proof_ids"):
                        st.session_state.evidence_locker.append({
                            "query": user_msg, "answer": ans, "ids": result["proof_ids"]
                        })
                        st.toast("Evidence saved to locker")

    # --- TAB 2: RAW CYPHER INPUT (NEW) ---
    with tab_cypher:
        st.markdown("### Safe Cypher Execution")
        st.caption("Read-Only mode active. Modifications (CREATE, SET, DELETE) are blocked.")
        
        # Default query to help the user start
        cypher_input = st.text_area("Enter Cypher Query", height=150, value="MATCH (n) RETURN n LIMIT 5")
        
        if st.button("Run Query"):
            # A. Security Check
            if SAFETY_REGEX.search(cypher_input):
                st.error("🚨 SECURITY ALERT: destructive commands (DELETE, MERGE, etc.) are not allowed.")
            else:
                # B. Execution
                creds = st.session_state.app_state["neo4j_creds"]
                driver = get_cached_driver(creds["uri"], creds["auth"])
                
                if driver:
                    try:
                        with driver.session() as session:
                            res = session.run(cypher_input)
                            data = [r.data() for r in res]
                            
                            if data:
                                st.dataframe(pd.DataFrame(data), use_container_width=True)
                                st.success(f"Returned {len(data)} records.")
                            else:
                                st.warning("Query returned no results.")
                    except Exception as e:
                        st.error(f"Cypher Syntax Error: {e}")
# FRAGMENT: Locker interaction updates independently
@st.fragment
def screen_locker():
    st.title("🗄️ Evidence Locker")
    locker = st.session_state.app_state["evidence_locker"]
    
    if not locker:
        st.info("Locker is empty.")
        return

    for i, entry in enumerate(locker):
        with st.container(border=True):
            c1, c2 = st.columns([0.1, 0.9])
            with c1:
                # Use a unique key for every checkbox to avoid state conflict in fragment
                is_sel = st.checkbox("Select", key=f"sel_{i}")
                if is_sel: 
                    for pid in entry["ids"]: st.session_state.app_state["selected_ids"].add(pid)
            with c2:
                st.write(f"**Query:** {entry['query']}")
                st.caption(f"Found IDs: {', '.join(entry['ids'])}")

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
            
        q = st.text_input("Ask about this evidence:")
        if q:
            llm = get_cached_llm(st.session_state.app_state["mistral_key"])
            resp = llm.invoke(f"Context:\n{context}\n\nQuestion: {q}")
            st.info(resp.content)

# --- MAIN NAVIGATION ---

if not st.session_state.app_state["connected"]:
    screen_connection()
else:
    nav = st.sidebar.radio("Navigation", ["Databook", "Search", "Locker", "Analysis"])
    if nav == "Databook": screen_databook()
    elif nav == "Search": screen_extraction()
    elif nav == "Locker": screen_locker()
    elif nav == "Analysis": screen_analysis()
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        st.session_state.app_state["connected"] = False
        st.rerun()
