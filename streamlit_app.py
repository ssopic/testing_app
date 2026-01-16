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
    # The grounding Agent has been modified to never use ANY other labels and properties other than for the persons name.  
    "Grounding Agent": (
        "You are a Graph Grounding expert. Map the blueprint to the specific Schema provided.\n"
        "CHAIN OF THOUGHT (Required):\n"
        "1. Analyze Entities: Is 'Island' a Person or a Location? Check the Schema labels.\n"
        "2. If you are sure the Node is a Person feel free to use the .name property. If it is any other label, leave the details empty.\n"
        "3. Define Path: If the destination is a Location, ensure the right relationship type is being used (e.g., `(p)-[:MOVED]->(l:)`).\n\n"
        "TASK: Create a blueprint where 'relationship_paths' uses the EXACT relationship types from the SCHEMA.\n"
        "MULTI-HOP: The 'proposed_relationships' list (e.g., ['paid', 'visited']) must be mapped to the valid schema types provided in the SCHEMA list.\n"
        "PROVENANCE FOR RELATIONSHIPS: Return provenance from the RELATIONSHIPS. Use `coalesce(r.source_pks)` to make sure the user can properly analyze the results. .\n"
        "PROVENANCE FOR Documents: If the relationship is 'MENTIONED_IN'. Use `coalesce(d.doc_id)` for nodes labeled as 'document'.\n"
        
        "CONSTRAINT RULE: Do NOT use properties in the WHERE clause that are not listed in the Schema's NodeProperties."

# Old variant saved for a version when the nodes are properly cleaned       
#        "1. Analyze Entities: Is 'Island' a Person or a Location? Check the Schema labels.\n"
#        "2. Check Properties: Does the `PERSON` label have a 'type' property? If no, do not use `n.type`.\n"
#        "3. Define Path: If the destination is a Location, ensure the path includes a node with that Label (e.g., `(p)-[:MOVED]->(l:LOCATION)`).\n\n"
#        "TASK: Create a blueprint where 'relationship_paths' uses the EXACT relationship types from the SCHEMA.\n"
#        "MULTI-HOP: The 'proposed_relationships' list (e.g., ['paid', 'visited']) must be mapped to the valid schema types provided in the SCHEMA list.\n"
#        "PROVENANCE: Return provenance from the RELATIONSHIPS. Use `coalesce(r.source_pks, r.doc_id)` to handle both fields.\n"
#        "CONSTRAINT RULE: Do NOT use properties in the WHERE clause that are not listed in the Schema's NodeProperties."
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
        "3. FUZZY MATCHING: For names/strings, prefer `toLower(n.name) CONTAINS 'island'` over strict equality `=` to handle messy data.\n"
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
    # FIX: Access schema_stats from app_state
    stats = st.session_state.app_state["schema_stats"]
    
    if stats.get("status") != "SUCCESS":
        st.error("Failed to load schema.")
        return

    # Unified Filter for both columns
    filter_text = st.text_input("Filter Labels & Types", placeholder="e.g. Person")

    col_nodes, col_rels = st.columns(2)

    # --- LEFT COLUMN: NODES (Advanced Features) ---
    with col_nodes:
        st.subheader("Nodes")
        labels = stats.get("NodeLabels", [])
        if filter_text:
            labels = [l for l in labels if filter_text.lower() in l.lower()]
        
        for label in labels:
            count = stats["NodeCounts"].get(label, 0)
            with st.expander(f"📦 {label} ({count:,})"):
                st.caption(f"Properties: {', '.join(stats['NodeProperties'].get(label, [])[:5])}...")
                
                # --- ADVANCED SHOWCASING RESTORED ---
                # 1. Search Bar specific to this node type
                node_search = st.text_input(f"Search {label}", key=f"s_{label}", placeholder="Value to find...")
                limit = st.selectbox("Limit", [5, 10, 50], key=f"l_{label}")
                
                if st.button(f"Fetch Data", key=f"b_{label}"):
                    # FIX: Credentials from app_state
                    creds = st.session_state.app_state["neo4j_creds"]
                    driver = get_cached_driver(creds["uri"], creds["auth"])
                    
                    if driver:
                        with driver.session() as session:
                            if node_search:
                                # Advanced Query: Search ANY property for the term
                                query = f"""
                                MATCH (n:`{label}`) 
                                WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $term)
                                RETURN n LIMIT toInteger($limit)
                                """
                                params = {"term": node_search, "limit": limit}
                            else:
                                # Standard Fetch
                                query = f"MATCH (n:`{label}`) RETURN n LIMIT toInteger($limit)"
                                params = {"limit": limit}
                            
                            results = session.run(query, params)
                            data = [r["n"] for r in results]
                            
                            if data:
                                # Render as interactive Table
                                st.dataframe(pd.DataFrame(data), use_container_width=True)
                            else:
                                st.info("No results found.")

    # --- RIGHT COLUMN: RELATIONSHIPS ---
    with col_rels:
        st.subheader("Relationships")
        rels = stats.get("RelationshipTypes", [])
        if filter_text:
            rels = [r for r in rels if filter_text.lower() in r.lower()]
            
        for rtype in rels:
            with st.expander(f"🔗 {rtype}"):
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
                    result = pipeline.run(user_msg)
                    
                    # --- ERROR HANDLING FIX START ---
                    if result.get("status") == "ERROR":
                        err_msg = result.get("error", "Unknown Error")
                        st.error(f"Pipeline Error: {err_msg}")
                        # Optionally add technical details
                        with st.expander("Technical Details"):
                            st.write(result)
                    else:
                        # Success Path
                        ans = result.get("final_answer", "No answer generated.")
                        st.write(ans)
                        
                        # Show Cypher if available (Debugging)
                        if result.get("cypher_query"):
                            with st.expander("Generated Cypher"):
                                st.code(result["cypher_query"], language="cypher")

                        # Save answer to app_state
                        st.session_state.app_state["chat_history"].append({"role": "assistant", "content": ans})
                        
                        if result.get("proof_ids"):
                            # Save evidence to app_state
                            st.session_state.app_state["evidence_locker"].append({
                                "query": user_msg, "answer": ans, "ids": result["proof_ids"]
                            })
                            st.toast("Evidence saved to locker")
                    # --- ERROR HANDLING FIX END ---

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
            
        q = st.chat_input("Ask about this evidence:")
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
