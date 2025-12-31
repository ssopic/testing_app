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

# Regex for safety check: Detects write/modification keywords
SAFETY_REGEX = re.compile(r"(?i)\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|INSERT|ALTER|GRANT|REVOKE)\b")


# --- SCHEMA DEFINITIONS ---

class StructuredBlueprint(BaseModel):
    """Output schema for the Intent Planner."""
    intent: str = Field(description="The primary action: FindEntity, FindPath, MultiHopAnalysis, etc.")
    target_entity_nl: str = Field(
        description="Natural language term for the primary entity (e.g., 'CEO', 'Jeffrey Epstein').")
    source_entity_nl: str = Field(
        description="Natural language term for the starting entity or context (e.g., 'Twitter', 'Island').")
    complexity: str = Field(description="Simple, MultiHop, or Aggregation.")
    proposed_relationships: List[str] = Field(default_factory=list,
                                              description="List of sequential relationships or verbs (e.g., ['PAID', 'VISITED']).")
    filter_on_verbs: List[str] = Field(default_factory=list,
                                       description="Specific raw verbs or phrases to filter by (e.g. ['stocks of', 'used to buy']).")
    constraints: List[str] = Field(description="Temporal or attribute constraints.")
    properties_to_return: List[str] = Field(description="Properties the user wants to see.")


class GroundingOutput(BaseModel):
    """
    Output of the Grounding Agent.
    """
    # NEW: Chain of Thought field to force reasoning
    thought_process: str = Field(
        description="Step-by-step reasoning. 1. Identify entities (Person vs Location). 2. Check Schema for valid properties. 3. Define path.")
    source_node_label: str = Field(description="Primary source node label.")
    target_node_label: str = Field(description="Primary target node label.")
    relationship_paths: List[str] = Field(description="Ordered list of specific relationship types from Schema.")
    target_node_variable: str = Field(description="Cypher var for target.")
    source_node_variable: str = Field(description="Cypher var for source.")
    cypher_constraint_fragment: str = Field(description="WHERE clause fragment.")
    cypher_return_fragment: str = Field(description="RETURN statement fragment.")


class GroundedComponent(BaseModel):
    """Final blueprint structure for the Cypher Generator."""
    source_node_label: str
    target_node_label: str
    relationship_paths: List[str]
    cypher_constraint_clause: str
    return_variables: str
    target_node_variable: str
    source_node_variable: str
    filter_on_verbs: List[str] = Field(default_factory=list)


class CorrectionReport(BaseModel):
    """Output schema for the Query Debugger."""
    error_source_agent: str
    correction_needed: str
    fixed_cypher: Optional[str] = Field(None, description="The fully corrected Cypher query string, if applicable.")
    new_blueprint_fragment: Optional[Dict[str, Any]] = Field(None)


class PrunedSchema(BaseModel):
    """
    Expanded schema model that includes valid properties for the selected labels/rels.
    """
    NodeLabels: List[str]
    RelationshipTypes: List[str]
    # NEW: Context for property awareness
    NodeProperties: Dict[str, List[str]] = Field(description="Map of Label -> [valid_property_keys]")
    RelationshipProperties: Dict[str, List[str]] = Field(description="Map of RelType -> [valid_property_keys]")


class SynthesisOutput(BaseModel):
    final_answer: str
    source_document_ids: List[str] = Field(default_factory=list)


class CypherWrapper(BaseModel):
    query: str


# --- SYSTEM PROMPTS ---

SYSTEM_PROMPTS = {
    "Intent Planner": (
        "You are a Cypher query planning expert. Analyze the user's natural language query. "
        "Determine if this is a simple lookup or a 'MultiHop' query.\n"
        "RELATIONSHIPS: Extract the sequence of actions or verbs into 'proposed_relationships' as a list. "
        "Example: 'Who paid Epstein and visited?' -> ['paid', 'visited'].\n"
        "VERB FILTERS: If the user specifically asks for exact phrases (e.g., 'relationships with the verb \"stocks of\"'), "
        "extract those exact strings into 'filter_on_verbs'."
    ),
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
        "1. Analyze Entities: Is 'Island' a Person or a Location? Check the Schema labels.\n"
        "2. Check Properties: Does the `PERSON` label have a 'type' property? If no, do not use `n.type`.\n"
        "3. Define Path: If the destination is a Location, ensure the path includes a node with that Label (e.g., `(p)-[:MOVED]->(l:LOCATION)`).\n\n"
        "TASK: Create a blueprint where 'relationship_paths' uses the EXACT relationship types from the SCHEMA.\n"
        "MULTI-HOP: The 'proposed_relationships' list (e.g., ['paid', 'visited']) must be mapped to the valid schema types provided in the SCHEMA list.\n"
        "PROVENANCE: Return provenance from the RELATIONSHIPS. Use `coalesce(r.source_pks, r.doc_id)` to handle both fields.\n"
        "CONSTRAINT RULE: Do NOT use properties in the WHERE clause that are not listed in the Schema's NodeProperties."
    ),
    "Cypher Generator": (
        "You are an expert Cypher Generator. Convert the Grounded Component into a VALID, READ-ONLY Cypher query. "
        "RULES:\n"
        "1. PATHS: Iterate through the 'relationship_paths' list to build the pattern. Assign variables to ALL relationships.\n"
        "   Example 2 steps: (a:Label1)-[r1:REL_TYPE_1]->(b)-[r2:REL_TYPE_2]->(c:Label2).\n"
        "   CRITICAL: Do NOT create self-loops like `(b)--(b)`. Ensure the path is continuous: `(a)-[r1]->(b)-[r2]->(c)`.\n"
        "2. PROPERTIES: Only use properties explicitly listed in the Schema. Do NOT invent properties like `.type`, `.category`, etc.\n"
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
        "If the result is empty, state clearly that no information was found in the graph."
    )
}


# --- STANDALONE UTILITIES ---

def fetch_schema_statistics(uri: str, auth: tuple) -> Dict[str, Any]:
    """
    Connects to DB and creates comprehensive schema stats including:
    - Node Labels & Counts
    - Relationship Types & Raw Verb lists
    - PROPERTY KEYS for every Label and Relationship Type (New in V3)
    """
    stats = {
        "status": "INIT",
        "error": None,
        "NodeCounts": {},
        "RelationshipVerbs": {},
        "NodeLabels": [],
        "RelationshipTypes": [],
        "NodeProperties": {},  # New
        "RelationshipProperties": {}  # New
    }

    driver = None
    try:
        # Standard URI normalization
        cleaned_uri = uri.strip()
        if not any(cleaned_uri.startswith(p) for p in ["neo4j+s://", "bolt://", "neo4j://"]):
            cleaned_uri = f"neo4j+s://{cleaned_uri}"

        print(f"DEBUG: Attempting to connect to Neo4j at: {cleaned_uri}")

        driver = GraphDatabase.driver(cleaned_uri, auth=auth)
        driver.verify_connectivity()

        with driver.session() as session:
            # 1. Basic Lists
            stats["NodeLabels"] = session.run("CALL db.labels()").value()
            stats["RelationshipTypes"] = session.run("CALL db.relationshipTypes()").value()

            # 2. Node Counts
            for label in stats["NodeLabels"]:
                count_q = f"MATCH (n:`{label}`) RETURN count(n) as c"
                count = session.run(count_q).single()["c"]
                stats["NodeCounts"][label] = count

            # 3. Relationship Raw Verbs
            for r_type in stats["RelationshipTypes"]:
                verb_q = (
                    f"MATCH ()-[r:`{r_type}`]->() "
                    f"WHERE r.raw_verbs IS NOT NULL "
                    f"UNWIND r.raw_verbs as v "
                    f"RETURN DISTINCT v"
                )
                verbs = [record["v"] for record in session.run(verb_q)]
                stats["RelationshipVerbs"][r_type] = verbs

            # 4. Detailed Property Schema (Node Types)
            # db.schema.nodeTypeProperties() returns properties for label combinations
            # We aggregate them by individual label for simplicity
            node_props_q = """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName
            RETURN nodeType, collect(propertyName) as props
            """
            for record in session.run(node_props_q):
                # nodeType is usually ":Label" or ":Label1:Label2"
                # We strip the leading colon and split if compound
                raw_type = record["nodeType"]
                props = record["props"]
                if raw_type.startswith(":"):
                    labels = raw_type[1:].split(":")
                    for label in labels:
                        # Merge properties if label appears in multiple combinations
                        current_props = set(stats["NodeProperties"].get(label, []))
                        current_props.update(props)
                        stats["NodeProperties"][label] = list(current_props)

            # 5. Detailed Property Schema (Relationship Types)
            rel_props_q = """
            CALL db.schema.relTypeProperties()
            YIELD relType, propertyName
            RETURN relType, collect(propertyName) as props
            """
            for record in session.run(rel_props_q):
                # relType is ":TYPE"
                raw_type = record["relType"]
                props = record["props"]
                if raw_type.startswith(":"):
                    r_type = raw_type[1:]
                    stats["RelationshipProperties"][r_type] = props

            stats["status"] = "SUCCESS"

    except Exception as e:
        stats["status"] = "ERROR"
        stats["error"] = str(e)
    finally:
        if driver:
            driver.close()

    return stats


# --- PIPELINE CLASS ---

class GraphRAGPipeline:
    """
    Graph RAG Pipeline V3
    Includes robust schema property awareness to prevent hallucinations.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str, llm_api_key: str,
        schema_stats: Dict[str, Any] = None):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_pass = neo4j_pass
        self.llm_api_key = llm_api_key

        self.schema_stats = schema_stats or {}

        self.driver = None
        self.llm = None

        self._init_llm()
        self._init_db()

    def _init_llm(self):
        if not self.llm_api_key:
            raise ValueError("LLM API Key is missing.")
        try:
            self.llm = ChatMistralAI(
                model=LLM_MODEL,
                api_key=self.llm_api_key,
                temperature=0.0
            )
        except Exception as e:
            print(f"[WARNING] LLM Init failed: {e}")

    def _init_db(self):
        try:
            uri = self.neo4j_uri.strip()
            if not any(uri.startswith(p) for p in ["neo4j+s://", "bolt://", "neo4j://"]):
                uri = f"neo4j+s://{uri}"

            print(f"DEBUG: Pipeline connecting to Neo4j at: {uri}")
            self.driver = GraphDatabase.driver(uri, auth=(self.neo4j_user, self.neo4j_pass))
            self.driver.verify_connectivity()
            print(f"[SUCCESS] Connected to Neo4j at {uri}")
        except Exception as e:
            print(f"[FATAL] Neo4j Connection Failed: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    # --- HELPER METHODS ---

    def _get_full_schema(self) -> Dict[str, Any]:
        """
        Retrieves schema context.
        V3 Update: Includes Property Maps to prevent hallucinations.
        """
        schema = {
            "NodeLabels": [],
            "RelationshipTypes": [],
            "NodeProperties": {},
            "RelationshipProperties": {}
        }

        if self.schema_stats and "NodeLabels" in self.schema_stats:
            schema["NodeLabels"] = self.schema_stats.get("NodeLabels", [])
            schema["RelationshipTypes"] = self.schema_stats.get("RelationshipTypes", [])
            # NEW: Pass property definitions
            schema["NodeProperties"] = self.schema_stats.get("NodeProperties", {})
            schema["RelationshipProperties"] = self.schema_stats.get("RelationshipProperties", {})
            return schema

        # Fallback (Basic)
        if not self.driver:
            raise ConnectionError("Database driver is not connected.")
        return schema  # Return empty if no cached stats (simpler than re-implementing live fetch here)

    def _extract_proof_ids(self, results: List[Dict]) -> List[str]:
        proof_ids = set()
        if not results:
            return []

        for row in results:
            for key, val in row.items():
                if key == "doc_id" and val:
                    proof_ids.add(str(val))
                if isinstance(val, dict):
                    if "doc_id" in val: proof_ids.add(str(val["doc_id"]))
                    if "source_pks" in val:
                        pks = val.get("source_pks")
                        if isinstance(pks, list):
                            for pk in pks: proof_ids.add(str(pk))
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item.isdigit():
                            pass
                        if isinstance(item, dict):
                            if "doc_id" in item: proof_ids.add(str(item["doc_id"]))
                if "source_pks" in key and isinstance(val, list):
                    for pk in val:
                        proof_ids.add(str(pk))
        return list(proof_ids)

    def _check_query_safety(self, cypher: str) -> bool:
        if SAFETY_REGEX.search(cypher):
            return False
        return True

    def _run_agent(self, agent_name: str, output_model: BaseModel, context: Dict) -> Any:
        if not self.llm:
            raise ConnectionError("LLM client not available.")

        sys_prompt = SYSTEM_PROMPTS.get(agent_name, "")

        formatted_context = {}
        for k, v in context.items():
            if isinstance(v, BaseModel):
                formatted_context[k] = v.model_dump_json(indent=2)
            elif isinstance(v, dict) or isinstance(v, list):
                formatted_context[k] = json.dumps(v, indent=2)
            else:
                formatted_context[k] = str(v)

        # Standard Prompt Construction
        if agent_name == "Intent Planner":
            msgs = [("system", sys_prompt), ("human", "QUERY: {user_query}")]
        elif agent_name == "Schema Selector":
            msgs = [("system", sys_prompt), ("human", "BLUEPRINT: {blueprint}\nFULL_SCHEMA: {full_schema}")]
        elif agent_name == "Grounding Agent":
            msgs = [("system", sys_prompt), ("human", "BLUEPRINT: {blueprint}\nSCHEMA: {schema}\nQUERY: {user_query}")]
        elif agent_name == "Cypher Generator":
            msgs = [("system", sys_prompt), ("human", "GROUNDED_COMPONENT: {blueprint}")]
        elif agent_name == "Query Debugger":
            msgs = [("system", sys_prompt),
                    ("human", "ERROR: {error}\nQUERY: {failed_query}\nBLUEPRINT: {blueprint}\nWARNINGS: {warnings}")]
        elif agent_name == "Synthesizer":
            msgs = [("system", sys_prompt),
                    ("human", "QUERY: {user_query}\nCYPHER: {final_cypher}\nRESULTS: {db_result}")]
        else:
            raise ValueError(f"Unknown Agent: {agent_name}")

        prompt = ChatPromptTemplate.from_messages(msgs)
        structured_llm = self.llm.with_structured_output(output_model)
        chain = prompt | structured_llm

        try:
            return chain.invoke(formatted_context)
        except Exception as e:
            print(f"[LLM ERROR] {agent_name}: {e}")
            raise e

    # --- MAIN EXECUTION METHOD ---

    def run(self, user_query: str) -> Dict[str, Any]:
        pipeline_object = {
            "user_query": user_query,
            "status": "INIT",
            "plan": None,
            "schema_context": None,
            "grounding": None,
            "cypher_query": None,
            "raw_results": [],
            "proof_ids": [],
            "final_answer": None,
            "error": None,
            "execution_history": []
        }

        if not self.driver:
            pipeline_object["status"] = "ERROR_DB_CONNECTION"
            pipeline_object["error"] = "Database driver is not initialized."
            return pipeline_object

        try:
            self.driver.verify_connectivity()
        except Exception as e:
            pipeline_object["status"] = "ERROR_DB_UNREACHABLE"
            pipeline_object["error"] = str(e)
            return pipeline_object

        try:
            # 1. Intent Planning
            pipeline_object["status"] = "PLANNING"
            blueprint = self._run_agent(
                "Intent Planner",
                StructuredBlueprint,
                {"user_query": user_query}
            )
            pipeline_object["plan"] = blueprint.model_dump()

            # 2. Schema Selection (Includes Property Pruning)
            full_schema = self._get_full_schema()
            pruned_schema = self._run_agent(
                "Schema Selector",
                PrunedSchema,
                {"blueprint": blueprint, "full_schema": full_schema}
            )
            pipeline_object["schema_context"] = pruned_schema.model_dump()

            # 3. Grounding
            grounding = self._run_agent(
                "Grounding Agent",
                GroundingOutput,
                {"blueprint": blueprint, "schema": pipeline_object["schema_context"], "user_query": user_query}
            )

            grounded_component = GroundedComponent(
                source_node_label=grounding.source_node_label,
                target_node_label=grounding.target_node_label,
                relationship_paths=grounding.relationship_paths,
                cypher_constraint_clause=grounding.cypher_constraint_fragment,
                return_variables=grounding.cypher_return_fragment,
                target_node_variable=grounding.target_node_variable,
                source_node_variable=grounding.source_node_variable,
                filter_on_verbs=blueprint.filter_on_verbs
            )
            pipeline_object["grounding"] = grounded_component.model_dump()

            # 4. Cypher Generation
            pipeline_object["status"] = "GENERATING"
            cypher_resp = self._run_agent(
                "Cypher Generator",
                CypherWrapper,
                {"blueprint": grounded_component}
            )
            raw_cypher = cypher_resp.query.replace("\n", " ")

            if not self._check_query_safety(raw_cypher):
                raise ValueError("Generated query contains unsafe WRITE operations.")

            pipeline_object["cypher_query"] = raw_cypher

            # 5. Execution Loop
            pipeline_object["status"] = "EXECUTING"
            max_retries = 3
            current_cypher = raw_cypher
            execution_history = []
            results = []

            for attempt in range(1, max_retries + 1):
                attempt_log = {
                    "attempt": attempt,
                    "cypher": current_cypher,
                    "status": "PENDING",
                    "error": None,
                    "warnings": []
                }

                try:
                    with self.driver.session() as session:
                        result_obj = session.run(current_cypher)
                        records = [record.data() for record in result_obj]
                        summary = result_obj.consume()

                        # Fix for DeprecationWarning
                        current_warnings = []
                        if hasattr(summary, "gql_status_objects"):
                            current_warnings = [{"code": s.gql_status, "message": s.status_description} for s in
                                                summary.gql_status_objects]
                        elif summary.notifications:
                            current_warnings = [{"code": n.code, "message": n.description} for n in
                                                summary.notifications]

                        attempt_log["warnings"] = current_warnings

                        # Fail on schema mismatches if no results
                        has_critical_warning = attempt_log["warnings"] and any(
                            "Property key does not exist" in w["message"] for w in attempt_log["warnings"])
                        if not records and has_critical_warning:
                            raise neo4j_exceptions.ClientError(
                                f"Query returned no results due to schema warnings: {attempt_log['warnings']}")

                        results = records
                        attempt_log["status"] = "SUCCESS"
                        execution_history.append(attempt_log)
                        pipeline_object["cypher_query"] = current_cypher
                        break

                except Exception as e:
                    error_msg = str(e)
                    attempt_log["status"] = "FAILED"
                    attempt_log["error"] = error_msg
                    execution_history.append(attempt_log)

                    print(f"[EXEC ERROR] Attempt {attempt} failed: {error_msg}. Debugging...")

                    if attempt < max_retries:
                        correction = self._run_agent(
                            "Query Debugger",
                            CorrectionReport,
                            {
                                "error": error_msg,
                                "failed_query": current_cypher,
                                "blueprint": grounded_component,
                                "schema": full_schema,
                                "warnings": attempt_log.get("warnings")
                            }
                        )

                        if correction.fixed_cypher:
                            current_cypher = correction.fixed_cypher.replace("\n", " ")
                            print(f"[DEBUGGER] Applying fix: {current_cypher}")
                        else:
                            break
                    else:
                        pipeline_object["execution_history"] = execution_history
                        pipeline_object["status"] = "ERROR_MAX_RETRIES_EXCEEDED"
                        pipeline_object["error"] = f"Final error after {max_retries} attempts: {error_msg}"
                        return pipeline_object

            pipeline_object["execution_history"] = execution_history
            pipeline_object["raw_results"] = results

            # 6. Post-Processing
            proofs = self._extract_proof_ids(results)
            pipeline_object["proof_ids"] = proofs

            pipeline_object["status"] = "SYNTHESIZING"
            if not results:
                pipeline_object["final_answer"] = "No direct information found matching this query in the database."
                pipeline_object["status"] = "COMPLETED_NO_RESULTS"
            else:
                synthesis = self._run_agent(
                    "Synthesizer",
                    SynthesisOutput,
                    {
                        "user_query": user_query,
                        "final_cypher": current_cypher,
                        "db_result": results[:50]
                    }
                )
                pipeline_object["final_answer"] = synthesis.final_answer
                pipeline_object["status"] = "SUCCESS"

        except Exception as e:
            pipeline_object["status"] = "ERROR_INTERNAL"
            pipeline_object["error"] = str(e)

        return pipeline_object







# --- STARTUP: DOWNLOAD DATA FROM GITHUB ---
@st.cache_data
def load_github_data():
    # Constructing raw URL for the CSV
    url = "https://raw.githubusercontent.com/ssopic/some_data/main/sum_data.csv"
    try:
        df = pd.read_csv(url)
        # Ensure PK is string for consistent matching
        df['PK'] = df['PK'].astype(str)
        return df
    except Exception as e:
        st.error(f"Failed to download data from GitHub: {e}")
        return pd.DataFrame(columns=['PK', 'details'])


# Load data into session once
if 'github_data' not in st.session_state:
    st.session_state.github_data = load_github_data()

# --- INITIALIZE SESSION STATE ---
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "connected": False,
        "mistral_key": "",
        "neo4j_creds": {"uri": "", "user": "neo4j", "pass": ""},
        "schema_stats": {},
        "evidence_locker": [],
        "selected_ids": set(),
        "chat_history": []
    }


# --- HELPER FUNCTIONS ---
def is_safe_cypher(query: str) -> bool:
    """Regex check to block modifying commands."""
    forbidden = r"(?i)\b(CREATE|DELETE|DETACH|SET|REMOVE|DROP|MERGE|CALL|SET)\b"
    return not bool(re.search(forbidden, query))


def extract_provenance_from_result(result: Dict) -> List[str]:
    """Scans JSON recursively for any keys containing 'provenance'."""
    ids = []

    def recurse(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if "provenance" in k.lower():
                    if isinstance(v, list):
                        ids.extend([str(i) for i in v])
                    else:
                        ids.append(str(v))
                else:
                    recurse(v)
        elif isinstance(data, list):
            for item in data: recurse(item)

    recurse(result)
    return list(set(ids))


# --- SCREEN 1: CONNECTION ---
def screen_connection():
    st.title("🔗 Connection Gatekeeper")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("1. Authentication Settings")
        m_key = st.text_input("Mistral API Key", type="password", key="m_key_conn")
        n_uri = st.text_input("Neo4j URI", type="password", key="n_uri_conn")
        n_user = st.text_input("Neo4j User", value="neo4j", key="n_user_conn")
        n_pass = st.text_input("Neo4j Password", type="password", key="n_pass_conn")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Submit Own Codes", use_container_width=True):
                auth = (n_user, n_pass)
                with st.spinner("Connecting and Fetching Schema..."):
                    stats = fetch_schema_statistics(n_uri, auth)
                    if stats.get("status") == "SUCCESS":
                        st.session_state.app_state.update({
                            "connected": True,
                            "mistral_key": m_key,
                            "neo4j_creds": {"uri": n_uri, "user": n_user, "pass": n_pass, "auth": auth},
                            "schema_stats": stats
                        })
                        st.rerun()
                    else:
                        st.error(f"Connection Failed: {stats.get('error')}")

        with col2:
            if st.button("📋 Use Defaults", use_container_width=True):
                d_uri = os.getenv("NEO4J_URI", "bolt://")
                d_user = os.getenv("NEO4J_USER", "neo4j")
                d_pass = os.getenv("NEO4J_PASSWORD", "")
                d_key = os.getenv("MISTRAL_API_KEY", "")
                auth = (d_user, d_pass)
                with st.spinner("Connecting with defaults..."):
                    stats = fetch_schema_statistics(d_uri, auth)
                    if stats.get("status") == "SUCCESS":
                        st.session_state.app_state.update({
                            "connected": True,
                            "mistral_key": d_key,
                            "neo4j_creds": {"uri": d_uri, "user": d_user, "pass": d_pass, "auth": auth},
                            "schema_stats": stats
                        })
                        st.rerun()


# --- SCREEN 2: DATABOOK ---
def screen_databook():
    st.title("📚 The Databook")
    stats = st.session_state.app_state["schema_stats"]

    search = st.text_input("Fuzzy Filter Labels", placeholder="Search schema (e.g. 'tramp' -> 'trump')...")

    col_n, col_r = st.columns(2)
    with col_n:
        st.subheader("Node Labels")
        labels = stats.get("NodeLabels", [])
        if search:
            labels = difflib.get_close_matches(search, labels, n=10, cutoff=0.3)
        for label in labels:
            with st.expander(f"Label: {label} ({stats['NodeCounts'].get(label, 0)})"):
                st.write("**Properties:**", stats["NodeProperties"].get(label, []))
                if st.button(f"Preview {label}", key=f"btn_p_{label}"):
                    creds = st.session_state.app_state["neo4j_creds"]
                    with GraphDatabase.driver(creds["uri"], auth=creds["auth"]) as d:
                        with d.session() as s:
                            res = s.run(f"MATCH (n:`{label}`) RETURN n LIMIT 5")
                            st.json([dict(r["n"]) for r in res])

    with col_r:
        st.subheader("Relationship Types")
        rels = stats.get("RelationshipTypes", [])
        if search:
            rels = difflib.get_close_matches(search, rels, n=10, cutoff=0.3)
        for r_type in rels:
            with st.expander(f"Type: {r_type}"):
                st.write("**Verbs:**", stats["RelationshipVerbs"].get(r_type, []))
                st.write("**Properties:**", stats["RelationshipProperties"].get(r_type, []))
                if st.button(f"Preview {r_type}", key=f"btn_r_{r_type}"):
                    creds = st.session_state.app_state["neo4j_creds"]
                    with GraphDatabase.driver(creds["uri"], auth=creds["auth"]) as d:
                        with d.session() as s:
                            res = s.run(f"MATCH ()-[r:`{r_type}`]->() RETURN r LIMIT 5")
                            st.json([dict(r["r"]) for r in res])


# --- SCREEN 3: EXTRACTION ---
def screen_extraction():
    st.title("🔍 Extraction & Search")

    left, right = st.columns(2)
    with left:
        st.subheader("⚡ Direct Cypher Query")
        c_input = st.text_area("Read-Only Cypher Console", height=200)
        if st.button("Execute Query"):
            if not is_safe_cypher(c_input):
                st.warning("⚠️ Modification commands detected! Query blocked.")
            else:
                try:
                    creds = st.session_state.app_state["neo4j_creds"]
                    with GraphDatabase.driver(creds["uri"], auth=creds["auth"]) as d:
                        with d.session() as s:
                            res = s.run(c_input)
                            st.dataframe([dict(r) for r in res])
                except Exception as e:
                    st.error(str(e))

    with right:
        st.subheader("🤖 Agent Chat")
        for chat in st.session_state.app_state["chat_history"]:
            with st.chat_message(chat["role"]): st.write(chat["content"])

    user_msg = st.chat_input("Ask Mistral about the graph...")
    if user_msg:
        st.session_state.app_state["chat_history"].append({"role": "user", "content": user_msg})
        with st.spinner("AI thinking..."):
            creds = st.session_state.app_state["neo4j_creds"]
            m_key = st.session_state.app_state["mistral_key"]
            pipeline = GraphRAGPipeline(creds["uri"], creds["user"], creds["pass"], m_key,
                                        schema_stats=st.session_state.app_state["schema_stats"])
            result = pipeline.run(user_msg)

            p_ids = extract_provenance_from_result(result)
            if p_ids:
                st.session_state.app_state["evidence_locker"].append({
                    "query": user_msg,
                    "cypher": result.get("cypher_query", "N/A"),
                    "ids": p_ids,
                    "answer": result.get("final_answer", "")
                })
                st.toast(f"Found {len(p_ids)} provenance items!")

            st.session_state.app_state["chat_history"].append(
                {"role": "assistant", "content": result.get("final_answer", "")})
            pipeline.close()
            st.rerun()


# --- SCREEN 4: LOCKER ---
def screen_locker():
    st.title("🗄️ Evidence Locker")
    locker = st.session_state.app_state["evidence_locker"]

    if not locker:
        st.info("Locker empty. Run searches to find items.")
        return

    for i, entry in enumerate(locker):
        with st.container(border=True):
            c1, c2 = st.columns([0.1, 0.9])
            with c1:
                sel = all(pid in st.session_state.app_state["selected_ids"] for pid in entry["ids"])
                if st.checkbox("Select", value=sel, key=f"sel_{i}"):
                    for pid in entry["ids"]: st.session_state.app_state["selected_ids"].add(pid)
                else:
                    for pid in entry["ids"]: st.session_state.app_state["selected_ids"].discard(pid)
            with c2:
                st.write(f"**Query:** {entry['query']}")
                st.caption(f"Cypher: `{entry['cypher']}`")
                st.write(f"**IDs Found:** {', '.join(entry['ids'])}")


# --- SCREEN 5: ANALYSIS ---
def screen_analysis():
    st.title("🔬 Analysis Pane")
    ids = list(st.session_state.app_state["selected_ids"])

    if not ids:
        st.warning("No documents selected in the Locker.")
        return

    # Join with the GitHub Dataset
    df = st.session_state.github_data
    matched_docs = df[df['PK'].isin(ids)]

    st.subheader(f"Analyzing {len(matched_docs)} Matched Documents")

    if not matched_docs.empty:
        # Construct the context for the LLM
        context_text = ""
        for _, row in matched_docs.iterrows():
            context_text += f"ID: {row['PK']}\nContent: {row['email_content']}\n{'-' * 20}\n"

        with st.expander("View Raw Evidence Details"):
            st.dataframe(matched_docs[['PK', 'email_content']])
            st.text_area("Full Context (for LLM)", context_text, height=300)

        st.divider()
        st.subheader("Synthesis Chat")
        q = st.text_input("Ask about this evidence:")
        if q:
            llm = ChatMistralAI(api_key=st.session_state.app_state["mistral_key"], model="mistral-large-latest")
            resp = llm.invoke(f"Context from Evidence:\n{context_text}\n\nQuestion: {q}")
            st.chat_message("assistant").write(resp.content)
    else:
        st.error("None of the selected IDs match the PKs in the GitHub dataset.")


# --- NAVIGATION ---
if not st.session_state.app_state["connected"]:
    screen_connection()
else:
    nav = st.sidebar.radio("Navigation", ["Databook", "Search", "Locker", "Analysis"])
    if nav == "Databook":
        screen_databook()
    elif nav == "Search":
        screen_extraction()
    elif nav == "Locker":
        screen_locker()
    elif nav == "Analysis":
        screen_analysis()

    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        st.session_state.app_state["connected"] = False
        st.rerun()
