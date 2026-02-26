import streamlit as st
import os
import json
import re
import pandas as pd
import difflib
from neo4j import GraphDatabase, Driver, exceptions as neo4j_exceptions
import uuid
from collections import defaultdict
import concurrent.futures
import math
import io

# --- LangChain/Mistral/LLM Imports ---
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from langchain_core.output_parsers import PydanticOutputParser


# ---Visualization and url parsing ---
import plotly.express as px
import urllib.parse

# ---Imports for QR code reading and generation ---
import base64, bz2, io, qrcode, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import zipfile

import asyncio
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

# ==========================================
### 1. CONSTANTS and stuff that breaks if I dont put it earlier ###
# ==========================================

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Graph Analyst", layout="wide")

# --- OTHER CONSTANTS ---
LLM_MODEL = "mistral-medium"
SAFETY_REGEX = re.compile(r"(?i)\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|INSERT|ALTER|GRANT|REVOKE)\b")

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
    "USAGE": "Refers to the act of employing, leveraging, or repurposing resources, information, or assets for a specific purpose or goal. This includes the strategic application of media, data, or personal connections to achieve an outcome, as well as the adaptation of content, platforms, or networks for new or expanded uses. The language used is functional and outcome-oriented, emphasizing the practical or tactical deployment of available tools or opportunities.",
    "MENTIONED_IN": "In which document is the entity mentioned. CTRL+F"
}
# ==========================================
### 2. STATE MANAGEMENT AND UTILITIES ###
# ==========================================
# --- SHARED ---

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
    if 'github_data' not in st.session_state:
        st.session_state.github_data = load_github_data()

    if "app_state" not in st.session_state:
        st.session_state.app_state = {
            "connected": False, "mistral_key": "", "neo4j_creds": {}, 
            "schema_stats": {}, "evidence_locker": [], "selected_ids": set(), "chat_history": []
        }


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
    
def get_config(key, default=""):
    """Helper to get credentials from Secrets (Cloud) or Env (Local)."""
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)
    
def set_page(page_name):
    """Helper to update the current page in session state."""
    st.session_state.current_page = page_name

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

def get_selected_cypher_queries():
    """Helper to extract cypher queries from the currently selected locker items."""
    if "app_state" not in st.session_state or "evidence_locker" not in st.session_state.app_state:
        return []
    selected = st.session_state.app_state.get("selected_ids", set())
    queries = []
    for entry in st.session_state.app_state["evidence_locker"]:
        entry_ids = {str(pid) for pid in entry["ids"]}
        if entry_ids and entry_ids.issubset(selected):
            if "cypher" in entry and entry["cypher"]:
                queries.append(entry["cypher"])
    return queries
    
# --- DESKTOP ONLY ---
def flatten_ids(container):
    """Recursively flattens a container of IDs (strings/ints/nested lists) into a set."""
    ids = set()
    if isinstance(container, (list, tuple, set)):
        for item in container:
            ids.update(flatten_ids(item))
    elif pd.notna(container):
        ids.add(container)
    return ids

def get_rel_definition(rel_name):
    """Helper to safely get a definition or a default prompt."""
    return RELATIONSHIP_DEFINITIONS.get(rel_name, "Relationship connection between entities.")
    
# ==========================================
### 2. DATA ACCESS LAYER ###
# ==========================================
# --- SHARED ---

#CACHED RESOURCES
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
    This is the actual dataframe containing the open data from the Oversight commitee
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

# USED BY THE GRAPH RAG PIPELINE
def fetch_schema_statistics(uri: str, auth: tuple) -> Dict[str, Any]:
    """
    Connects to DB and creates comprehensive schema stats required for context management of the graph rag pipeline.
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
                    verb_q = f"MATCH ()-[r:`{r_type}`]->() WHERE r.raw_verbs IS NOT NULL UNWIND r.raw_verbs as v RETURN DISTINCT v "
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

# --- DESKTOP ONLY ---
# These contain exclusively the fetchers for the manual data extraction screen(with the sunbursts)

#The get_db_driver function actually needs to be refactored out but that is in a future sprint
def get_db_driver():
    """Retrieves the cached driver from the main app state."""
    if "app_state" in st.session_state and "neo4j_creds" in st.session_state.app_state:
        creds = st.session_state.app_state["neo4j_creds"]
        uri = creds.get("uri")
        auth = creds.get("auth")
        if uri and auth:
            return GraphDatabase.driver(uri, auth=auth)
    return None

#FETCH INVENTORY CURRENTLY DEFAULTS TO FETCH INVENTORY FROM DB BECAUSE THE EXACT JSONS ARE STILL BEING WORKED ON. THEY WILL BE COMPLETED ONLY AFTER THE INITIAL USER SCREENING
# THE IDEA IS TO PREPOPULATE THE QUERIES TO LESSEN THE STRAIN ON THE DATABASE
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
        
def fetch_inventory_from_db():
    """
    Refactored Fallback: Generates the inventory dict with strict segregation.
    Uses pattern matching and sizing for robust filtering of Semantic vs Lexical nodes.
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
                # Matches only nodes that have at least one relationship that is NOT 'MENTIONED_IN'
                q_entities = f"""
                MATCH (n:`{label}`)-[r]->()
                WHERE n.name IS NOT NULL 
                  AND type(r) <> 'MENTIONED_IN'
                RETURN DISTINCT n.name as name
                """
                names_entities = [r["name"] for r in session.run(q_entities)]
                if names_entities:
                    inventory["Entities"][label] = sorted(names_entities)

                # --- B. TEXT MENTIONS (Purely Lexical) ---
                q_lexical = f"""
                MATCH (n:`{label}`)-[:MENTIONED_IN]->()
                WHERE n.name IS NOT NULL
                RETURN DISTINCT n.name as name
                """
                names_lexical = [r["name"] for r in session.run(q_lexical)]
                if names_lexical:
                    inventory["Text Mentions"][label] = sorted(names_lexical)

            # 2. POPULATE VERBS (Relationships - Semantic Only)
            rels_result = session.run("CALL db.relationshipTypes()")
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
            if selector_type == "Connections":
                query = """
                MATCH (n)-[r]->(m)
                WHERE type(r) IN $names 
                  AND type(r) <> 'MENTIONED_IN'
                RETURN 
                    type(r) as edge, 
                    labels(n)[0] as source_node_label, 
                    labels(m)[0] as connected_node_label, 
                    count(*) as count,
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list"""
                result = session.run(query, names=names)
                data = [r.data() for r in result]
                return pd.DataFrame(data)

            elif selector_type == "Text Mentions":
                query = f"""
                MATCH (n:`{label}`)-[r:MENTIONED_IN]->(m:Document)
                WHERE n.name IN $names
                RETURN 
                    type(r) as edge,
                    labels(n)[0] as node, 
                    coalesce(n.name, 'Unknown') as node_name, 
                    count(m) as count,
                    'Document' as connected_node_label, 
                    collect(coalesce(r.source_pks, m.doc_id)) as id_list
                """
                result = session.run(query, names=names)
                data = [r.data() for r in result]
                return pd.DataFrame(data)
            
            elif selector_type == "Entities":
                query = f"""
                MATCH (n:`{label}`)-[r]->(m)
                WHERE n.name IN $names
                    AND type(r) <> 'MENTIONED_IN'
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
                
            else:
                return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Live Query failed: {e}")
        return pd.DataFrame()
    finally:
        driver.close()


# The same as with the fetch_inventory function. The actual jsons are not created yet due to often changes in the data. Will be completed after initial user analysis.
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
### 4. GRAPH RAG PIPELINE  ###
# ==========================================

# --- PYDANTIC MODELS ---
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

# --- GRAPH-RAG PIPELINE ---
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

# ==========================================
### 5. MAP REDUCE ENGINE  ###
# ==========================================

# --- PYDANTIC MODELS ---
class ExtractionGoal(BaseModel):
    goal_description: str = Field(description="Specific instructions on what information to extract.")
    keywords: List[str] = Field(description="List of specific keywords or entities to look for.")

class ChunkFact(BaseModel):
    """Output of Pass 1 (Parallel Chunk Processing)"""
    is_relevant: bool = Field(description="True if this text chunk contains info relevant to the goal.")
    fact_summary: str = Field(description="Self-contained summary of the fact. If the sentence cuts off, provide what you have.")
    exact_quote: str = Field(description="Verbatim extraction.")
    boundary_cut_off: bool = Field(
        default=False,
        description="CRITICAL: Set to True ONLY if the thought abruptly ends due to the document hitting the character limit."
    )

class ChunkExtractionResult(BaseModel):
    """Wrapper for Pass 1 results"""
    facts: List[ChunkFact]

class UnifiedDocumentFact(BaseModel):
    """Output of Pass 2 (Document-Level Deduplication & Synthesis)"""
    merged_summary: str = Field(
        description="Synthesized fact resolving overlaps. Completes any thoughts flagged with boundary_cut_off."
    )
    supporting_quotes: List[str] = Field(description="All exact quotes supporting this merged fact.")
    unresolved_fragment: bool = Field(
        default=False,
        description="COMPLIANCE FLAG: Set to True if Pass 2 could not locate the missing context for a chunk flagged with boundary_cut_off (e.g., OCR failure)."
    )

class DocumentSynthesis(BaseModel):
    """The final clean object passed to the Global Reducer"""
    internal_doc_id: str = Field(description="The masked internal ID.")
    verified_facts: List[UnifiedDocumentFact]
    requires_human_review: bool = Field(
        default=False,
        description="CRITICAL: Set to True if ANY verified fact contains an unresolved_fragment."
    )


# --- ENGINE CLASS CONFIGURATION ---
TOKEN_THRESHOLD_CHARS = 40000 
MAX_WORKERS = 4 
# Batch Size Strategy:
# 15,000 chars is roughly 3,500 - 4,000 tokens. 
BATCH_SIZE_CHARS = 15000 
# New: Overlap for splitting large docs (Split Brain Fix)
OVERLAP_CHARS = 1000 

# --- THE ENGINE CLASS ---

class MapReduceEngine:
    TOKEN_THRESHOLD_CHARS = 40000 
    CHUNK_SIZE_CHARS = 15000 
    OVERLAP_CHARS = 1000 

    ARCHITECT_PROMPT = """You are an Expert Legal Analyst. 
    User Query: {query}
    Create a specific 'Extraction Goal' for junior analysts to follow.
    {format_instructions}"""

    PASS1_MAP_PROMPT = """You are a Fact Extraction Agent analyzing a single, isolated text chunk.
    GOAL: {goal_description}
    KEYWORDS: {keywords}
    
    Extract all relevant facts.
    CRITICAL: If a relevant sentence abruptly cuts off at the end of the text, extract it anyway and set `boundary_cut_off` to true.
    
    TEXT CHUNK:
    {content}
    
    {format_instructions}"""

    PASS2_SYNTHESIS_PROMPT = """You are a Document-Level Synthesizer.
    You are receiving raw facts extracted from overlapping chunks of the SAME document.
    
    YOUR TASK:
    1. Merge overlapping/identical facts into a single clear summary.
    2. Resolve fragmented sentences (flagged with boundary_cut_off) by finding the rest of the thought in the other chunks.
    3. If a thought is cut off and the missing context CANNOT be found in the provided facts, set `unresolved_fragment` to true.
    
    RAW EXTRACTED FACTS:
    {raw_facts}
    
    {format_instructions}"""

    PASS3_REDUCE_PROMPT = """You are a Lead Investigator.
    Answer the user's question using ONLY the synthesized facts below.
    Cite your sources using the [Source ID] format. Note any compliance warnings if human review is required due to missing context.
    
    User Question: {query}
    
    Synthesized Evidence:
    {context}
    """

    def __init__(self, api_key: str, model_small: str = "mistral-small", model_large: str = "mistral-medium"):
        self.api_key = api_key
        self.llm_map = ChatMistralAI(model=model_small, api_key=api_key, temperature=0.0)
        self.llm_reduce = ChatMistralAI(model=model_large, api_key=api_key, temperature=0.0)

    def estimate_strategy(self, docs_content: List[str]) -> str:
        total_chars = sum(len(d) for d in docs_content if d)
        if total_chars < self.TOKEN_THRESHOLD_CHARS:
            return "DIRECT"
        return "MAP_REDUCE"

    def architect_query(self, user_query: str) -> ExtractionGoal:
        parser = PydanticOutputParser(pydantic_object=ExtractionGoal)
        prompt = ChatPromptTemplate.from_template(self.ARCHITECT_PROMPT)
        chain = prompt | self.llm_reduce | parser
        try:
            return chain.invoke({"query": user_query, "format_instructions": parser.get_format_instructions()})
        except Exception:
            return ExtractionGoal(goal_description=f"Extract facts relevant to: {user_query}", keywords=[])

    # --- ASYNC WORKERS ---

    async def _amap_chunk(self, chunk_text: str, goal: ExtractionGoal, sem: asyncio.Semaphore) -> List[ChunkFact]:
        """Pass 1: Isolated chunk extraction with concurrency throttling."""
        parser = PydanticOutputParser(pydantic_object=ChunkExtractionResult)
        prompt = ChatPromptTemplate.from_template(self.PASS1_MAP_PROMPT)
        chain = prompt | self.llm_map | parser
        
        async with sem:
            try:
                res = await chain.ainvoke({
                    "goal_description": goal.goal_description,
                    "keywords": ", ".join(goal.keywords),
                    "content": chunk_text,
                    "format_instructions": parser.get_format_instructions()
                })
                return [f for f in res.facts if f.is_relevant]
            except Exception as e:
                print(f"Chunk Map Error: {e}")
                return []

    async def _asynthesize_doc(self, internal_id: str, chunk_facts: List[ChunkFact], sem: asyncio.Semaphore) -> DocumentSynthesis:
        """Pass 2: Synthesize chunks back into a unified document."""
        if not chunk_facts:
            return DocumentSynthesis(internal_doc_id=internal_id, verified_facts=[], requires_human_review=False)
            
        parser = PydanticOutputParser(pydantic_object=DocumentSynthesis)
        prompt = ChatPromptTemplate.from_template(self.PASS2_SYNTHESIS_PROMPT)
        chain = prompt | self.llm_reduce | parser
        
        # Prepare data payload for LLM
        raw_facts_json = [f.model_dump() for f in chunk_facts]
        
        async with sem:
            try:
                res = await chain.ainvoke({
                    "raw_facts": raw_facts_json,
                    "format_instructions": parser.get_format_instructions()
                })
                # Override the LLM's returned ID to strictly enforce Python-level ID safety
                res.internal_doc_id = internal_id 
                
                # Double check compliance flag propagation
                if any(fact.unresolved_fragment for fact in res.verified_facts):
                    res.requires_human_review = True
                    
                return res
            except Exception as e:
                print(f"Document Synthesis Error: {e}")
                return DocumentSynthesis(internal_doc_id=internal_id, verified_facts=[], requires_human_review=True)

    # --- MAIN ASYNC ORCHESTRATOR ---

    async def arun_parallel_map(self, docs: List[dict], goal: ExtractionGoal, status_container=None) -> List[DocumentSynthesis]:
        """The core async engine running Pass 1 and Pass 2 with masking."""
        sem = asyncio.Semaphore(10) # 1. Strict Concurrency Throttling limit
        
        # Phase 1: Pre-Processing (ID Masking & Chunking)
        doc_chunks = {}       # internal_id -> list of text chunks
        id_translation = {}   # internal_id -> real Bates ID
        
        total_chunks = 0
        split_docs = 0

        for i, doc in enumerate(docs):
            real_id = doc.get('Bates_Identity', f'Unknown_{i}')
            internal_id = f"INTERNAL_DOC_{i}"
            id_translation[internal_id] = real_id
            
            body = doc.get('Body', '')
            if len(body) > self.CHUNK_SIZE_CHARS:
                split_docs += 1
                chunks = []
                start = 0
                while start < len(body):
                    end = min(start + self.CHUNK_SIZE_CHARS, len(body))
                    chunks.append(body[start:end])
                    if end == len(body): break
                    start = end - self.OVERLAP_CHARS
                doc_chunks[internal_id] = chunks
            else:
                doc_chunks[internal_id] = [body]
            
            total_chunks += len(doc_chunks[internal_id])

        if status_container:
            status_container.write(f"📦 Masked and chunked {len(docs)} documents into {total_chunks} isolated chunks.")
            if split_docs > 0:
                status_container.write(f"ℹ️ Split {split_docs} large documents to prevent token overflow.")
            status_container.write(f"🚀 Spawning Pass 1 Map Agents (Max 10 concurrent)...")

        # Pass 1: Map all chunks concurrently
        # We process chunk by chunk, tracking which internal_id they belong to via an async wrapper
        chunk_tasks = []
        
        async def map_with_id(i_id: str, text: str):
            """Wrapper to safely pass the internal ID along with the coroutine result."""
            result = await self._amap_chunk(text, goal, sem)
            return i_id, result

        for internal_id, chunks in doc_chunks.items():
            for chunk_text in chunks:
                chunk_tasks.append(map_with_id(internal_id, chunk_text))

        doc_extracted_facts = {i_id: [] for i_id in id_translation.keys()}
        
        completed = 0
        for coro in asyncio.as_completed(chunk_tasks):
            i_id, facts = await coro
            doc_extracted_facts[i_id].extend(facts)
            completed += 1
            if status_container:
                status_container.update(label=f"Pass 1: Processed {completed}/{total_chunks} chunks...", state="running")

        # Pass 2: Document-Level Synthesis for large documents
        if status_container:
            status_container.update(label="Pass 2: Synthesizing split documents...", state="running")

        synthesis_tasks = []
        for internal_id, facts in doc_extracted_facts.items():
            # Run synthesis to merge overlapping chunks, resolve splits, and deduplicate
            task = asyncio.create_task(self._asynthesize_doc(internal_id, facts, sem))
            synthesis_tasks.append(task)

        final_syntheses: List[DocumentSynthesis] = await asyncio.gather(*synthesis_tasks)

        # Phase 3: Post-Processing (Unmasking)
        for synth in final_syntheses:
            synth.internal_doc_id = id_translation[synth.internal_doc_id] # Restore real Bates ID!

        return [s for s in final_syntheses if s.verified_facts]

    def reduce_facts(self, final_docs: List[DocumentSynthesis], user_query: str) -> str:
        """Pass 3: Global Reducer"""
        if not final_docs:
            return "I analyzed the documents but found no relevant specific information matching your criteria."

        context_str = ""
        for doc in final_docs:
            context_str += f"Source ({doc.internal_doc_id}):\n"
            if doc.requires_human_review:
                context_str += "WARNING: Contains unresolved fragments or missing context.\n"
            
            for fact in doc.verified_facts:
                context_str += f"- {fact.merged_summary}\n"
            context_str += "---\n"
            
        prompt = ChatPromptTemplate.from_template(self.PASS3_REDUCE_PROMPT)
        chain = prompt | self.llm_reduce
        return chain.invoke({"query": user_query, "context": context_str}).content



# ==========================================
### 6. QR code generator and reader  ###
# ==========================================


class SocialQRMaster:
    """
    The complete engine for generating Secure, Social-Media-Ready,
    and Verified QR Codes in chunked batches.
    """

    def __init__(self):
        self.unsafe_pattern = re.compile(
            r'\b(CREATE|DELETE|SET|MERGE|REMOVE|DETACH|DROP|LOAD CSV|CALL)\b',
            re.IGNORECASE
        )

    def _validate_safety(self, query):
        if self.unsafe_pattern.search(query):
            raise ValueError(f"SECURITY BLOCK: Write operation detected in query: {query[:30]}...")
        return True

    def _compress_payload(self, queries, instruction=None):
        cleaned_queries = [q.strip() for q in queries]
        queries_joined = "|||".join(cleaned_queries)
        text_part = instruction.strip() if instruction else ""
        full_payload = f"{text_part}||SEP||{queries_joined}"
        compressed = bz2.compress(full_payload.encode('utf-8'), compresslevel=9)
        return base64.b64encode(compressed).decode('utf-8')

    @staticmethod
    def extract_payload(encoded_payload):
        try:
            compressed = base64.b64decode(encoded_payload)
            full_string = bz2.decompress(compressed).decode('utf-8')
            parts = full_string.split("||SEP||", 1)
            if len(parts) != 2:
                return None, full_string.split("|||")
            desc_text, queries_str = parts
            return desc_text if desc_text else None, queries_str.split("|||")
        except Exception:
            return None, []

    def _check_contrast(self, fill_hex, back_hex):
        def get_lum(hex_code):
            rgb = ImageColor.getrgb(hex_code)
            return (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255.0
        if get_lum(fill_hex) > get_lum(back_hex):
            return False, "INVERTED: Dots must be darker than background."
        if (get_lum(back_hex) - get_lum(fill_hex)) < 0.4:
            return False, "LOW CONTRAST: Colors are too similar."
        return True, "OK"

    def _verify_readability(self, pil_img, original_data):
        import cv2
        import numpy as np
        
        def run_test(img_to_test):
            target_short_side = 1500
            w, h = img_to_test.size
            if min(w, h) > target_short_side:
                ratio = target_short_side / min(w, h)
                small_img = img_to_test.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            else:
                small_img = img_to_test

            buffer = io.BytesIO()
            small_img.save(buffer, format="JPEG", quality=70, subsampling=2)
            buffer.seek(0)
            cv_img = cv2.imdecode(np.asarray(bytearray(buffer.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
            detector = cv2.QRCodeDetector()
            
            def test_scan(image_array):
                ret, dec, _, _ = detector.detectAndDecodeMulti(image_array)
                if ret and dec and original_data in dec: return True
                dec_text, _, _ = detector.detectAndDecode(image_array)
                if dec_text == original_data: return True
                return False

            if test_scan(cv_img): return True
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            if test_scan(gray): return True
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            if test_scan(thresh): return True
            return False
            
        return run_test(pil_img)

    def _verify_raw_export(self, pil_img, original_data):
        """Custom verifier for raw exports. Simulates moderate JPEG transfer but skips brutal downscaling."""
        import cv2
        import numpy as np
        import io
        
        target_short_side = 1500
        w, h = pil_img.size
        if min(w, h) > target_short_side:
            ratio = target_short_side / min(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            small_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            small_img = pil_img
        
        buf = io.BytesIO()
        small_img.save(buf, format="JPEG", quality=85) 
        file_bytes = np.asarray(bytearray(buf.getvalue()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        detector = cv2.QRCodeDetector()
        
        def test_scan(image_array):
            ret, dec, _, _ = detector.detectAndDecodeMulti(image_array)
            if ret and dec and original_data in dec: return True
            decoded_text, _, _ = detector.detectAndDecode(image_array)
            if decoded_text == original_data: return True
            return False
            
        if test_scan(cv_img): return True
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        if test_scan(gray): return True
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        if test_scan(thresh): return True
        return False

    def _generate_single_image(self, payload, title, fill_color, back_color, width, height, app_address, logo_path, raw_export):
        """Internal worker that draws and tests a single QR code image."""
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_Q, box_size=1, border=4)
        qr.add_data(payload)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color=fill_color, back_color=back_color).convert('RGB')
        
        total_modules = qr_img.size[0]
        # --- FIX: Reduced the height multiplier from 0.60 to 0.50 to give more breathing room for larger fonts
        qr_display_size = int(min(width * 0.85, height * 0.50))
        pixels_per_module = max(1, qr_display_size // total_modules)
        optimal_display_size = pixels_per_module * total_modules
        qr_resized = qr_img.resize((optimal_display_size, optimal_display_size), Image.Resampling.NEAREST)

        final_img = Image.new('RGB', (width, height), back_color)
        draw = ImageDraw.Draw(final_img)

        avg_dim = (width + height) // 2
        
        def get_font(max_size, is_bold=False):
            # Try multiple cross-platform fonts so Linux/Streamlit servers don't fail
            fonts_to_try = [
                "arialbd.ttf" if is_bold else "arial.ttf",
                "Arial Bold.ttf" if is_bold else "Arial.ttf",
                "DejaVuSans-Bold.ttf" if is_bold else "DejaVuSans.ttf",
                "LiberationSans-Bold.ttf" if is_bold else "LiberationSans-Regular.ttf"
            ]
            for font_name in fonts_to_try:
                try:
                    return ImageFont.truetype(font_name, max_size)
                except IOError:
                    continue
            
            print("WARNING: No scalable fonts found. Falling back to microscopic default font.")
            return ImageFont.load_default()
            
        # --- FIX: Safely increased the font sizes
        title_font = get_font(int(avg_dim * 0.05), False)       # Increased from 0.045
        warning_font = get_font(int(avg_dim * 0.05), True)      # Increased from 0.08
        footer_font = get_font(int(avg_dim * 0.05), False)      # Increased from 0.035

        def draw_centered(text, y, font, fill=None):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            draw.text(((width - w) // 2, y), text, fill=fill or "black", font=font)

        qr_y_pos = (height - optimal_display_size) // 2
        qr_x_pos = (width - optimal_display_size) // 2
        gap = int(min(width, height) * 0.08)

        draw_centered(title, qr_y_pos - gap - int(avg_dim * 0.06), title_font)
        draw_centered("Don't trust me blindly!", qr_y_pos - gap - int(avg_dim * 0.06) - int(avg_dim * 0.10) - (gap//2), warning_font)
        final_img.paste(qr_resized, (qr_x_pos, qr_y_pos))

        if raw_export:
            draw.rectangle([15, 15, width - 15, height - 15], outline="#FF0000", width=15)
            draw_centered("RAW EXPORT", qr_y_pos + optimal_display_size + gap, warning_font, "#FF0000")
            draw_centered("Import directly into the app.", qr_y_pos + optimal_display_size + gap + int(avg_dim*0.12), title_font, "#FF0000")
        else:
            draw_centered("See For Yourself", qr_y_pos + optimal_display_size + gap, warning_font)
            
        draw_centered(app_address, height - gap - int(avg_dim * 0.05), footer_font)

        # Route tests flawlessly
        if raw_export:
            if not self._verify_raw_export(final_img, payload):
                raise ValueError("Raw Quality Check Failed: This chunk could not be verified.")
        else:
            if not self._verify_readability(final_img, payload):
                raise ValueError("Quality Check Failed: This chunk could not be verified by OpenCV.")

        return final_img

    def generate_batch(self, queries, title, instruction=None, fill_color="black", back_color="white",
                       width=1080, height=1080, app_address="www.analyzegraph.com", logo_path=None, raw_export=False):
        """
        Public Entry Point: Returns a LIST of verified PIL Images.
        Automatically chunks large payloads dynamically based on canvas space.
        """
        # --- NEW: Safe Truncation for custom titles ---
        # Ensures that "Title 10/10" does not spill off the edges of the canvas
        if len(title) > 22:
            title = title[:19] + "..."
            
        for q in queries: self._validate_safety(q)
        is_safe, msg = self._check_contrast(fill_color, back_color)
        if not is_safe: raise ValueError(f"Color Error: {msg}")

        full_payload = self._compress_payload(queries, instruction)
        
        # Significantly reduced chunk sizes dynamically to guarantee OpenCV readability!
        qr_display_size = int(min(width * 0.85, height * 0.60))
        
        if raw_export:
            CHUNK_SIZE = 300
        elif qr_display_size >= 800: # Instagram Format
            CHUNK_SIZE = 350
        elif qr_display_size >= 500: # Square Format
            CHUNK_SIZE = 150
        else:                        # X / LinkedIn format (Heavily squished height)
            CHUNK_SIZE = 80 
            
        chunks = [full_payload[i:i+CHUNK_SIZE] for i in range(0, len(full_payload), CHUNK_SIZE)]
        total = len(chunks)
        
        if raw_export:
            width, height = 1500, 1500
            
        generated_images = []
        for i, chunk in enumerate(chunks):
            if total > 1:
                chunk_payload = f"[CHUNK {i+1}/{total}]{chunk}"
                page_title = f"{title} {i+1}/{total}"  # --- UPDATED: New n/N format!
            else:
                chunk_payload = chunk
                page_title = title
                
            img = self._generate_single_image(chunk_payload, page_title, fill_color, back_color, width, height, app_address, logo_path, raw_export)
            generated_images.append(img)
            
        return generated_images
                     

# ==========================================
### 7. SCREENS  ###
# ==========================================

# --- SHARED ---

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
        st.caption("*NOTE: The evidence output might have direction issues. In example if you ask for outgoing communication you might (also) get incoming. Prefer asking for communication in general to obtain both sides and filter later. *")
        st.caption("*Make sure to view the evidence before making conclusions.*")

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
                                "query": user_msg, "answer": ans, "ids": result["proof_ids"], "cypher": result["cypher_query"]
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

        cypher_input = st.text_area("Enter Cypher Query", height=150, value="MATCH (n) RETURN n")
        
        if st.button("Run Query"):
            if SAFETY_REGEX.search(cypher_input):
                st.error("SECURITY ALERT: destructive commands are not allowed.")
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
                        "query": f"Manual Cypher: {st.session_state.manual_query_text}",
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
    locker = st.session_state.app_state.get("evidence_locker", [])
    
    if not locker:
        st.info("Locker is empty.")
        return

    # 1. Initialize current selection session state if needed
    current_selection = set()
    global_selected = st.session_state.app_state.get("selected_ids", set())

    for i, entry in enumerate(locker):
        with st.container(border=True):
            c1, c2 = st.columns([0.15, 0.85])
            
            # Preparation for Checkbox Persistence
            entry_ids_str = {str(pid) for pid in entry["ids"]}
            is_checked_default = entry_ids_str.issubset(global_selected) if entry_ids_str else False

            with c1:
                # The user-facing selection functionality
                is_sel = st.checkbox("Select", key=f"sel_{i}", value=is_checked_default)
                if is_sel:
                    # Keep propagation alive by adding all IDs to the selection set
                    for pid in entry["ids"]:
                        current_selection.add(str(pid))
            
            with c2:
                st.markdown(f"**Query:** {entry['query']}")
                # COSMETIC CHANGE: Show count instead of the full ID list
                st.markdown(f"**Evidence Count:** `{len(entry['ids'])} documents`")
                st.caption(f"Summary: {entry.get('answer', 'No description available.')}")
                # # Cypher showcase here is exclusively for testing purposes. Commented out for the user. 
                st.markdown(f"**Cypher:**  {entry['cypher']}")

    # Commit selection back to the global state for the analyst
    st.session_state.app_state["selected_ids"] = current_selection



def get_selected_cypher_queries():
    """Helper to extract cypher queries from the currently selected locker items."""
    if "app_state" not in st.session_state or "evidence_locker" not in st.session_state.app_state:
        return []
    selected = st.session_state.app_state.get("selected_ids", set())
    queries = []
    for entry in st.session_state.app_state["evidence_locker"]:
        entry_ids = {str(pid) for pid in entry["ids"]}
        if entry_ids and entry_ids.issubset(selected):
            if "cypher" in entry and entry["cypher"]:
                queries.append(entry["cypher"])
    return queries

# --- FIX: Clears the old QR codes when you switch formats ---
def clear_qr_cache():
    st.session_state.pop("generated_qr_batch", None)
    st.session_state.pop("generated_qr_size", None)


@st.fragment
def screen_analysis():
    st.title("Analysis Pane")
    
    # 1. Retrieve State
    ids = list(st.session_state.app_state.get("selected_ids", []))
    if not ids:
        st.warning("No documents selected.")
        return
    
    current_ids_sorted = sorted(ids)
    if st.session_state.get("last_analysis_ids") != current_ids_sorted:
        st.session_state.last_analysis = None
        st.session_state.last_analysis_ids = current_ids_sorted
    
    if 'github_data' not in st.session_state:
        pass 
        
    df = st.session_state.github_data
    matched = df[df['PK'].isin(ids)]
    st.subheader(f"Analyzing {len(matched)} Documents")

    # Document Reader Snippet
    doc_options = matched['Bates_Identity'].tolist()
    selected_bates = st.selectbox("Select Document to Read:", options=doc_options)
    
    if selected_bates:
        view_row = matched[matched['Bates_Identity'] == selected_bates].iloc[0]
        with st.container(border=True):
            st.caption(f"Viewing: {view_row['Bates_Identity']}")
            st.text_area("Body", view_row.get('Body', ''), height=200, disabled=True)

    st.divider()
    st.write("### Agentic Analysis")
    
    auto_q = st.session_state.pop("auto_trigger_question", None)
    q = st.chat_input("Ask about this evidence set:")
    
    if auto_q:
        q = auto_q
        with st.chat_message("user"): 
            st.write(f"*(Auto-Triggered from QR)*: {q}")
    
    if q:
        api_key = st.session_state.app_state.get("mistral_key", "")
        if not api_key:
            st.error("Mistral API Key is missing.")
            return

        engine = MapReduceEngine(api_key=api_key) 
        
        docs_payload = [{"Body": row.get('Body', ''), "Bates_Identity": row.get('Bates_Identity', 'Unknown')} for _, row in matched.iterrows()]

        with st.status("Initializing Agent Swarm...", expanded=True) as status:
            
            strategy = engine.estimate_strategy([d['Body'] for d in docs_payload])
            st.write(f"🧠 Gatekeeper Decision: **{strategy}** Mode")
            
            if strategy == "DIRECT":
                status.update(label="Running Direct Analysis...", state="running")
                context = "\n".join([f"Doc: {d['Bates_Identity']}\n{d['Body']}" for d in docs_payload])
                
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {q}")
                chain = prompt | engine.llm_reduce
                response = chain.invoke({"context": context, "q": q})
                final_answer = response.content
                status.update(label="Analysis Complete", state="complete", expanded=False)
                facts = []

            else:
                status.write("🏗️ Architect is analyzing your question...")
                extraction_goal = engine.architect_query(q)
                status.write(f"🎯 Goal Set: *{extraction_goal.goal_description}*")
                
                # 2. STREAMLIT ASYNC SAFETY WRAPPER
                def safe_async_run(coroutine):
                    """Safely executes an async coroutine without crashing Streamlit's event loop."""
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                        
                    if loop and loop.is_running():
                        # If Streamlit is polluting the thread with an active loop, isolate via ThreadPool
                        with concurrent.futures.ThreadPoolExecutor(1) as pool:
                            return pool.submit(lambda: asyncio.run(coroutine)).result()
                    else:
                        # Standard pristine async run
                        return asyncio.run(coroutine)

                # Execute the new async pipeline
                facts = safe_async_run(engine.arun_parallel_map(docs_payload, extraction_goal, status_container=status))
                
                status.update(label="Pass 3: Synthesizing Final Answer...", state="running")
                final_answer = engine.reduce_facts(facts, q)
                status.update(label="Analysis Complete", state="complete", expanded=False)

        st.session_state.last_analysis = {
            "q": q,
            "final_answer": final_answer,
            "strategy": strategy,
            "facts": facts
        }

    # Display Result & Citations
    if st.session_state.get("last_analysis"):
        analysis_data = st.session_state.last_analysis
        st.info(analysis_data["final_answer"])
        
        if analysis_data["strategy"] == "MAP_REDUCE" and analysis_data.get("facts"):
            with st.expander("View Source Citations & Compliance Warnings"):
                for doc in analysis_data["facts"]:
                    st.markdown(f"**Document: {doc.internal_doc_id}**")
                    
                    # Display Compliance Alert if needed
                    if doc.requires_human_review:
                        st.error("🚨 COMPLIANCE ALERT: This document contained split/fragmented thoughts that the AI could not resolve. Human review of the raw text is required.")
                    
                    for fact in doc.verified_facts:
                        st.caption(f"Summary: {fact.merged_summary}")
                        for quote in fact.supporting_quotes:
                            st.text(f"\"{quote}\"")
                    st.divider()
        st.write("### 🔗 Share Analysis (QR Code)")
        
        qr_presets = {
            "Raw Data / Full Quality (Archive)": "RAW",
            "Instagram / TikTok Story (1080 x 1920)": (1080, 1920),
            "Square Feed Post (1080 x 1080)": (1080, 1080),
            "X / LinkedIn Post (1200 x 675)": (1200, 675)
        }
        
        # --- FIX: We bind the on_change callback here to clear old images ---
        selected_preset = st.selectbox("Select Target Platform / Image Size:", list(qr_presets.keys()), on_change=clear_qr_cache)
        
        # Add the text input here! Max_chars strictly enforces the UI limit
        custom_title = st.text_input("QR Code Title:", value="Graph Analysis", max_chars=22, on_change=clear_qr_cache)
        
        if st.button("Generate QR Code(s)", type="primary"):
            clear_qr_cache() 
            
            queries = get_selected_cypher_queries()
            if queries:
                try:
                    with st.spinner(f"Generating verified QR batch ({selected_preset})..."):
                        qr_master = SocialQRMaster()
                        
                        is_raw = qr_presets[selected_preset] == "RAW"
                        w, h = (1080, 1080) if is_raw else qr_presets[selected_preset]
                        
                        img_list = qr_master.generate_batch(
                            queries=queries, 
                            title=custom_title,   # <--- Change this to use the new custom_title variable
                            instruction=analysis_data["q"],
                            fill_color="#000000",
                            back_color="#FFFFFF",
                            width=w,
                            height=h,
                            app_address="silvios.ai",
                            raw_export=is_raw
                        )
                        
                        st.session_state.generated_qr_batch = []
                        for img in img_list:
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            st.session_state.generated_qr_batch.append(buf.getvalue())
                            
                        st.session_state.generated_qr_size = f"{img_list[0].width}x{img_list[0].height}"
                        
                except Exception as e:
                    st.error(f"Failed to generate QR: {e}")
            else:
                st.warning("No Cypher queries found in the selected evidence to share.")
                
        if "generated_qr_batch" in st.session_state:
            images = st.session_state.generated_qr_batch
            total_images = len(images)
            
            st.success(f"Generated {total_images} QR code(s) successfully!")
            
            if total_images > 1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for i, img_bytes in enumerate(images):
                        zip_file.writestr(f"graph_analysis_part_{i+1}_of_{total_images}.png", img_bytes)
                
                st.download_button(
                    label="📦 Download Complete Archive (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="graph_analysis_qrs.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
                st.write("Or download images individually below:")
            
            cols = st.columns(total_images if total_images <= 4 else 4)
            for i, img_bytes in enumerate(images):
                with cols[i % 4]:
                    st.image(img_bytes, caption=f"Part {i+1} of {total_images}")
                    st.download_button(
                        label=f"⬇️ Download Part {i+1}",
                        data=img_bytes,
                        file_name=f"graph_analysis_{st.session_state.generated_qr_size}_part{i+1}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_btn_{i}"
                    )
#teh process_imorted_qr can be moved to the helper functions part    

def process_imported_qr(queries, instruction, analyze_now):
    """Helper to run the queries, fetch IDs, and update state."""
    driver = get_db_driver() 
    if not driver:
        st.error("Not connected to database.")
        return
        
    all_found_ids = set()
    with st.spinner("Executing imported Cypher queries against the database..."):
        with driver.session() as session:
            for q in queries:
                try:
                    res = session.run(q)
                    data = [r.data() for r in res]
                    all_found_ids.update(extract_provenance_from_result(data))
                except Exception as e:
                    st.warning(f"A query failed to execute: {e}")

    if not all_found_ids:
        st.warning("Queries ran, but no matching documents were found in the current database.")
        return

    payload = {
        "query": f"Imported QR: {instruction or 'Custom Analysis'}",
        "answer": "Imported via QR Code",
        "ids": list(all_found_ids),
        "cypher": " \nUNION\n ".join(queries) 
    }
    
    st.session_state.app_state["evidence_locker"].append(payload)
    
    if analyze_now:
        st.session_state.app_state["selected_ids"] = set(all_found_ids)
        if instruction:
            st.session_state.auto_trigger_question = instruction
        st.session_state.current_page = "Analysis"
        st.rerun()
    else:
        st.toast("✅ Saved to Evidence Locker!")

# --- Helper Class to mock Streamlit's UploadedFile for ZIP extraction ---
class ZipImageWrapper:
    def __init__(self, data):
        self.data = data
    def getvalue(self):
        return self.data

@st.fragment
def screen_import_qr():
    st.title("Import Analysis (QR)")
    st.write("Upload a QR code (or a batch of QR codes / ZIP file) to instantly retrieve the context and run the analysis.")
    
    # --- FIX: Included 'zip' in supported types ---
    upload_list = st.file_uploader("Upload QR Code Image(s) or ZIP", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True)
    camera = st.camera_input("Or Scan with Camera")
    
    all_images = []
    
    if upload_list:
        for uploaded_file in upload_list:
            # --- FIX: Automatically extract and load images if a ZIP is uploaded ---
            if uploaded_file.name.lower().endswith('.zip'):
                with zipfile.ZipFile(uploaded_file, "r") as z:
                    for filename in z.namelist():
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_data = z.read(filename)
                            all_images.append(ZipImageWrapper(img_data))
            else:
                all_images.append(uploaded_file)
                
    if camera:
        all_images.append(camera)
    
    if all_images:
        detector = cv2.QRCodeDetector()
        
        def extract_qr_string(image_array):
            retval, decoded_info, _, _ = detector.detectAndDecodeMulti(image_array)
            if retval and decoded_info and any(decoded_info):
                return [info for info in decoded_info if info][0]
            decoded_text, _, _ = detector.detectAndDecode(image_array)
            if decoded_text: return decoded_text
            return None

        chunks = {}
        total_expected = None
        single_payload = None
        
        for img_input in all_images:
            file_bytes = np.asarray(bytearray(img_input.getvalue()), dtype=np.uint8)
            cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            h, w = cv_img.shape[:2]
            if min(w, h) > 1500:
                ratio = 1500 / min(w, h)
                cv_img = cv2.resize(cv_img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

            decoded_str = extract_qr_string(cv_img)
            if not decoded_str:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                decoded_str = extract_qr_string(gray)
            if not decoded_str:
                _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                decoded_str = extract_qr_string(thresh)
                
            if decoded_str:
                match = re.match(r"^\[CHUNK (\d+)/(\d+)\](.*)", decoded_str, flags=re.DOTALL)
                if match:
                    idx = int(match.group(1))
                    tot = int(match.group(2))
                    total_expected = tot
                    chunks[idx] = match.group(3)
                else:
                    single_payload = decoded_str

        final_payload_to_process = None
        
        if single_payload:
            final_payload_to_process = single_payload
        elif total_expected and len(chunks) == total_expected:
            final_payload_to_process = "".join([chunks[i] for i in range(1, total_expected + 1)])
            st.success(f"Successfully assembled {total_expected} QR Code chunks!")
        elif total_expected:
            st.info(f"⏳ Collected {len(chunks)} of {total_expected} required QR codes. Please upload the remaining images.")

        if final_payload_to_process:
            instruction, queries = SocialQRMaster.extract_payload(final_payload_to_process)
            if queries:
                st.success("✅ Payload Decoded Successfully!")
                st.markdown(f"**Question/Instruction:** {instruction or 'None'}")
                with st.expander("View Embedded Cypher Queries"):
                    for q in queries: st.code(q, language="cypher")
                
                st.divider()
                st.write("How would you like to proceed?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Proceed with Analysis", type="primary", use_container_width=True):
                        process_imported_qr(queries, instruction, analyze_now=True)
                with col2:
                    if st.button("No, Just Save to Locker", use_container_width=True):
                        process_imported_qr(queries, instruction, analyze_now=False)
            else:
                st.error("QR Code was read, but the payload format is unrecognized.")
        elif not single_payload and not total_expected:
            st.error("No valid QR codes found in the uploaded images. Make sure you extracted or uploaded the ZIP correctly.")


# --- DESKTOP ---

@st.fragment
def render_explorer_workspace(selector_type, selected_items):
    accent_line = "<hr style='border: 2px solid #00ADB5; opacity: 0.5; margin-top: 15px; margin-bottom: 15px;'>"

    # --- Accessible Hacker Palette ---
    COLOR_ROOT = "#FF8C00"          # Amber (Subject)
    COLOR_RELATIONSHIP = "#FFFFFF" #"#4E545C"  # Gunmetal Gray (Relationship)
    COLOR_TARGET = "#00ADB5"        # Teal (Object)
    COLOR_BORDER = "#FFFFFF"        # White borders

    c_mid, c_right = st.columns([2, 1])
    
    with c_mid:
        if not selected_items:
            st.info("Select entities from the left and click 'Show Data'.")
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
                (COLOR_RELATIONSHIP, "Connection (⬜)", "Layer 2: The Action/Connection", "border: 1px solid #666;"),
                (COLOR_TARGET, "Target Type (🟦)", "Layer 3: The Target Entity", "box-shadow: 0 0 5px " + COLOR_TARGET + ";")
            ]
            
        legend_html = '<div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; justify-content: center;">'
        for col, label, title, style_extra in legend_items:
             legend_html += f'<span style="display: flex; align-items: center;" title="{title}"><span style="width: 12px; height: 12px; background: {col}; border-radius: 50%; display: inline-block; margin-right: 5px; {style_extra}"></span>{label}</span>'
        legend_html += '</div>'
        
        st.markdown(legend_html, unsafe_allow_html=True)

        # 1. Fetch Data (Make sure this function is defined elsewhere in your script)
        if 'fetch_sunburst_data' in globals():
            df = fetch_sunburst_data(selector_type, selected_items)
        else:
             st.warning("fetch_sunburst_data is not defined in this scope. Provide a mock for now.")
             df = pd.DataFrame()

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
            id_to_count = {}
            for _, row in df.iterrows():
                root_val = row[path[0]]
                mid_val = row[path[1]]
                leaf_val = row[path[2]]
                
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
                    if depth == 0:
                        colors.append(COLOR_RELATIONSHIP)
                    elif depth == 1:
                        colors.append(COLOR_ROOT)
                    elif depth >= 2:
                        colors.append(COLOR_TARGET)
                    else:
                        colors.append('#333333')
                else:
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

            fig.update_traces(
                marker=dict(colors=colors), 
                hovertext=hover_texts, 
                hovertemplate="%{hovertext}<br><br><i>For definition, see Glossary below.</i><extra></extra>"
            )
            
        except Exception as e:
            pass

        # 5. Styling & UX
        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0), 
            height=500,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',  
            font=dict(color="white")       
        )
        
        fig.update_traces(marker=dict(line=dict(color=COLOR_BORDER, width=2)))
        st.plotly_chart(fig, use_container_width=True)

        # 6. Glossary
        visible_edges = sorted(df['edge'].unique()) if 'edge' in df.columns else []
        
        with st.expander("📖 Relationship Glossary", expanded=False):
            if not visible_edges:
                st.caption("No relationships visible.")
            else:
                for edge in visible_edges:
                    if 'get_rel_definition' in globals():
                        definition = get_rel_definition(edge)
                    else:
                        definition = "Definition not found."
                    st.markdown(f"**{edge}**: {definition}")

    with c_right:
        st.subheader("Filter Data", divider = "gray")

        # --- Conditional Filtering Logic ---
        
        if selector_type == "Text Mentions":
            st.caption("Standard Extraction: Entities found in Documents via 'MENTIONED_IN'.")
            st.info("No filters applicable.")
            final_filtered_df = df

        elif selector_type == "Connections":
            st.caption("Filter by Source and Target Nodes")
            
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
            st.caption("Filter by Relationships and Target Types")

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

        st.markdown(accent_line, unsafe_allow_html=True)
        st.subheader(":arrow_down_small: Add to Evidence Cart :arrow_down_small:", divider="gray")
        if st.button("Add to Evidence Cart", type="primary", use_container_width=True):
            # Safely assign ALL filter variables, catching if they don't exist in the current view
            try:
                edges_to_pass = selected_edges
            except (NameError, UnboundLocalError):
                edges_to_pass = []
                
            try:
                targets_to_pass = selected_targets
            except (NameError, UnboundLocalError):
                targets_to_pass = []
                
            try:
                sources_to_pass = selected_sources
            except (NameError, UnboundLocalError):
                sources_to_pass = []
        
            cypher = generate_cart_cypher(
                st.session_state.active_explorer_items, 
                selector_type, 
                edges_to_pass, 
                targets_to_pass,
                sources_to_pass 
            )
            
            cart_unique_ids = set()
            driver = get_db_driver()
            if driver and cypher:
                try:
                    with driver.session() as session:
                        result = session.run(cypher)
                        for rec in result:
                            if rec.get('id_list'):
                                for item in rec['id_list']:
                                    if isinstance(item, list):
                                        # Strict string cast to match Pandas behavior
                                        cart_unique_ids.update([str(i) for i in item])
                                    elif item is not None:
                                        cart_unique_ids.add(str(item))
                except Exception as e:
                    st.error(f"Error executing Cypher for payload: {e}")
        
            if not cart_unique_ids:
                st.error("No documents to add based on current filters.")
            else:
                names = [item['name'] for item in st.session_state.active_explorer_items]
                
                if len(names) > 1:
                    name_str = f"{selector_type}: {', '.join(names)}"
                elif len(names) == 1:
                    name_str = names[0]
                else:
                    name_str = "Unknown"
                    
                query_desc = f"Manual Explorer: {name_str}"
                
                payload = {
                    "query": query_desc,
                    "answer": f"Visual discovery found {len(cart_unique_ids)} related documents.",
                    "ids": list(cart_unique_ids),
                    "cypher": cypher 
                }
                
                if "app_state" not in st.session_state:
                    st.session_state.app_state = {}
                    
                if "evidence_locker" not in st.session_state.app_state:
                    st.session_state.app_state["evidence_locker"] = []
                    
                st.session_state.app_state["evidence_locker"].append(payload)
                st.toast(f"✅ Added {len(cart_unique_ids)} docs to Evidence Cart!")
                
        # The test mode, this can be commented out for the user
        with st.expander("🧪 TEST MODE: Verify Cypher Generation Parity", expanded=False):
            st.write("Click below to test if the generated Cypher retrieves the expected data based on the current filters.")
            
            if st.button("Run Cypher Parity Test"):
                # Safely assign ALL filter variables
                try:
                    edges_to_pass_test = selected_edges
                except (NameError, UnboundLocalError):
                    edges_to_pass_test = []
                    
                try:
                    targets_to_pass_test = selected_targets
                except (NameError, UnboundLocalError):
                    targets_to_pass_test = []
                    
                try:
                    sources_to_pass_test = selected_sources
                except (NameError, UnboundLocalError):
                    sources_to_pass_test = []
        
                test_cypher = generate_cart_cypher(
                    st.session_state.active_explorer_items, 
                    selector_type,
                    edges_to_pass_test, 
                    targets_to_pass_test,
                    sources_to_pass_test
                )
                
                st.markdown("**Self-Contained Cypher Query:**")
                st.code(test_cypher, language="cypher")
                
                try:
                    if 'get_db_driver' in globals():
                        driver = get_db_driver()
                    else:
                        driver = None
                        
                    if not driver:
                        st.error("Could not retrieve Neo4j driver from session state. Ensure your app is connected to the database.")
                    else:
                        with driver.session() as session:
                            result = session.run(test_cypher)
                            records = [dict(record) for record in result]
                        
                        flattened_db_ids = []
                        for rec in records:
                            if not rec.get('id_list'):
                                continue
                            for item in rec['id_list']:
                                if isinstance(item, list):
                                    # strict string cast!
                                    flattened_db_ids.extend([str(i) for i in item])
                                elif item is not None:
                                    flattened_db_ids.append(str(item))
                                    
                        distinct_db_ids = set(flattened_db_ids)
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric("Raw Rows Returned", len(records))
                        col_b.metric("Distinct IDs Found", len(distinct_db_ids))
                        
                        if len(records) > 0 and len(distinct_db_ids) == 0:
                            st.warning("⚠️ The query found matching paths, but `r.source_pks` and `m.doc_id` were null for all of them.")
                        elif len(records) == 0:
                            st.warning("⚠️ The query ran successfully but found 0 matching paths.")
        
                        st.write("**Raw Cypher Results:**")
                        
                        display_records = []
                        for rec in records:
                            display_rec = rec.copy()
                            display_rec['id_list'] = str(rec.get('id_list', []))
                            display_records.append(display_rec)
                            
                        st.dataframe(display_records)
                    
                except Exception as e:
                    st.error(f"Error executing test Cypher: {e}")

# Generate cypher from cart items function. This should be able to properly output the cypher we generated by modifying the output from the cypher manually
def generate_cart_cypher(active_items, selector_type, selected_edges=None, selected_targets=None, selected_sources=None):
    if not active_items:
        return ""

    selected_edges = list(selected_edges) if selected_edges is not None else []
    selected_targets = list(selected_targets) if selected_targets is not None else []
    selected_sources = list(selected_sources) if selected_sources is not None else []

    if selector_type == "Connections":
        rel_types = [item['name'] for item in active_items]
        formatted_rels = json.dumps(rel_types)
        
        actual_sources = selected_sources if len(selected_sources) > 0 else selected_edges
        
        source_filter = ""
        if len(actual_sources) > 0:
            formatted_sources = json.dumps(actual_sources)
            source_filter = f"\n      AND labels(n)[0] IN {formatted_sources}"
            
        target_filter = ""
        if len(selected_targets) > 0:
            formatted_targets = json.dumps(selected_targets)
            target_filter = f"\n      AND labels(m)[0] IN {formatted_targets}"
        
        cypher = f"""
    MATCH (n)-[r]->(m)
    WHERE type(r) IN {formatted_rels}
      AND type(r) <> 'MENTIONED_IN'{source_filter}{target_filter}
    RETURN 
        labels(n)[0] AS source_label, 
        type(r) AS edge, 
        labels(m)[0] AS target_label,
        collect(coalesce(r.source_pks, m.doc_id)) AS id_list
    """
        return cypher

    elif selector_type == "Text Mentions":
        label_groups = defaultdict(list)
        for item in active_items:
            label_groups[item['label']].append(item['name'])

        source_clauses = []
        for label, names in label_groups.items():
            formatted_names = json.dumps(names)
            source_clauses.append(f"(n:`{label}` AND n.name IN {formatted_names})")

        source_where = " OR ".join(source_clauses) if source_clauses else "TRUE"

        cypher = f"""
    MATCH (n)-[r:MENTIONED_IN]->(m:Document)
    WHERE {source_where}
    RETURN 
        n.name AS source_name, 
        type(r) AS edge, 
        labels(m)[0] AS target_label,
        collect(coalesce(r.source_pks, m.doc_id)) AS id_list
    """
        return cypher

    elif selector_type == "Entities":
        label_groups = defaultdict(list)
        for item in active_items:
            label_groups[item['label']].append(item['name'])

        source_clauses = []
        for label, names in label_groups.items():
            formatted_names = json.dumps(names)
            source_clauses.append(f"(n:`{label}` AND n.name IN {formatted_names})")

        source_where = " OR ".join(source_clauses) if source_clauses else "TRUE"

        edge_filter = ""
        if len(selected_edges) > 0:
            formatted_edges = json.dumps(selected_edges)
            edge_filter = f"\n      AND type(r) IN {formatted_edges}"

        target_filter = ""
        if len(selected_targets) > 0:
            formatted_targets = json.dumps(selected_targets)
            target_filter = f"\n      AND labels(m)[0] IN {formatted_targets}"

        cypher = f"""
    MATCH (n)-[r]->(m)
    WHERE ({source_where})
      AND type(r) <> 'MENTIONED_IN'{edge_filter}{target_filter}
    RETURN 
        n.name AS source_name, 
        type(r) AS edge, 
        labels(m)[0] AS target_label,
        collect(coalesce(r.source_pks, m.doc_id)) AS id_list
    """
        return cypher
        
    else:
        return ""
        
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
                captions=["Directed", "Directed", "Entity mentioned in Document"],
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
### 7. PROMPTS  ###
# ==========================================

# --- PROMPTS FOR GRAPHRAG PIPELINE  ---

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
   - **FALLBACK FOR NON-PERSONS:** If the user asks for a specific organization like 'Project Omega', map it to the `ORGANIZATION` label but DO NOT add a `WHERE` clause for its name. The Synthesizer will filter the results post-query.
   - **FUZZY MATCHING REQUIRED:** When filtering PERSON properties, you MUST use `toLower(n.prop) CONTAINS 'value'`.
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
1. STRICT VARIABLE NAMING (MANDATORY):
   - Always use `n` for the source/starting node.
   - Always use `r` for the primary relationship.
   - Always use `m` for the target/destination node.
   - Always use `d` if matching a Document node.

2. SAFE RETURN POLICY (STRICT):
   - For NODES: You MUST ONLY return the `.name` property (e.g., `n.name`, `m.name`).
   - DO NOT return generic properties like `.title`, `.age`, `.role` even if the user asks, as they are unreliable.
   - For PROVENANCE: Always return `coalesce(r.source_pks, r.doc_id)` to satisfy source requirements.

3. STRING MATCHING (MANDATORY): For ALL string property filters in WHERE clauses, you MUST use `toLower(n.prop) CONTAINS 'value'`.
   - BAD: `WHERE n.name = 'John Doe'`
   - GOOD: `WHERE toLower(n.name) CONTAINS 'john doe'`
   - **NEGATIVE MATCHING:** For exclusion, use `NOT ... CONTAINS`.
     - BAD: `WHERE n.name <> 'Gates'`
     - GOOD: `WHERE NOT toLower(n.name) CONTAINS 'gates'`

4. PATHS & LOGIC:
   - **Continuous Paths:** Ensure the path is fully connected. `(n)-[r1]->(m)-[r2]->(o)`. NEVER use comma-separated disconnected patterns like `MATCH (n), (m)` (Cartesian Product).
   - **OR Logic:** If the Grounding Agent provided pipes `|` in labels (e.g., `Person|Organization`), write them EXACTLY as provided in the Cypher (e.g., `(n:Person|Organization)`).

5. PROPERTIES IN WHERE CLAUSES: You may use other properties (e.g. `.date`, `.status`) ONLY in the `WHERE` clause to filter data, and ONLY if they are explicitly listed in the Schema.
6. DISTINCT: Always use `RETURN DISTINCT` to avoid duplicates.
7. SYNTAX: Do not include semicolons at the end.

EXAMPLES:

Input: 
  source_node_label="Person", target_node_label="Person"
  relationship_paths=["FINANCIAL_TRANSACTION"]
  cypher_constraint_clause="toLower(n.name) CONTAINS 'john doe'"
Output: 
  MATCH (n:Person)-[r:FINANCIAL_TRANSACTION]->(m:Person) 
  WHERE toLower(n.name) CONTAINS 'john doe' 
  RETURN DISTINCT n.name, type(r), m.name, coalesce(r.source_pks, r.doc_id)

Input: 
  source_node_label="Person", target_node_label="Organization"
  relationship_paths=["VISITED"]
  cypher_constraint_clause="toLower(n.name) CONTAINS 'smith'"
Output: 
  MATCH (n:Person)-[r:VISITED]->(m:Organization) 
  WHERE toLower(n.name) CONTAINS 'smith' 
  RETURN DISTINCT n.name, type(r), m.name, coalesce(r.source_pks, r.doc_id)
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

# ==========================================
# 8. AUTHENTICATION & SETTINGS LOGIC
# ==========================================


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

#######################################################################
# --- 9. GLOBAL UI COMPONENTS AND CSS(THEME, CUSTOM CSS INJECTION) ---
######################################################################
#WELCOME BUTTON AND TEXT
WELCOME_TEXT="""
Terms of Use and Legal Disclaimer

Last Updated: February 15, 2026

1. Nature of the Project & Age Restriction

1.1. Portfolio Demonstration Only
This application ("AI Graph Analyst") is a portfolio project created solely for educational and demonstration purposes. It is designed to showcase technical capabilities in GraphRAG (Retrieval Augmented Generation), AI-driven document analysis, and data visualization. It is NOT a commercial product, a legal investigation tool, or an official source of government information.

1.2. 18+ Age Requirement (Sensitive Content)
This application processes and visualizes data from the "Epstein Documents" and House Oversight Committee releases. These documents contain explicit descriptions of sexual abuse, crimes involving minors, and other disturbing content.

By accessing this application, you certify that you are at least 18 years of age (or the age of majority in your jurisdiction).

Access by minors is strictly prohibited.

2. Disclaimer of Data Accuracy (The "Public Record" Clause)

The data visualized in this application is derived from publicly available government documents.

No Guarantee of Accuracy: Errors may occur during data cleaning (OCR), ingestion, or processing.

Contextual Limitations: A connection in the graph (e.g., "MENTIONED_IN") does not imply guilt, criminal association, or verified personal relationships. It simply indicates that two terms appear structurally linked in the raw text.

Source Material: Users are strictly advised to verify any findings against the original, official source documents provided by the United States Government. This tool is a secondary visualization aid, not a primary source.

3. AI and Large Language Model (LLM) Warning

This application utilizes Artificial Intelligence (Mistral AI) to interpret text.

Risk of Hallucination: Generative AI can fabricate information. The AI may misinterpret a relationship, invent a citation, or misquote a document.

No Human Review: The outputs are automated and have not been reviewed by human editors.

User Responsibility: You accept that any summary, graph, or answer provided by the AI is a probabilistic generation, not a verified fact.

4. Limitation of Liability

TO THE FULLEST EXTENT PERMITTED BY LAW:

"As Is" Service: The software is provided "AS IS", without warranty of any kind.

No Liability for Damages: The Developer shall not be liable for any direct, indirect, incidental, special, or consequential damages (including reputational harm, loss of data, or legal reliance) arising from the use of this application.

Service Interruptions: We do not guarantee uptime or data persistence.

5. User Conduct & Indemnification

5.1. Prohibited Acts
You agree NOT to:

Use outputs to harass, defame, or doxx individuals.

Present AI "theories" as verified fact in public forums.

Input prompts that violate the Acceptable Use Policy of our AI provider (Mistral AI), including generating non-consensual sexual content or hate speech.

5.2. Indemnification (The "You Pay if You Break It" Clause)
You agree to indemnify, defend, and hold harmless the Developer from any claims, liabilities, damages, and expenses (including legal fees) arising from your use of the application, your violation of these Terms, or your violation of any rights of a third party (e.g., posting a defamatory screenshot).

6. Intellectual Property

The underlying code is the intellectual property of the Developer. The underlying data remains in the public domain or subject to original copyright.

7. Privacy and Cookie Policy (EU/GDPR)

7.1. No Direct Data Collection: The Developer does not create accounts or store PII.
7.2. Hosting (Streamlit/Snowflake): Essential cookies are used by the host for site function.
7.3. AI Processing: Inputs are sent to Mistral AI for processing. The Developer may review anonymized logs for debugging.
7.4. Consent: By using the app, you consent to this processing.

8. Modifications

We reserve the right to modify these terms or the application features at any time without notice.

9. Contact & Right to Erasure (GDPR/Accuracy)

We respect individual privacy and data accuracy.

Technical Corrections: If you identify a factual error in the graph structure (e.g., the AI hallucinated a link that does not exist in the source text), please report it. We prioritize fixing technical inaccuracies.

GDPR Rights: EU citizens have the right to request erasure of personal data under specific conditions. While public interest exceptions may apply to government records, we will review all removal requests in compliance with applicable law.

Contact: Please direct requests to the repository owner via [GitHub Issues] at ssopic.

10. Governing Law & Dispute Resolution

10.1. Jurisdiction
These Terms shall be governed by and construed in accordance with the laws of The Republic of Croatia (or applicable EU law), without regard to its conflict of law provisions.

10.2. Exclusive Venue
Any legal action or proceeding arising under these Terms shall be brought exclusively in the competent courts located in Croatia. You hereby consent to the jurisdiction of such courts and waive any objection regarding venue (e.g., claiming it is an "inconvenient forum").

By clicking "Get Started," you acknowledge you are 18+ and agree to these Terms.
"""

@st.dialog("Welcome")
def show_welcome_popup():
    # 2. Add welcome pop up
    st.write(WELCOME_TEXT)
    if st.button("Get Started"):
        st.session_state.welcome_shown = True
        st.rerun()

THEME = {
    "bg_base": "#09090b",       # Zinc-950 (was #0E1117)
    "bg_panel": "#18181b",      # Zinc-900 (was #1F2129)
    "bg_input": "#27272a",      # Zinc-800
    "border_subtle": "#3f3f46", # Zinc-700 (was #41444C)
    "text_primary": "#f4f4f5",  # Zinc-100 (was #FFFFFF)
    "text_secondary": "#a1a1aa",# Zinc-400
    "accent": "#d97706",        # Amber-600 (was #00ADB5 Cyan) - The "Fix"
    "accent_hover": "#b45309",  # Amber-700
    "shadow_subtle": "0 1px 2px rgba(0,0,0,0.5)",
}

def inject_custom_css():
    """
    Hides the standard Streamlit sidebar and applies styling for the cockpit layout.
    Refactored to "Industrial Graph" aesthetics to fix AI Tells while keeping all selectors.
    """
    st.markdown(
        f"""
        <style>
            /* 1. Hide the default Streamlit Sidebar elements */
            [data-testid="stSidebar"] {{ display: none; }}
            [data-testid="collapsedControl"] {{ display: none; }}
            
            /* 2. Main Layout Adjustment */
            .stApp {{
                background-color: {THEME['bg_base']}; /* Fixed: Deep Zinc */
                color: {THEME['text_primary']};
            }}

            .block-container {{
                padding-top: 4rem; 
                padding-bottom: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
                max_width: 100%;
            }}

            /* 3. TECH BUTTON STYLING */
            /* Target ALL buttons. Using "Industrial" tactile feel instead of "Neon" */
            div.stButton > button {{
                width: 100%;
                background: {THEME['bg_panel']} !important; 
                background-color: {THEME['bg_panel']} !important;
                color: {THEME['text_primary']} !important; 
                border: 1px solid {THEME['border_subtle']} !important; 
                border-radius: 4px;
                
                /* SIZE ADJUSTMENT: Preserved from original */
                min-height: 42px !important; 
                height: auto !important;
                padding-top: 0.25rem !important;
                padding-bottom: 0.25rem !important;

                font-family: 'Inter', 'Source Sans Pro', sans-serif;
                font-weight: 600 !important;
                letter-spacing: 0.5px;
                transition: all 0.2s ease-in-out; 
                box-shadow: {THEME['shadow_subtle']} !important; /* Fixed: Removed Glow */
            }}
            
            div.stButton > button p {{
                color: {THEME['text_primary']} !important;
            }}

            /* Focus/Active States */
            div.stButton > button:focus,
            div.stButton > button:active {{
                background: {THEME['bg_input']} !important;
                color: {THEME['text_primary']} !important;
                border-color: {THEME['accent']} !important;
                box-shadow: none !important;
            }}

            /* ACCENT: Amber Border & Text on Hover (Fixed: No Neon Glow) */
            div.stButton > button:hover {{
                background: {THEME['bg_input']} !important; 
                border-color: {THEME['text_secondary']} !important; /* Subtle hover */     
                color: {THEME['text_primary']} !important;        
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4) !important; /* Physical depth */          
            }}
            
            div.stButton > button:hover p {{
                color: {THEME['text_primary']} !important;
            }}
            
            /* 4. GLOBAL TEXT STYLING */
            h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stText, .stCaption {{
                color: {THEME['text_primary']} !important;
                font-weight: 500 !important; 
            }}

            /* FIX FOR MULTI-LINE HEADERS */
            h1 p, h2 p, h3 p, h4 p, h5 p, h6 p,
            h1 span, h2 span, h3 span, h4 span, h5 span, h6 span {{
                font-size: inherit !important;
                font-weight: inherit !important;
                color: inherit !important;
                line-height: 1.2 !important; 
                margin-bottom: 0 !important;
            }}

            /* 5. Tighter Dividers */
            hr {{
                border-color: {THEME['border_subtle']};
                margin-top: 0.5em !important;
                margin-bottom: 0.5em !important;
            }}

            /* 6. EXPANDER (DATABOOK DROPDOWNS) FIX */
            
            div[data-testid="stExpander"] details {{
                background-color: {THEME['bg_panel']} !important;
                border-color: {THEME['border_subtle']} !important;
                border-radius: 4px;
            }}

            div[data-testid="stExpander"] summary {{
                background-color: {THEME['bg_panel']} !important;
                color: {THEME['text_primary']} !important;
                border: 1px solid {THEME['border_subtle']} !important;
                border-radius: 4px;
                transition: border-color 0.2s, color 0.2s;
            }}

            /* ACCENT: Hover state for the Expander Header - Uses Amber */
            div[data-testid="stExpander"] summary:hover {{
                background-color: {THEME['bg_panel']} !important; 
                border-color: {THEME['accent']} !important; /* Amber Border */
                color: {THEME['accent']} !important; /* Amber Text */
            }}

            div[data-testid="stExpander"] summary span,
            div[data-testid="stExpander"] summary p {{
                 color: inherit !important;
            }}

            div[data-testid="stExpander"] summary svg {{
                fill: {THEME['text_secondary']} !important;
            }}
            div[data-testid="stExpander"] summary:hover svg {{
                fill: {THEME['accent']} !important;
            }}
            
            div[data-testid="stExpander"] div[role="group"] {{
                 background-color: {THEME['bg_base']} !important; 
                 color: {THEME['text_primary']} !important;
                 border: 1px solid {THEME['border_subtle']};
                 border-top: none;
            }}

            /* 7. CHECKBOX & INPUT FIXES INSIDE EXPANDER */
            label[data-baseweb="checkbox"] span {{
                color: {THEME['text_primary']} !important;
            }}
            
            /* 8. MULTISELECT & DROPDOWN LIST FIXES */
            span[data-baseweb="tag"] {{
                background-color: {THEME['bg_input']} !important; 
                color: {THEME['text_primary']} !important;
                border: 1px solid {THEME['border_subtle']};
            }}
            
            span[data-baseweb="tag"] svg {{
                fill: {THEME['text_primary']} !important;
            }}
            
            div[data-baseweb="popover"],
            div[data-baseweb="menu"] {{
                background-color: {THEME['bg_panel']} !important;
                border: 1px solid {THEME['border_subtle']} !important;
            }}
            
            li[role="option"] {{
                background-color: {THEME['bg_panel']} !important;
                color: {THEME['text_primary']} !important;
            }}
            
            li[role="option"]:hover,
            li[role="option"][aria-selected="true"] {{
                background-color: {THEME['bg_input']} !important;
                color: {THEME['accent']} !important; /* Amber Text */
            }}
            
            div[data-baseweb="select"] > div {{
                background-color: {THEME['bg_panel']} !important;
                color: {THEME['text_primary']} !important;
                border-color: {THEME['border_subtle']} !important;
            }}

            /* 9. NOTIFICATIONS & ALERTS (Toasts) */
            div[data-testid="stToast"] {{
                background-color: {THEME['bg_panel']} !important;
                border: 1px solid {THEME['border_subtle']} !important;
                color: {THEME['text_primary']} !important;
            }}
            
            div[data-testid="stToast"] p, 
            div[data-testid="stToast"] div {{
                color: {THEME['text_primary']} !important;
            }}
            
            div[data-testid="stAlert"] {{
                background-color: {THEME['bg_panel']} !important;
                color: {THEME['text_primary']} !important;
                border: 1px solid {THEME['border_subtle']};
            }}
            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] div {{
                color: {THEME['text_primary']} !important;
            }}

            /* 10. CODE BLOCKS & TRACES */
            div[data-testid="stCodeBlock"] {{
                background-color: {THEME['bg_base']} !important;
                border: 1px solid {THEME['border_subtle']};
                border-radius: 4px;
            }}
            
            code {{
                color: {THEME['accent']} !important; /* Amber for code */
                background-color: transparent !important; 
                font-family: 'JetBrains Mono', 'Courier New', monospace !important;
            }}
            
            pre {{
                background-color: {THEME['bg_base']} !important;
                color: {THEME['text_primary']} !important;
                border: 1px solid {THEME['border_subtle']};
            }}

            /* 11. JSON & RAW TEXT FIXES */
            div[data-testid="stJson"],
            .react-json-view {{
                background-color: {THEME['bg_base']} !important;
                color: {THEME['text_primary']} !important;
            }}
            
            .react-json-view span {{
                color: {THEME['accent']} !important; 
            }}
            
            div[data-testid="stText"] {{
                background-color: {THEME['bg_base']} !important;
                color: {THEME['accent']} !important;
                font-family: 'JetBrains Mono', 'Courier New', monospace !important;
            }}

            /* 12. SETTINGS DIALOG (MODAL) FIXES */
            div[role="dialog"][aria-modal="true"] {{
                background-color: {THEME['bg_panel']} !important;
                border: 1px solid {THEME['border_subtle']} !important;
                color: {THEME['text_primary']} !important;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5) !important;
            }}

            div[role="dialog"] header {{
                background-color: {THEME['bg_panel']} !important;
                color: {THEME['text_primary']} !important;
            }}

            div[role="dialog"] div, 
            div[role="dialog"] label,
            div[role="dialog"] p {{
                color: {THEME['text_primary']} !important;
            }}

            button[aria-label="Close"] {{
                color: {THEME['text_primary']} !important;
                background-color: transparent !important;
                border: none !important;
            }}
            button[aria-label="Close"]:hover {{
                color: {THEME['accent']} !important;
            }}

            /* 13. VERTICAL SEPARATION LINES (Column Borders) - THE CRITICAL FIX */
            /* Removed 3px Hard Neon Border -> Replaced with 1px Subtle Border + Panel Background */
            
            /* LEFT FRAME (First Column) */
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(1),
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(1),
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1),
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) {{
                border-right: 1px solid {THEME['border_subtle']} !important; /* Fixed: 3px -> 1px */
                background-color: {THEME['bg_panel']}; /* Fixed: Distinct Panel Color */
                padding: 1rem !important;
                min-height: 85vh; 
                box-shadow: none; /* Fixed: Removed Neon Glow */
            }}
            
            /* RIGHT FRAME (Last Column) */
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
            .block-container > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
            .block-container > div > div > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {{
                border-left: 1px solid {THEME['border_subtle']} !important; /* Fixed: 3px -> 1px */
                background-color: {THEME['bg_panel']};
                padding: 1rem !important;
                min-height: 85vh;
                box-shadow: none;
            }}
            
            /* Reset nested columns */
            [data-testid="stColumn"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"],
            [data-testid="column"] [data-testid="stHorizontalBlock"] [data-testid="column"] {{
                border: none !important;
                background-color: transparent !important;
                box-shadow: none !important;
                min-height: 0 !important;
                padding: 0 !important;
            }}

            /* 14. SEARCH BAR & TEXT INPUT FIXES */
            
            div[data-testid="stTextInput"] div[data-baseweb="input"] {{
                background-color: {THEME['bg_panel']} !important;
                border-color: {THEME['border_subtle']} !important;
                border-radius: 4px;
            }}
            
            div[data-testid="stTextInput"] input {{
                color: {THEME['text_primary']} !important;  
                background-color: {THEME['bg_panel']} !important;
                caret-color: {THEME['accent']} !important; /* Amber caret */
            }}
            
            div[data-testid="stTextInput"] input::placeholder {{
                color: {THEME['text_secondary']} !important;
            }}

            div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
                border-color: {THEME['accent']} !important;
                box-shadow: 0 0 0 1px {THEME['accent']} !important; /* Clean focus ring, no fuzz */
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
accent_line = "<hr style='border: 2px solid #3f3f46; opacity: 0.5; margin-top: 15px; margin-bottom: 15px;'>"

###############################################
# --- MAIN ---
###############################################

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
        
    # Welcome button
    if "welcome_shown" not in st.session_state:
        show_welcome_popup()

    # 3. Cockpit Layout (3 Columns)
    # Ratios: [1.2, 8, 1.2]
    c_left, c_main, c_right = st.columns([1.2, 8, 1.2], gap="medium")

    # --- LEFT COLUMN (Input & Config) ---
    with c_left:
        st.markdown("<div style='text-align: center; font-size: 1.3em;'><b>Find Evidence and add to cart</b></div>", unsafe_allow_html=True)
        st.markdown(accent_line, unsafe_allow_html=True) # Assuming accent_line is defined in your main file
        
        # Navigation Buttons (Using callbacks for single-click nav)
        st.button("Find Evidence Manually",  use_container_width=True, 
                  on_click=set_page, args=("Find Evidence Manually",))
            
        st.button("Find Evidence via Chat or Cypher", use_container_width=True,
                  on_click=set_page, args=("Find Evidence via Chat or Cypher",))

        # --- NEW BUTTON: Import Analysis (QR) ---
        st.button("Import Analysis (QR)", use_container_width=True,
                  on_click=set_page, args=("Import QR",))

        # Vertical Spacer to push Config to bottom
        st.markdown("<br>"*10, unsafe_allow_html=True)
        
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
            # --- NEW ROUTE: Display the QR screen ---
            elif current == "Import QR":
                screen_import_qr()
            else:
                st.error(f"Unknown page: {current}")

    # --- RIGHT COLUMN (Output & Tools) ---
    with c_right:
        st.markdown("<div style='text-align: center; font-size: 1.3em;'><b>Select from Cart and Analyze</b></div>", unsafe_allow_html=True)
        st.markdown(accent_line, unsafe_allow_html=True)
        
        # Locker Badge Calculation
        locker_count = len(st.session_state.app_state["evidence_locker"])
        badge = f" ({locker_count})" if locker_count > 0 else ""
        
        st.button(f"Evidence Cart{badge}",  use_container_width=True,
                  on_click=set_page, args=("Evidence Cart",))
            
        st.button("Analysis",  use_container_width=True,
                  on_click=set_page, args=("Analysis",))

# --- Standard Python Runner ---
if __name__ == "__main__":
    main()
            


