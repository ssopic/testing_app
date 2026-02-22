# AI Graph Analyst: Multi-Agent GraphRAG & Document Intelligence

An advanced investigative platform designed to analyze public government records through a hybrid of **Graph-based Retrieval Augmented Generation (GraphRAG)** and **MapReduce document synthesis**.

This application transforms unstructured public records into a queryable knowledge graph (Neo4j) and utilizes an agentic swarm (Mistral AI) to navigate complex relationships, financial transactions, and document evidence.

## Key Features

### 1. Hybrid RAG Architecture
* **GraphRAG Pipeline**: Converts natural language into precise Cypher queries using a 5-stage agent pipeline (Intent → Schema → Grounding → Generation → Synthesis).
* **MapReduce Engine**: Handles high-volume document analysis. When context exceeds token limits, the system spawns an "Architect" agent and a "Map Worker" swarm to scan fragments in parallel before synthesizing a final report.

### 2. Investigative Tools
* **Visual Explorer (The Databook)**: Interactive sunburst visualizations to browse entities, connections, and text mentions.
* **Evidence Cart**: A session-based "locker" to save findings from both visual exploration and AI chat for final deep-dive analysis.
* **Safe Cypher Sandbox**: A read-only interface for expert analysts to run manual queries with automated provenance extraction.

### 3. Intelligence Layers
* **Agentic Grounding**: Ensures AI-generated queries adhere strictly to the database schema while supporting fuzzy matching for person entities.
* **Automated Debugger**: A self-healing loop that catches Cypher syntax errors and retries execution with corrected logic.

---

## 🛠️ Tech Stack

* **UI/UX**: [Streamlit](https://streamlit.io/) (Custom "Industrial Cockpit" Theme)
* **Graph Database**: [Neo4j](https://neo4j.com/)
* **LLM Framework**: [LangChain](https://www.langchain.com/)
* **Models**: Mistral AI (`mistral-medium`, `mistral-small`)
* **Observability**: [LangSmith](https://www.langchain.com/langsmith) (Integrated tracing)
* **Visualization**: [Plotly Express](https://plotly.com/python/)

---

## ⚙️ Setup & Installation

### Prerequisites
* A Neo4j instance (AuraDB or Local)
* Mistral AI API Key
* Python 3.9+

### Environment Configuration
The app retrieves credentials from `st.secrets` or environment variables. Create a `.env` file or use Streamlit's `secrets.toml`:

```toml
MISTRAL_API_KEY = "your_mistral_key"
NEO4J_URI = "bolt://your_neo4j_instance"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/ai-graph-analyst.git](https://github.com/your-username/ai-graph-analyst.git)
   cd ai-graph-analyst
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## ⚖️ Legal Disclaimer
This tool is for educational and portfolio purposes. It processes data from public government releases (e.g., House Oversight Committee). Users must be 18+ and are responsible for verifying all AI-generated findings against original source documents.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
