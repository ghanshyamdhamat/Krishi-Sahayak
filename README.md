# SAHAYAK KRISHI: Multilingual Agricultural AI Assistant

This project is an intelligent, multilingual AI assistant for farmers, integrating **RAG (Retrieval-Augmented Generation)**, **Elasticsearch**, **Neo4j Knowledge Graph**, and **LLMs (Qwen3-8B)**. It supports text, speech, and image-based queries, and provides recommendations, government scheme info, crop advice, and more.

-----

## \#\# Table of Contents

  * [Features](https://www.google.com/search?q=%23features)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [Setup Instructions](https://www.google.com/search?q=%23setup-instructions)
  * [How the Workflow Runs](https://www.google.com/search?q=%23how-the-workflow-runs)
  * [Usage](https://www.google.com/search?q=%23usage)
  * [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
  * [Acknowledgements](https://www.google.com/search?q=%23acknowledgements)

-----

## \#\# Features

  * **Multilingual support** (Indian languages & English) via AI4Bharat IndicTrans2.
  * **Text, speech, and image input** (soil & rice disease classification).
  * **RAG pipeline** that combines LLM reasoning with Elasticsearch and Knowledge Graph search.
  * **Government scheme search** and **crop recommendations**.
  * Live **weather** and **market price** APIs.
  * **Neo4j Knowledge Graph** for storing and querying agricultural expertise.
  * **On-demand model loading** for efficient GPU/CPU memory usage.

-----

## \#\# Project Structure

capital_one_agent_ai/
│
├── backend/
│   ├── chatbot_2.py           # Main Streamlit app (UI, orchestrator)
│   ├── farmer_clean_agent.py  # Core agent logic, workflow, LLM reasoning
│   ├── agent_tools_clean.py   # Tools: ES, KG, crop model, weather, etc.
│   ├── es_utils.py            # Elasticsearch utility functions
│   ├── indexing/
│   │   └── index.py           # Indexing schemes into Elasticsearch
│   ├── model/
│       │- crop-recommendation model
│       │- plant disease classification model
│       │- soil identification model
│
├── requirements_cap_one.txt   # Python dependencies
└── README.md                  # (You are here)

-----

## \#\# Setup Instructions

#### \#\#\#\# 1. Clone the Repository

```bash
git clone git@github.com:pushpayush007/farming_agent.git
cd <repository-directory>
```

#### #### 2. Install Python Dependencies

It is highly recommended to use a virtual environment.

```bash
conda create -n cap_one python=3.10
conda activate cap_one
pip install -r requirements.txt
```
Oye# \#\#\#\# 3. Set Up Elasticsearch

  * **Option 1: Run Elasticsearch using Docker (Recommended)**
      1. Make sure you have [Docker installed](https://docs.docker.com/get-docker/).
          2. Start Elasticsearch 8.x with the following command:
                  ```bash
                          docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.13.4
                                  ```
                                      3. Wait a few seconds for Elasticsearch to start. You can check with:
                                              ```bash document installed and Docker
        curl http://localhost:9200
        ```
       You should see a JSON response with cluster information.

  * **Option 2: Install Elasticsearch manually**
    1. Download Elasticsearch 8.x from [the official website](https://www.elastic.co/downloads/elasticsearch).
    2. Follow the installation instructions for your OS.
    3. Start the Elasticsearch service.

  * By default, the code expects Elasticsearch at `http://localhost:9200`.

  * **Index the data into Elastic search:**
    ```bash
    python indexing/index.py
    python indexing/indexing_pdf_data.py
    python indexing/indexing_metadata_schemes.py
    python indexing/elasticsearch_ingest_paddy.py
    ```
    This will create the `schemes` index and upload all scheme documents with their corresponding embeddings.

#### #### 4. Set Up Neo4j Knowledge Graph (using Docker)

    1. Make sure you have [Docker installed](https://docs.docker.com/get-docker/).
    2. Start Neo4j Community Edition with the following command:
        ```bash
        docker run -d --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test1234 neo4j:4.4
        ```
    3. Wait a few seconds for Neo4j to start. You can check the browser UI at: [http://localhost:7474](http://localhost:7474)

      To ingest extracted agricultural knowledge (from PDF or JSON) into the Neo4j Knowledge Graph, use the provided script:

      ```bash
      python backend/neo4j_files/store_pdf_data.py
      ```

      By default, this script is set up to process the file:
      ```
      backend/indexing/paddy_data2_extractions.json
      ```
      and store its contents in Neo4j at `bolt://localhost:7687` (user: `neo4j`, password: `test1234`).

      - You can modify the JSON file path in `store_pdf_data.py` if you want to ingest a different file.
      - After running, the script will print statistics about the nodes and relationships created in the knowledge graph.

**Make sure Neo4j is running before executing this script.**

#### \#\#\#\# 5. Download/Place ML Models

  * Place your crop recommendation, soil, and rice disease models in the **`backend/model/`** directory.
  * **Do not commit large model files to git.** Add them to your **`.gitignore`** file.

#### \#\#\#\# 6. Environment Variables

Create a **`.env`** file in the **`backend/`** directory if you need to override default settings (e.g., model paths, API keys).

-----

## \#\# How the Workflow Runs

#### \#\#\#\# Step 1: Start the Chatbot UI

Run the main Streamlit app to launch the web UI for the assistant. This handles user login, profile setup, and all interactions.

```bash
streamlit run chatbot_2.py
```

  * The UI loads models on-demand to save memory.

#### \#\#\#\# Step 2: Agent Reasoning and Tool Use

When a user submits a query, **`chatbot_2.py`**:

1.  Detects the language and translates the input to English if necessary.
2.  Calls the agent pipeline in **`farmer_clean_agent.py`**, which performs a three-step process:
      * **`analyze_query_with_reasoning`**: Uses an LLM for chain-of-thought reasoning to decide which tools (Elasticsearch, Knowledge Graph, APIs, crop model) to use.
      * **`execute_tools_intelligently`**: Runs the selected tools and fetches the required data.
      * **`generate_intelligent_response`**: The LLM generates a final, coherent answer using all the retrieved context.
3.  Translates the final response back to the user's original language.

All tool logic for Elasticsearch, the Knowledge Graph, the crop model, weather, and market prices is located in **`agent_tools_clean.py`** and **`es_utils.py`**.

#### \#\#\#\# Step 3: Indexing and Data Preparation

Before using the chatbot for the first time, ensure **`index.py`** has been run to populate Elasticsearch with government schemes and their vector embeddings.

-----

## \#\# Usage

  * Text, speech, and image queries are supported.
  * The farmer's profile can be set up in the sidebar for personalized recommendations.
  * Model selection and memory management options are available in the sidebar.
  * Chat history and usage statistics are displayed in the main UI.

-----

## \#\# Troubleshooting

  * **Elasticsearch connection errors**: Ensure Elasticsearch is running at `localhost:9200` and the `schemes` index has been created.
  * **Neo4j errors**: Verify that the Neo4j service is running and accessible with the correct credentials.
  * **Model loading errors**: Check that the model paths are correct and that you have sufficient GPU/CPU memory.
  * **Large files**: Do not commit files larger than 100MB to git. Use **`.gitignore`** and consider Git LFS if needed.

-----

## \#\# Acknowledgements

  * [AI4Bharat IndicTrans2](https://www.google.com/search?q=https://github.com/AI4Bharat/indic-trans2)
  * [Qwen3-8B](https://huggingface.co/Qwen)
  * [Elasticsearch](https://www.elastic.co/)
  * [Neo4j](https://neo4j.com/)
  * [Streamlit](https://streamlit.io/)
  * [LangChain](https://www.langchain.com/)

For any issues, please open an issue on GitHub or contact the maintainers. Happy farming\! 🌾