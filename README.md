# KRISHI SAHAYAK: Multilingual Agricultural AI Assistant

<!-- Add a banner image or a screenshot of the application here -->
<!-- ![Project Banner](https://via.placeholder.com/800x200.png?text=Sahayak+Krishi+AI+Assistant) -->

**Sahayak Krishi** is an intelligent, multilingual AI assistant designed to empower farmers with immediate, data-driven insights. By integrating a state-of-the-art RAG pipeline, knowledge graphs, and multimodal input, it provides crucial information on government schemes, crop management, soil health, and market prices, breaking down language and literacy barriers.

---

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Scope of the Project](#scope-of-the-project)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---
## Setup and Installation

#### Prerequisites
- **Git:** To clone the repository.
- **Conda:** To manage the Python environment.
- **Docker:** To run the necessary database services. Please ensure the Docker daemon is running before you begin.

---
#### Step 1: Clone the repository and enter the directory:
This is a **one-time process**. The script will prepare everything your project needs to run.

  ```bash
  git clone <your-repo-url>
  cd <repository-name>
  ```

#### Step 2: Make the setup script executable and run it:
  ```bash
  chmod +x setup.sh
  ./setup.sh
  ```
  This script handles creating the Conda environment, installing all dependencies, and launching the required databases in Docker.

---

## Usage
1.  **Activate the environment:**
    ```bash
    conda activate cap_one
    ```
2.  **Run the application:**
    ```bash
    cd src
    streamlit run chatbot.py
    ```
3.  **Login Details:**
    ```
    username: admin
    password: admin
    ```

---
## Architecture
![Krishi Sahayak Architecture](https://github.com/manapureanshul7/Krishi-Sahayak/blob/main/Assets/Krishi%20Sahayak%20Architecture.png)

The application is built on a robust, three-tier architecture where a central agent orchestrates a suite of specialized tools to answer user queries. The workflow is designed to handle multimodal and multilingual input efficiently.

1.  **User Interface:** A user interacts with the system through a **Streamlit** frontend (`chatbot.py`), providing queries as either text or images.

2.  **Input Preprocessing:** The backend receives the input and standardizes it. Text is translated into a common language (English), while images are analyzed by specialized **Soil/Plant Disease classification models**.

3.  **Core Agent Logic:** The processed query is sent to the central agent (`main_agent.py`). This component acts as the "brain," using an **LLM (Qwen3LLM)** to reason about the user's intent and create a plan by selecting the appropriate tools.

4.  **Tool Execution:** The agent (`agent_tools.py`) executes the plan by calling a suite of tools that interface with various external services. These services include:
    * **Elasticsearch:** For Retrieval-Augmented Generation (RAG) and searching documents like government schemes.
    * **Neo4j Knowledge Graph:** For querying structured data and relationships between agricultural concepts.
    * **Live APIs:** For fetching real-time weather and market price data.

5.  **Response Generation & Delivery:** The agent gathers the data from the tools and uses the LLM to synthesize a single, coherent answer. This answer is then translated back into the user's original language and sent to the UI to be displayed.

---

## Features
- **Multimodal Interaction:** The system accepts user queries via **text**, **voice** (speech-to-text), and **image** inputs, making it accessible to farmers regardless of literacy or the nature of their query.
- **Full Multilingual Support:** Built to be inclusive, the agent supports **22 Indic languages** for all its interactions, handling translation for both incoming queries and outgoing responses.
- **Personalized Recommendations:** It creates and utilizes **farmer and land profiles** to deliver tailored advice. The core of this is a **Neo4j knowledge graph** that maps complex relationships between a farmer's unique context and relevant agricultural data.
- **AI-Powered Visual Diagnostics:** Farmers can upload images to:
  - **Diagnose Plant Diseases:** An integrated vision model analyzes photos of leaves to identify diseases.
  - **Classify Soil Type:** Another model analyzes soil images to determine its type.
- **Advanced RAG & Reasoning:** The agent uses a **Retrieval-Augmented Generation (RAG)** system powered by **Elasticsearch** and a **Qwen3-8B LLM**. This ensures that answers are not just generated, but are grounded in a factual knowledge base of expert agricultural data.

---

## Limitations of our Implementation
Here are some of the known limitations of the current implementation:

-   **Partial Knowledge Graph Integration:** While the foundational integration with the **Neo4j** graph database is complete, the agent's core reasoning workflow does not yet fully leverage its complex relational querying capabilities. The current prototype relies more heavily on the Elasticsearch-based RAG system for generating responses.

-   **Response Latency due to Dynamic Model Loading:** To support the execution of multiple large models (for vision, translation, and reasoning) on resource-constrained hardware, models are loaded into memory on-demand for each query and unloaded afterward. This design choice optimizes memory usage but introduces significant latency, as the loading/unloading time constitutes a major part of the total response time.

-   **Highly Specialized Scope:** The project's current knowledge base is specialized as a high-fidelity proof-of-concept. The scope is limited to:
    -   **Crop Type:** Paddy
    -   **Region:** Tamil Nadu, India
    The information ingested into Elasticsearch and Neo4j is primarily sourced from the TNAU Agritech Portal's expert system on paddy cultivation. Therefore, its recommendations are most accurate for this specific domain.

---

## Troubleshooting
- If for some case you aren't able to run our implementation due to .pth file error. Please refer to the zip file submitted. 

---

## Acknowledgements

We would like to express our sincere gratitude to the organizations, open-source communities, and data providers who made this project possible.

### Key Technologies & Libraries
Our work is built upon a foundation of powerful open-source software. We extend our thanks to the teams and contributors behind:
- **Frameworks & Orchestration:** LangChain, Streamlit, and LangGraph
- **Databases & Search:** Elasticsearch and Neo4j
- **AI/ML Models & Libraries:** Qwen3-8B, IndicTrans2, SentenceTransformers, and the PyTorch ecosystem.

### Data Sources & APIs
This project would not have been possible without access to high-quality data. We are grateful to the following sources:
- **Public Portals:** TNAU Agritech Portal, MyScheme, Indian Meteorological Department (IMD), and the Indian Council of Agricultural Research (ICAR).
- **Machine Learning Datasets:** The publicly available datasets on Kaggle for soil classification and rice plant disease detection.
- **Live Data:** The providers of the real-time Weather and Market Price APIs.

### Event & Support
Finally, we would like to thank the organizers and sponsors of the **Capital One Launchpad Hackathon** for providing the platform and opportunity to develop and showcase our solution. 🌾
