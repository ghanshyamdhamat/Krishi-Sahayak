#!/bin/bash

# This script is designed for Linux and macOS environments.
set -e

# --- Color Definitions for better readability ---
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SAHAYAK KRISHI: Automated Setup Script ===${NC}"

# --- 1. Prerequisites Check ---
echo -e "\n${YELLOW}[1/8] Checking for prerequisites (Conda and Docker)...${NC}"

# Check for Conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: Conda is not installed or not in your PATH. Please install Anaconda/Miniconda first.${NC}"
    exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed. Please install and start the Docker daemon to continue.${NC}"
    exit 1
fi
echo -e "${GREEN}Prerequisites are satisfied.${NC}"


# --- 2. Create and activate conda environment ---
echo -e "\n${YELLOW}[2/8] Creating conda environment 'krishi_sahayak' with Python 3.10...${NC}"
conda create -y -n krishi_sahayak python=3.10

echo -e "\n${YELLOW}[3/8] Activating conda environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate krishi_sahayak


# --- 3. Install Python dependencies ---
echo -e "\n${YELLOW}[4/8] Installing Python dependencies...${NC}"
if [ -f src/requirements.txt ]; then
    pip install -r src/requirements.txt
else
    echo -e "${RED}ERROR: src/requirements.txt not found!${NC}"
    exit 1
fi


# --- 4. Start Elasticsearch with Docker ---
echo -e "\n${YELLOW}[5/8] Starting Elasticsearch Docker container...${NC}"
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.13.4

echo "Waiting for Elasticsearch to start..."
until curl -s http://localhost:9200 >/dev/null; do
    sleep 3
done
echo -e "${GREEN}Elasticsearch is up!${NC}"


# --- 5. Index data into Elasticsearch ---
echo -e "\n${YELLOW}[6/8] Indexing data into Elasticsearch...${NC}"
# NOTE: Assumes your indexing scripts are inside 'src/core/elastic_search/scripts/'
cd src/core/elastic_search/scripts
python index.py || true
python indexing_pdf_data.py || true
python indexing_metadata_schemes.py || true
python elasticsearch_ingest_paddy.py || true
cd ../../../  # Return to the project root


# --- 6. Start Neo4j with Docker ---
echo -e "\n${YELLOW}[7/8] Starting Neo4j Docker container...${NC}"
docker run -d --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test1234 neo4j:4.4

echo "Waiting for Neo4j to start..."
until curl -s http://localhost:7474 >/dev/null; do
    sleep 3
done
echo -e "${GREEN}Neo4j is up!${NC}"


# --- 7. Ingest data into Neo4j Knowledge Graph ---
echo -e "\n${YELLOW}[8/8] Ingesting data into Neo4j Knowledge Graph...${NC}"
cd src/neo4j_files
python store_pdf_data.py || true
cd ../..  # Return to the project root


# --- Final Instructions ---
echo -e "\n${GREEN}=== ✅ Setup Complete! ===${NC}"
echo "To start the assistant UI, run these commands:"
echo -e "  ${YELLOW}conda activate krishi_sahayak${NC}"
echo -e "  ${YELLOW}cd src${NC}"
echo -e "  ${YELLOW}streamlit run chatbot.py${NC}"