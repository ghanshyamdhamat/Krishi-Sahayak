import re
import requests
from elasticsearch import Elasticsearch, helpers
import torch
from sentence_transformers import SentenceTransformer

# --- Elasticsearch connection with fallback ---
try:
    es = Elasticsearch(
        hosts=["http://localhost:9200"],
        verify_certs=False,
        # Remove headers param if you get version errors!
    )
    print("Testing Elasticsearch connection...")
    info = es.info()
    print("✅ Elasticsearch connection successful!")
    print(f"Cluster: {info['cluster_name']}, Version: {info['version']['number']}")
    USE_HELPERS = True
except Exception as e:
    print(f"❌ Elasticsearch client failed: {e}")
    print("Using fallback SimpleElasticsearch client...")

    class SimpleElasticsearch:
        def __init__(self, base_url):
            self.base_url = base_url
            self.indices = self.IndicesClient(base_url)

        def ping(self):
            try:
                resp = requests.get(f"{self.base_url}")
                return resp.status_code == 200
            except:
                return False

        def info(self):
            resp = requests.get(f"{self.base_url}")
            return resp.json()

        def index(self, index, document, id=None):
            url = f"{self.base_url}/{index}/_doc"
            if id:
                url += f"/{id}"
            resp = requests.post(url, json=document, headers={"Content-Type": "application/json"})
            print("Index response:", resp.status_code, resp.text)  # Add this for debugging
            return resp.json()

        def search(self, index, body):
            url = f"{self.base_url}/{index}/_search"
            resp = requests.post(url, json=body)
            return resp.json()

        class IndicesClient:
            def __init__(self, base_url):
                self.base_url = base_url

            def exists(self, index):
                resp = requests.head(f"{self.base_url}/{index}")
                return resp.status_code == 200

            def create(self, index, body):
                resp = requests.put(f"{self.base_url}/{index}", json=body)
                return resp.json()

            def delete(self, index):
                resp = requests.delete(f"{self.base_url}/{index}")
                return resp.json()

    es = SimpleElasticsearch("http://localhost:9200")
    USE_HELPERS = False

if not es.ping():
    raise ConnectionError("❌ Failed to establish connection to Elasticsearch")

# --- Embedding model ---
qwen_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    device="cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
    model_kwargs={
        "load_in_8bit": True,
        "device_map": "auto"
    }
)
model = qwen_model  # Use this model everywhere below
INDEX_NAME = "paddy"
EMBED_DIM = 2560  # Qwen3-Embedding-4B output dim is 1024

mapping = {
    "mappings": {
        "properties": {
            "section": {"type": "keyword"},
            "subsection": {"type": "keyword"},
            "text": {"type": "text"},
            "section_embedding": {"type": "dense_vector", "dims": EMBED_DIM},
            "subsection_embedding": {"type": "dense_vector", "dims": EMBED_DIM},
            "text_embedding": {"type": "dense_vector", "dims": EMBED_DIM}
        }
    }
}

def create_index_if_not_exists(es, index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"✅ Created index: {index_name}")

# Delete the index if it already exists
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
    print(f"🗑️ Deleted existing index: {INDEX_NAME}")

# --- Read chunks.txt ---
chunks = []
with open("../data/chunks.txt", "r", encoding="utf-8") as f:
    chunk = {}
    for line in f:
        if line.startswith("chunksize"):
            if chunk:
                chunks.append(chunk)
            chunk = {"section": None, "subsection": None, "text": ""}
        elif line.startswith("section:"):
            chunk["section"] = line[len("section:"):].strip()
        elif line.startswith("subsection:"):
            chunk["subsection"] = line[len("subsection:"):].strip()
        elif line.strip() == "":
            continue
        else:
            chunk["text"] += line
    if chunk:
        chunks.append(chunk)

# --- Prepare documents ---
docs = []
for idx, chunk in enumerate(chunks):
    section_text = chunk["section"] if chunk["section"] and chunk["section"].lower() != "null" else "None"
    subsection_text = chunk["subsection"] if chunk["subsection"] and chunk["subsection"].lower() != "null" else "None"
    text = chunk["text"].strip()

    # For tables, flatten for embedding
    if text.startswith("**Table") or text.startswith('|'):
        lines = text.splitlines()
        title = lines[0] if lines[0].startswith("**Table") else ""
        table_rows = [l for l in lines if l.startswith('|')]
        embed_text = title + " " + " ".join(table_rows)
    else:
        embed_text = text

    doc = {
        "section": section_text,
        "subsection": subsection_text,
        "text": text,
        "text_embedding": model.encode(embed_text).tolist()
    }
    doc["_id"] = idx
    docs.append(doc)

# --- Index documents ---
create_index_if_not_exists(es, INDEX_NAME)

if USE_HELPERS:
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": doc["_id"],
            "_source": {k: v for k, v in doc.items() if k != "_id"}
        }
        for doc in docs
    ]
    helpers.bulk(es, actions)
else:
    for doc in docs:
        es.index(index=INDEX_NAME, document={k: v for k, v in doc.items() if k != "_id"}, id=doc["_id"])

print(f"✅ Indexed {len(docs)} chunks into '{INDEX_NAME}'")

if hasattr(es, 'indices') and hasattr(es.indices, 'refresh'):
    es.indices.refresh(index=INDEX_NAME)

query = {
    "query": {
        "query_string": {
            "query": "Oryza sativa",
            "fields": ["text"]
        }
    },
    "size": 5
}
results = es.search(index=INDEX_NAME, body=query)
for hit in results["hits"]["hits"]:
    print("Section:", repr(hit["_source"]["section"]))
    print("Text:", hit["_source"]["text"][:300], "\n")

sections = set()
for hit in results["hits"]["hits"]:
    sections.add(hit["_source"]["section"])
print("Unique sections:", sections)

results = es.search(index=INDEX_NAME, body={"query": {"match_all": {}}, "size": 10})
for hit in results["hits"]["hits"]:
    print(hit["_source"])
