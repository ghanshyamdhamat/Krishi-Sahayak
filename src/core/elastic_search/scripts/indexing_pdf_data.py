import json
from pathlib import Path
import requests
from sentence_transformers import SentenceTransformer
import gc
from torch.cuda import empty_cache
import sys

# ---- Config ----
# Assuming the JSON file is in the same directory as the script for easy execution.
# Adjust the path if necessary.
INPUT_PATH = Path("src/core/elastic_search/data/paddy_data2_extractions.json")
# Elasticsearch index names are conventionally lowercase.
INDEX_NAME = "farming_kb"
# This dimension MUST match the output dimension of your model. 1024 is correct for Qwen.
EMBEDDING_DIM = 2560

# ---- SimpleElasticsearch Wrapper with Error Handling ----

def _check_response(resp):
    """Checks if the request was successful and raises an error with details if not."""
    # A status code of 400 or higher indicates an error.
    if resp.status_code >= 400:
        # Try to parse the JSON error response from Elasticsearch.
        try:
            error_details = resp.json()
        except json.JSONDecodeError:
            error_details = resp.text
        
        print(f"❌ Elasticsearch API Error (Status Code: {resp.status_code})", file=sys.stderr)
        # Pretty-print the JSON error details for readability.
        print(json.dumps(error_details, indent=2), file=sys.stderr)
        
        # This will raise an HTTPError, stopping the script.
        resp.raise_for_status()
        
    return resp.json()


class IndicesClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def exists(self, index):
        try:
            resp = requests.head(f"{self.base_url}/{index}")
            return resp.status_code == 200
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error checking index existence: {e}", file=sys.stderr)
            return False

    def delete(self, index):
        resp = requests.delete(f"{self.base_url}/{index}")
        # A 404 (Not Found) is acceptable when deleting, as the index is already gone.
        return resp.json() if resp.status_code != 404 else {"acknowledged": True}

    def create(self, index, body):
        resp = requests.put(f"{self.base_url}/{index}", json=body)
        # FIXED: Added response validation
        return _check_response(resp)


class SimpleElasticsearch:
    def __init__(self, base_url):
        self.base_url = base_url
        self.indices = IndicesClient(base_url)

    def ping(self):
        try:
            requests.get(f"{self.base_url}", timeout=2)
            return True
        except requests.exceptions.ConnectionError:
            return False

    def info(self):
        resp = requests.get(f"{self.base_url}")
        return _check_response(resp)

    def index(self, index, body=None, id=None, document=None):
        data = document if document is not None else body
        
        # FIXED: Use PUT for user-specified IDs, POST for auto-generated IDs.
        if id:
            # PUT creates or replaces a document at a specific ID.
            url = f"{self.base_url}/{index}/_doc/{id}"
            resp = requests.put(url, json=data)
        else:
            # POST is for letting Elasticsearch generate the ID.
            url = f"{self.base_url}/{index}/_doc"
            resp = requests.post(url, json=data)
            
        # FIXED: Added response validation
        return _check_response(resp)
        
    def search(self, index, body):
        url = f"{self.base_url}/{index}/_search"
        resp = requests.post(url, json=body)
        return _check_response(resp)

# ---- Main Indexing Logic ----
def main():
    # Load data
    if not INPUT_PATH.exists():
        print(f"❌ Error: Input file not found at '{INPUT_PATH}'. Please ensure the file exists.", file=sys.stderr)
        sys.exit(1) # Exit the script if the data file is missing.
        
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        paddy_chunks = json.load(f)

    # Connect to Elasticsearch
    es = SimpleElasticsearch("http://localhost:9200")
    
    if not es.ping():
        raise ConnectionError("❌ Failed to connect to Elasticsearch at http://localhost:9200. Is it running?")

    info = es.info()
    print(f"✅ Connected to Elasticsearch cluster: {info.get('cluster_name', 'N/A')}")

    # Delete and recreate index with mapping for dense vector
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"🗑️ Deleted existing index: {INDEX_NAME}")

    mapping = {
        "mappings": {
            "properties": {
                "source_file": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_text": {"type": "text"},
                "chunk_classification": {"type": "keyword"},
                # It's good practice to disable indexing on arbitrary object fields unless you need to search inside them.
                "extractions": {"type": "object", "enabled": False},
                "chunk_embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM, # Must be 1024 for this model
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    try:
        response = es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"📄 Created index '{INDEX_NAME}' with response: {response}")
    except Exception as e:
        print(f"❌ Index creation failed. Please check the mapping and Elasticsearch logs.", file=sys.stderr)
        # If index creation fails, we cannot proceed.
        sys.exit(1)

    # Load original embedding model
    print(f"\n🧠 Loading embedding model 'Qwen/Qwen3-Embedding-4B'...")
    gc.collect()
    empty_cache()
    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-4B", trust_remote_code=True)

    def get_embedding(text: str):
        # Using the model-specific parameters for encoding
        return embedder.encode(text, prompt_name="query", output_dim=EMBEDDING_DIM).tolist()

    # Index documents
    print("\n🚀 Starting document indexing...")
    for i, chunk in enumerate(paddy_chunks):
        chunk_text = chunk.get("chunk_text", "")
        # Skip chunks that have no text content
        if not chunk_text.strip():
            print(f"⚠️ Skipping chunk {chunk.get('chunk_index')} due to empty text.")
            continue

        try:
            chunk_embedding = get_embedding(chunk_text)
        except Exception as e:
            print(f"❌ Embedding error for chunk {chunk.get('chunk_index')}: {e}", file=sys.stderr)
            continue # Skip to the next chunk

        doc = {
            "source_file": chunk.get("source_file", ""),
            "chunk_index": chunk.get("chunk_index"),
            "chunk_text": chunk_text,
            "chunk_classification": chunk.get("chunk_classification", ""),
            "extractions": chunk.get("extractions", {}),
            "chunk_embedding": chunk_embedding
        }
        
        try:
            # Create a unique and repeatable ID from the source file and chunk index
            doc_id = f"{doc['source_file']}_{doc['chunk_index']}"
            res = es.index(index=INDEX_NAME, id=doc_id, body=doc)
            print(f"  -> Indexed chunk {i+1}/{len(paddy_chunks)} (ID: {res.get('_id')})")
        except Exception:
            # The error is already printed by _check_response, so we just note the failure and continue.
            print(f"❌ Failed to index chunk {chunk.get('chunk_index')}. See error above.", file=sys.stderr)

    print("\n✅ Finished indexing paddy data to Elasticsearch.")

if __name__ == "__main__":
    main()