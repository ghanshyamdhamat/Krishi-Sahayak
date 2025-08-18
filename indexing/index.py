from elasticsearch import Elasticsearch
import json
import os
import numpy as np
import openai
import numpy as np
import time
import requests
from sentence_transformers import SentenceTransformer
import torch
import gc
import logging

def get_qwen_embedding(text: str) -> list:
    """Get embedding using Qwen3-Embedding-4B in 8-bit quantized mode."""


        # logger.info("✅ Loaded Qwen3-Embedding-4B in 8-bit quantized mode.")
    return qwen_model.encode(text, prompt_name="query").tolist()

# Initialize Elasticsearch client with version compatibility
try:
    # Fix for Elasticsearch client v9 with server v8
    es = Elasticsearch(
        hosts=["http://localhost:9200"],
        verify_certs=False,
        api_key=None,
        headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
    )
    
    print("Testing Elasticsearch connection...")
    info = es.info()
    print("✅ Elasticsearch connection successful!")
    print(f"Cluster: {info['cluster_name']}, Version: {info['version']['number']}")
    
except Exception as e:
    print(f"❌ Elasticsearch client v9 with headers failed: {e}")
    
    # Try downgrading client compatibility or use requests directly
    try:
        print("Trying alternative connection method...")
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        es = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False
        )
        
        # Test with a simple ping
        if es.ping():
            print("✅ Alternative connection successful!")
        else:
            raise ConnectionError("Ping failed")
            
    except Exception as e2:
        print(f"❌ Alternative connection failed: {e2}")
        
        # Final fallback: Use requests for Elasticsearch operations
        print("Using requests as final fallback...")
        try:
            response = requests.get("http://localhost:9200")
            if response.status_code == 200:
                print("✅ HTTP fallback successful - will use requests for Elasticsearch operations")
                # Create a simple wrapper for basic operations
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
                    
                    def index(self, index, body=None, id=None, document=None):
                        # Handle both 'body' and 'document' parameters
                        data = document if document is not None else body
                        url = f"{self.base_url}/{index}/_doc"
                        if id:
                            url += f"/{id}"
                        resp = requests.post(url, json=data)
                        return resp.json()
                    
                    def search(self, index, body, **kwargs):  # Accept extra kwargs
                        url = f"{self.base_url}/{index}/_search"
                        resp = requests.post(url, json=body)
                        return resp.json()
                    
                    class IndicesClient:
                        def __init__(self, base_url):
                            self.base_url = base_url
                        
                        def exists(self, index):
                            try:
                                resp = requests.head(f"{self.base_url}/{index}")
                                return resp.status_code == 200
                            except:
                                return False
                        
                        def delete(self, index):
                            resp = requests.delete(f"{self.base_url}/{index}")
                            return resp.json() if resp.status_code != 404 else {"acknowledged": True}
                        
                        def create(self, index, body):
                            resp = requests.put(f"{self.base_url}/{index}", json=body)
                            return resp.json()
                
                es = SimpleElasticsearch("http://localhost:9200")
                info = es.info()
                print(f"Fallback connection info: {info.get('cluster_name', 'N/A')}")
            else:
                raise ConnectionError("HTTP fallback failed")
        except Exception as e3:
            print(f"❌ All connection methods failed: {e3}")
            raise ConnectionError("Failed to connect to Elasticsearch with any method")

# Final connection verification
if not es.ping():
    raise ConnectionError("Failed to establish connection to Elasticsearch")

# Load schemes data with corrected path
schemes_path = 'indexing/enhanced_schemes.json'
if not os.path.exists(schemes_path):
    # Try alternative paths
    alternative_paths = [
        './enhanced_schemes.json'
    ]
    
    for path in alternative_paths:
        if os.path.exists(path):
            schemes_path = path
            break
    else:
        raise FileNotFoundError(f"schemes.json not found. Looked in: {alternative_paths}")

print(f"Loading schemes from: {schemes_path}")
with open(schemes_path, 'r') as file:
    schemes = json.load(file)

print(f"Loaded {len(schemes)} schemes from JSON")

client = openai.OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key="dea915c6-756a-4db4-889a-f539683a70ec",
)

def get_sambanova_embedding(text):
    resp = client.embeddings.create(
        model="E5-Mistral-7B-Instruct",
        input=text
    )
    return np.array(resp.data[0].embedding)

# Create index with mappings
index_name = "schemes"

mappings = {
    "mappings": {
        "properties": {
            "scheme_id": {"type": "keyword"},
            "name": {"type": "text"},
            "scheme type": {"type": "text"},
            "funding source": {"type": "text"},
            "purpose or objective of scheme": {"type": "text"},
            "when scheme started or launched": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM||yyyy||strict_date_optional_time||epoch_millis"},
            "Scheme validity period": {"type": "text"},
            "Scheme coverage area": {"type": "text"},
            "Crops covered": {"type": "text"},
            "benefits": {"type": "text"},  # JSON string or object, so keep as text
            "eligibility": {"type": "text"},  # JSON string or object, so keep as text
            "excluded or ineligibility": {"type": "text"},  # JSON string or array, so keep as text
            "steps to apply": {"type": "text"},  # JSON string or object, so keep as text
            "documents required": {"type": "text"},  # JSON string or array, so keep as text
            "other_details": {"type": "text"},  # JSON string or object, so keep as text
            "references or sources": {"type": "text"},  # JSON string or array, so keep as text
            "last_updated": {"type": "date", "format": "yyyy-MM-dd"},
            "metadata": {
                "type": "object",
                "properties": {
                    "category": {"type": "keyword"},
                    "subcategory": {"type": "keyword"},
                    "target_group": {"type": "text"},
                    "voluntary": {"type": "boolean"},
                    "state_specific": {"type": "boolean"},
                    "status": {"type": "keyword"},
                    "launch_year": {"type": "integer"},
                    "description": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "tags": {"type": "keyword"}
                }
            },
            "embedding_text": {"type": "text"},
            "benefits_currency": {"type": "keyword"},
            "benefits_amount": {"type": "float"},
            "benefits_disbursement_mode": {"type": "keyword"},
            "eligibility_aadhaar_required": {"type": "boolean"},
            "eligibility_bank_account_required": {"type": "boolean"},
            "eligibility_voluntary": {"type": "boolean"}
        }
    }
}

# delete the index if exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Create index if it does not exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mappings)

# Index documents with embeddings, skipping duplicates already in Elasticsearch
gc.collect()
torch.cuda.empty_cache()
qwen_model =  SentenceTransformer(
            "Qwen/Qwen3-Embedding-4B",
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            model_kwargs={
                "load_in_8bit": True,
                "device_map": "auto"
            }
        )
for scheme in schemes:
    scheme_id = scheme.get("scheme_id")
    if not scheme_id:
        print(f"⚠️ Skipping scheme with missing scheme_id")
        continue

    # Check if scheme already exists in Elasticsearch
    exists_resp = es.search(
        index=index_name,
        body={
            "query": {
                "term": {
                    "scheme_id": scheme_id
                }
            },
            "size": 1
        }
    )
    if exists_resp.get("hits", {}).get("total", {}).get("value", 0) > 0:
        print(f"⚠️ Skipping already present scheme_id: {scheme_id}")
        continue

    try:
        if "embedding_text" in scheme:
            embedding = get_qwen_embedding(scheme["embedding_text"])
            scheme["embedding_vector"] = embedding
        result = es.index(index=index_name, id=scheme_id, document=scheme)
        print(f"Indexed {scheme_id}: {result}")
    except Exception as e:
        print(f"Error indexing {scheme_id}: {e}")
    gc.collect()
    torch.cuda.empty_cache()

print("Indexing complete.")

