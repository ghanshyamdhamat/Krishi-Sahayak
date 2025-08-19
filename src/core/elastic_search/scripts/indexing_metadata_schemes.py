import json
from elasticsearch import Elasticsearch
from pathlib import Path
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# from /mnt/bb586fde-943d-4653-af27-224147bfba7e/Capital_One/capital_one_agent_ai/backend/enhanced_schemes.json 
input_path = Path("src/core/elastic_search/data/schemes.json")
schemes_metadata_subcategory=[]
schemes_metadata_category=[]
scheme_ids = []
with input_path.open("r",encoding="utf-8") as f:
    schemes = json.load(f)
    for scheme in schemes:
        scheme_id = scheme.get("scheme_id")
        scheme_ids.append(scheme_id) 
        meta = scheme.get("metadata",{})
        if "category" in meta:
            schemes_metadata_category.append(meta.get("category",""))
        if "subcategory" in meta:
            schemes_metadata_subcategory.append(meta.get("subcategory",""))


print(scheme_ids)
index_name = "scheme_metadata_filter"
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
                    
                    def search(self, index, body):
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

# Delete the entire index if it already exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Deleted existing index: {index_name}")

# (Re)create the index if needed (optional: add mappings/settings here)
try:
    es.indices.create(index=index_name)
    print(f"Created index: {index_name}")
except Exception as e:
    print(f"Index creation error (may already exist): {e}")

# Prepare unique (scheme_id, category, subcategory) tuples
unique_entries = set()
for scheme in schemes:
    scheme_id = scheme.get("scheme_id")
    meta = scheme.get("metadata", {})
    category = meta.get("category", "")
    subcategory = meta.get("subcategory", "")
    if scheme_id:
        unique_entries.add((scheme_id, category, subcategory))

# Load MiniLM model from sentence-transformers
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")  # or another MiniLM variant

def get_minilm_embedding_st(text):
    embedding = minilm_model.encode(text)
    return embedding

# Index each unique entry into Elasticsearch, skipping duplicate scheme_ids
indexed_scheme_ids = set()
for scheme_id, category, subcategory in unique_entries:
    if scheme_id in indexed_scheme_ids:
        print(f"Duplicate skipped: {scheme_id}")
        continue

    # Get embedding for subcategory (empty string if missing)
    subcat_text = subcategory if subcategory else ""
    try:
        sub_category_embedding = get_minilm_embedding_st(subcat_text).tolist()
    except Exception as e:
        print(f"Embedding error for '{subcat_text}': {e}")
        sub_category_embedding = []

    doc = {
        "scheme_id": scheme_id,
        "category": category,
        "subcategory": subcategory,
        "sub_category_embedding": sub_category_embedding
    }
    es.index(index=index_name, body=doc)
    print(f"Indexed: {scheme_id}, category: {category}, subcategory: {subcategory}")
    indexed_scheme_ids.add(scheme_id)
# print(set(schemes_metadata_subcategory))