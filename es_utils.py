"""
Elasticsearch utility functions for RAG integration with the chatbot
"""

import logging
from typing import List, Dict, Any, Optional
import requests
from sentence_transformers import SentenceTransformer
import openai
import torch
import gc
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleElasticsearch:
    """Simple Elasticsearch client using requests"""
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
    
    def search(self, index, body, **kwargs):
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

class ElasticsearchRAG:
    """
    Elasticsearch Retrieval Augmented Generation utility class
    Provides vector search and retrieval functions for integration with chatbot
    """
    
    def __init__(self, es_url="http://localhost:9200", index_name="schemes"):
        """Initialize Elasticsearch connection"""
        self.index_name = index_name
        self.es_url = es_url
        
        # Initialize SimpleElasticsearch client
        self.es = SimpleElasticsearch(es_url)
        
        # Test connection
        if not self.es.ping():
            raise ConnectionError(f"Failed to connect to Elasticsearch at {es_url}")
        
        logger.info("✅ Elasticsearch connection successful!")
        
        # Model components for lazy loading
        self.minilm_model = None
        self.qwen_model = None
        self.sambanova_client = None

    def get_minilm_embedding(self, text: str) -> list:
        """Get embedding using MiniLM model."""
        if self.minilm_model is None:
            self.minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.minilm_model.encode(text).tolist()

    # def get_qwen_embedding(self, text: str) -> list:
    #     """Get embedding using Qwen3 model."""
    #     import gc
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     if self.qwen_model is None:
    #         self.qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    #     return self.qwen_model.encode(text, prompt_name="query").tolist()

    def get_qwen_embedding(self, text: str, quantized: bool = True) -> list:
        """Get embedding using quantized Qwen3 model if quantized=True."""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if self.qwen_model is None:
            from sentence_transformers import SentenceTransformer
            if quantized:
                try:
                    # Try to load quantized (8-bit) version if supported
                    self.qwen_model = SentenceTransformer(
                        "Qwen/Qwen3-Embedding-4B",
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        trust_remote_code=True,
                        # Pass quantization config to underlying transformers model
                        model_kwargs={
                            "load_in_8bit": True,
                            "device_map": "auto"
                        }
                    )
                    logger.info("✅ Loaded Qwen3-Embedding-4B in 8-bit quantized mode.")
                except Exception as e:
                    logger.warning(f"Could not load quantized model, falling back to full precision: {e}")
                    self.qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
            else:
                self.qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
        return self.qwen_model.encode(text, prompt_name="query", output_dim = 1024).tolist()

    def get_sambanova_embedding(self, text: str, model: str = "E5-Mistral-7B-Instruct", api_key: str = None) -> list:
        """Get embedding from Sambanova API."""
        if self.sambanova_client is None:
            self.sambanova_client = openai.OpenAI(
                base_url="https://api.sambanova.ai/v1",
                api_key=api_key or "feaa7821-5089-48be-8373-5644e525a9e1",
            )
        response = self.sambanova_client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    def get_embedding(self, text: str, model: str = "qwen", api_key: str = None) -> list:
        """Unified embedding getter."""
        if model == "minilm":
            return self.get_minilm_embedding(text)
        elif model == "qwen":
            return self.get_qwen_embedding(text)
        elif model == "sambanova":
            return self.get_sambanova_embedding(text, api_key=api_key)
        else:
            raise ValueError(f"Unknown embedding model: {model}")

    def vector_search(self, query: str, top_k: int = 3, search_type: str = None, model="qwen", index_name: Optional[str] = None, query_embedding: Optional[list] = None) -> List[Dict[str, Any]]:
        """
        Search Elasticsearch using vector similarity and text matching
        """
        index = index_name if index_name else self.index_name

        # Define mapping configs per index
        index_configs = {
            "scheme_metadata_filter": {
                "vector_field": "sub_category_embedding",
                "text_fields": ["category^2", "subcategory^2"],
                "embedding_model": "minilm"  # use MiniLM for metadata
            },
            "schemes": {
                "vector_field": "embedding_vector",
                "text_fields": ["name^3", "objective^2", "metadata.description^2", "embedding_text"],
                "embedding_model": model  # use selected model
            },
            "crops": {
                "vector_field": "crop_vector",
                "text_fields": ["crop_name^3", "description^2"],
                "embedding_model": model
            },
            "diseases": {
                "vector_field": "disease_vector",
                "text_fields": ["disease_name^3", "symptoms^2", "description^2"],
                "embedding_model": model
            }
        }

        if index not in index_configs:
            logger.error(f"No mapping config found for index '{index}'")
            return []

        config = index_configs[index]

        # Get embeddings using the correct model
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = self.get_embedding(query, config["embedding_model"])

        # Ensure embedding is a list for JSON serialization
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        elif isinstance(embedding, (tuple, set)):
            embedding = list(embedding)

        if embedding is None or (hasattr(embedding, "__len__") and len(embedding) == 0):
            logger.warning("Failed to generate embeddings for query")
            return []

        # Build ES query
        es_query = {
            "size": top_k,
            "knn": {
                "field": config["vector_field"],
                "query_vector": embedding,
                "k": top_k,
                "num_candidates": top_k * 20
            },
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": config["text_fields"],
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            }
        }

        try:
            results = self.es.search(index=index, body=es_query)
            hits = results.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}")
            return []


    def hybrid_search_with_metadata(self, query_text, top_k: int = 3, index_name="schemes", metadata_list=None, model="qwen"):
        # Step 1: Create metadata filter
        scheme_ids = [m["scheme_id"] for m in metadata_list] if metadata_list else []
        query_embedding = self.get_embedding(query_text, model)
        index = index_name if index_name else self.index_name
        if not query_embedding:
            logger.warning("Failed to generate embeddings for query")
            return []
        # Step 2: Hybrid search: filter by metadata, then combine vector + text
        index_configs = {
            "schemes": {
                "vector_field": "embedding_vector",
                "text_fields": ["name^3", "objective^2", "metadata.description^2", "embedding_text"]
            },
            "crops": {
                "vector_field": "crop_vector",
                "text_fields": ["crop_name^3", "description^2"]
            },
            "diseases": {
                "vector_field": "disease_vector",
                "text_fields": ["disease_name^3", "symptoms^2", "description^2"]
            }
        }

        if index not in index_configs:
            logger.error(f"No mapping config found for index '{index}'")
            return []

        config = index_configs[index]
        es_query = {
            "size": top_k,
            "knn": {
                "field": config["vector_field"],
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": 100
            },
            "query": {
                "bool": {
                    "should": [
                        # Text search with fuzziness
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": config["text_fields"],
                                "fuzziness": "AUTO"
                            }
                        },
                        # Metadata match (optional boost)
                        {
                            "terms": {
                                "scheme_id": scheme_ids
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        response = self.es.search(index=index_name, body=es_query)
        if "hits" not in response:
            logger.error(f"Unexpected Elasticsearch response: {response}")
            return []
        return response["hits"]["hits"]
    
    def farming_search(self, query_text, top_k=5, model="qwen", index_name=None, query_embedding=None):
        """
        Hybrid search for farming info: combines embedding similarity and keyword match.
        """
        index_name = index_name
        if index_name is None:
            return []

        # Compute embedding if not provided
        if query_embedding is None:
            query_embedding = self.get_embedding(query_text, model=model)

        # Elasticsearch dense vector query (ANN) + keyword match
        es_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'chunk_embedding') + 1.0",
                                    "params": {"query_vector": query_embedding.tolist()}
                                }
                            }
                        },
                        {
                            "match": {
                                "chunk_text": {
                                    "query": query_text,
                                    "fuzziness": "AUTO"
                                }
                            }
                        }
                    ]
                }
            }
        }

        try:
            response = self.es.search(index=index_name, body=es_query)
            # logger.info(f"Elasticsearch farming_search response: {response}")
            hits = response.get("hits", {}).get("hits", [])
            return hits
        except Exception as e:
            print(f"Elasticsearch farming_search error: {e}")
            return []
   
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM context"""
        if not results:
            return "No relevant information found."
        
        formatted_context = "I found the following agricultural schemes that may be relevant to your query:\n\n"
        
        for i, doc in enumerate(results, 1):
            formatted_context += f"{i}. {doc.get('name', 'Unnamed Scheme')}\n"
            
            if 'objective' in doc and doc['objective']:
                formatted_context += f"   Objective: {doc['objective']}\n"
                
            if 'benefits' in doc and doc['benefits']:
                formatted_context += f"   Benefits: {doc['benefits']}\n"
                
            if 'eligibility' in doc and doc['eligibility']:
                formatted_context += f"   Eligibility: {doc['eligibility']}\n"
            
            formatted_context += "\n"
            
        return formatted_context
    
    def generate_rag_prompt(self, query: str) -> str:
        """
        Generate RAG prompt with retrieved context
        
        Args:
            query: User's original query
            
        Returns:
            Prompt for LLM with RAG context
        """
        # Search for relevant documents
        search_results = self.vector_search(query, top_k=3)
        
        # Format results as context
        context = self.format_search_results(search_results)
        
        # Create RAG prompt
        rag_prompt = f"""Context information:
{context}

Using the context information provided above, please answer the following question:
{query}

If the context doesn't contain the information needed to answer the question, say so clearly
and provide general information about agricultural schemes or farming practices that might be helpful.
"""
        return rag_prompt
