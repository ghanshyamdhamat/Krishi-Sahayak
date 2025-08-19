"""
Clean Tools and utilities for the farmer agent system
Lightweight version without heavy model loading for query analysis
"""
import requests
import os
import json
import logging
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import torch
import pandas as pd
from neo4j_files.knowlwdge_graph_service import get_kg_service, KGResult

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the ElasticsearchRAG utility
try:
    from es_utils import ElasticsearchRAG
    ES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ElasticsearchRAG not available: {e}")
    ES_AVAILABLE = False

class FarmerTools:
    _embedding_model = None  # Class-level cache

    @staticmethod
    def get_embedding_model():
        if FarmerTools._embedding_model is None:
            FarmerTools._embedding_model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-4B",
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                model_kwargs={
                    "load_in_8bit": True,
                    "device_map": "auto"
                }
            )
        return FarmerTools._embedding_model

    """
    Collection of tools that can be used by the agent to help farmers
    Lightweight version focused on API calls and search without heavy model loading
    """
    def __init__(self):
        """Initialize tools and clients"""
        # Initialize Elasticsearch RAG if available
        self.kg_service = get_kg_service()
        self.kg_available = self.kg_service is not None
        
        if self.kg_available:
            logger.info("✅ Knowledge Graph service initialized")
        else:
            logger.warning("⚠️ Knowledge Graph service not available")

        if ES_AVAILABLE:
            try:
                self.es_rag = ElasticsearchRAG()
                self.es_available = True
                logger.info("✅ ElasticsearchRAG initialized successfully")
            except Exception as e:
                logger.error(f"❌ ElasticsearchRAG initialization failed: {e}")
                self.es_available = False
        else:
            self.es_available = False

        # Add crop model initialization
        self.crop_model = None
        self.crop_model_features = None
        self.crop_model_path = os.getenv("CROP_MODEL_PATH", 
            "./model/crop_recommendation.pkl"
        )
        self._load_crop_model()
        
        logger.info("✅ FarmerTools initialized (lightweight version)")

        #keys in ES indexes
        self.keys_schemes = [
            "scheme_id",
            "name",
            "scheme type",
            "funding source",
            "purpose or objective of scheme",
            "when scheme started or launched",
            "Scheme validity period",
            "Scheme coverage area",
            "Crops covered",
            "benefits",
            "eligibility",
            "excluded or ineligibility",
            "steps to apply",
            "documents required",
            "other_details"
        ]
        self.keys_paddy = []
        
        logger.info("✅ FarmerTools initialized (lightweight version)")

    def search_knowledge_graph_intelligent(self, query: str, farmer_profile: Dict = None, top_k: int = 3) -> List[Dict]:
        """
        Intelligent knowledge graph search using query planning
        """
        if not self.kg_available:
            logger.warning("Knowledge Graph not available")
            return []
        
        try:
             # Create farmer profile in KG if farmer_id exists and not already created
            if farmer_profile and farmer_profile.get('id'):
                self.kg_service.create_farmer_profile_in_kg(farmer_profile)
            
            # Use intelligent search with farmer context
            kg_results = self.kg_service.intelligent_search(query, farmer_profile, top_k)
            
            # Convert to format compatible with existing RAG pipeline
            formatted_results = []
            for result in kg_results:
                formatted_results.append({
                    "content": result.content,
                    "source": result.source,
                    "confidence": result.confidence,
                    "type": result.metadata.get('type', 'knowledge_graph'),
                    "metadata": result.metadata,
                    "search_strategy": result.metadata.get('type', 'unknown')
                })
            
            logger.info(f"✅ Intelligent KG search found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in intelligent KG search: {e}")
            return []
        
    def store_farmer_interaction(self, farmer_id: str, query: str, response: str, 
                           problem_type: str = None):
        """Store farmer's interaction for learning"""
        if not self.kg_available or not farmer_id:
            return
        
        try:
            self.kg_service.store_farmer_interaction(farmer_id, query, response, problem_type)
        except Exception as e:
            logger.error(f"Error storing farmer interaction: {e}")

    def get_farmer_context_for_query(self, farmer_id: str) -> Dict:
        """Get farmer's historical context to inform current query"""
        if not self.kg_available or not farmer_id:
            return {}
        
        try:
            history = self.kg_service.get_farmer_history(farmer_id, limit=5)
            
            # Extract patterns from history
            common_problems = []
            successful_solutions = []
            
            for interaction in history:
                if interaction.get('problem_type'):
                    common_problems.append(interaction['problem_type'])
                if interaction.get('solution_applied'):
                    successful_solutions.append(interaction['solution_applied'])
            
            return {
                'recent_interactions': len(history),
                'common_problems': list(set(common_problems)),
                'successful_solutions': list(set(successful_solutions)),
                'last_interaction_date': history[0].get('date') if history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting farmer context: {e}")
            return {}

    def search_crop_knowledge(self, crop: str, topic: str = None) -> List[Dict]:
        """Search for crop-specific knowledge from the graph"""
        if not self.kg_available:
            return []
        
        try:
            kg_results = self.kg_service.search_crop_specific(crop, topic)
            
            formatted_results = []
            for result in kg_results:
                formatted_results.append({
                    "content": result.content,
                    "source": result.source,
                    "confidence": result.confidence,
                    "type": "crop_knowledge",
                    "crop": crop,
                    "metadata": result.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching crop knowledge: {e}")
            return []


    def _load_crop_model(self, path: Optional[str] = None) -> None:
        """Load the saved crop model (scikit-learn style)."""
        path = path or self.crop_model_path
        if not path or not os.path.exists(path):
            logger.warning(f"Crop model not found at: {path}. Set CROP_MODEL_PATH or pass path explicitly.")
            self.crop_model = None
            return
        try:
            if JOBLIB_AVAILABLE:
                self.crop_model = joblib.load(path)
            else:
                with open(path, "rb") as f:
                    self.crop_model = pickle.load(f)
            
            # Default feature order based on your dataset
            self.crop_model_features = getattr(
                self.crop_model, "feature_names_in_", 
                ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            )
            logger.info(f"✅ Loaded crop model from {path}. Features: {list(self.crop_model_features)}")
        except Exception as e:
            logger.error(f"❌ Failed to load crop model: {e}")
            self.crop_model = None

    def _fetch_weather_raw(self, location: str) -> Dict:
        """Fetch raw wttr.in JSON for numeric features."""
        try:
            resp = requests.get(f"http://wttr.in/{location}", params={"format": "j1"}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Weather API status {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}
        
    def get_weather_features_for_crop_model(self, location: str) -> Dict[str, float]:
        """
        Return numeric weather features suitable for the crop model.
        Returns: temperature [°C], humidity [%], rainfall [mm/day].
        """
        raw = self._fetch_weather_raw(location)
        if "error" in raw:
            logger.warning(f"Weather error for {location}: {raw['error']}")
            return {}

        try:
            current = raw["current_condition"][0]
            today = raw["weather"][0] if raw.get("weather") else None

            temperature = float(current.get("temp_C", 25.0))  # default fallback
            humidity = float(current.get("humidity", 70.0))   # default fallback

            # Calculate daily rainfall from hourly data
            rainfall = 0.0
            if today and "hourly" in today and today["hourly"]:
                for h in today["hourly"]:
                    val = h.get("precipMM")
                    if val is not None:
                        try:
                            rainfall += float(val)
                        except Exception:
                            pass

            return {
                "temperature": temperature, 
                "humidity": humidity, 
                "rainfall": rainfall
            }
        except Exception as e:
            logger.error(f"Failed parsing weather features: {e}")
            return {}
        
    def _assemble_feature_vector(
        self,
        soil: Dict[str, float],
        weather_feats: Dict[str, float]
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """
        Build the feature vector in the correct order expected by the model.
        Expected keys: N, P, K, temperature, humidity, ph, rainfall
        """
        ordered_feats = list(self.crop_model_features or 
                           ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        
        # Merge weather and soil data (soil can override weather keys if any)
        merged = {**weather_feats, **soil}

        # Validate and convert to float
        final_map = {}
        missing = []
        for key in ordered_feats:
            if key in merged and merged[key] is not None:
                try:
                    final_map[key] = float(merged[key])
                except Exception:
                    missing.append(key)
            else:
                missing.append(key)

        if missing:
            raise ValueError(
                f"Missing required features for crop model: {', '.join(missing)}. "
                f"Provided: {sorted(list(merged.keys()))}"
            )

        x = np.array([final_map[k] for k in ordered_feats], dtype=float).reshape(1, -1)
        return x, final_map

    def recommend_crop(
        self,
        soil: Dict[str, float],
        location: str,
        model_path: Optional[str] = None,
        top_k: int = 3
    ) -> Dict:
        """
        Predict best crop using saved model + live weather + soil health card data.
        
        Args:
            soil: Dict with soil parameters {'N': 90, 'P': 42, 'K': 43, 'ph': 6.5}
            location: 'District, State' for weather data
            model_path: Optional path to override default model
            top_k: Number of top crop recommendations to return
            
        Returns:
            Dict with prediction, confidence scores, and feature data used
        """
        if model_path:
            self._load_crop_model(model_path)
        
        if self.crop_model is None:
            return {
                "error": "Crop model not loaded. Check CROP_MODEL_PATH environment variable.",
                "suggestion": "Make sure RF.pkl model file exists and is accessible."
            }

        # Get live weather data
        weather_feats = self.get_weather_features_for_crop_model(location)
        if not weather_feats:
            return {
                "error": "Could not fetch weather data for crop prediction.",
                "location_attempted": location
            }

        try:
            # Combine soil and weather data into feature vector
            x, used_features = self._assemble_feature_vector(soil, weather_feats)
            
            # Make prediction
            prediction = self.crop_model.predict(x)[0]

            # Get confidence scores if available (for RandomForest, etc.)
            top_recommendations = []
            if hasattr(self.crop_model, "predict_proba"):
                proba = self.crop_model.predict_proba(x)[0]
                classes = getattr(self.crop_model, "classes_", None)
                if classes is not None:
                    ranked = sorted(zip(classes, proba), key=lambda z: z[1], reverse=True)
                    for crop_name, confidence in ranked[:top_k]:
                        top_recommendations.append({
                            "crop": str(crop_name),
                            "confidence": round(float(confidence) * 100, 2)  # Convert to percentage
                        })

            result = {
                "success": True,
                "recommended_crop": str(prediction),
                "top_recommendations": top_recommendations,
                "soil_parameters_used": {k: v for k, v in used_features.items() 
                                       if k in ["N", "P", "K", "ph"]},
                "weather_parameters_used": weather_feats,
                "location": location,
                "model_features": list(self.crop_model_features or [])
            }
            
            logger.info(f"✅ Crop recommendation: {prediction} for location {location}")
            return result
            
        except Exception as e:
            logger.error(f"Crop recommendation failed: {e}")
            return {
                "error": f"Crop prediction failed: {str(e)}",
                "soil_provided": soil,
                "weather_location": location
            }

    def parse_soil_health_card(self, soil_data: Dict) -> Dict[str, float]:
        """
        Parse soil health card data into format expected by crop model.
        
        Args:
            soil_data: Raw soil data (could be from JSON, form input, etc.)
            
        Returns:
            Cleaned soil parameters dict
        """
        try:
            # Expected keys for the crop model
            required_keys = ["N", "P", "K", "ph"]
            optional_keys = ["EC", "OC", "Zn", "Fe", "Mn", "Cu", "B", "S"]
            
            cleaned_soil = {}
            
            # Handle different key naming conventions
            key_mappings = {
                "nitrogen": "N", "n": "N",
                "phosphorus": "P", "p": "P", "phosphorous": "P",
                "potassium": "K", "k": "K",
                "pH": "ph", "PH": "ph"
            }
            
            # Apply mappings and convert to float
            for key, value in soil_data.items():
                mapped_key = key_mappings.get(key, key)
                if mapped_key in required_keys or mapped_key in optional_keys:
                    try:
                        cleaned_soil[mapped_key] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert soil parameter {key}={value} to float")
            
            # Validate required parameters
            missing_required = [k for k in required_keys if k not in cleaned_soil]
            if missing_required:
                logger.warning(f"Missing required soil parameters: {missing_required}")
            
            return cleaned_soil
            
        except Exception as e:
            logger.error(f"Error parsing soil health card: {e}")
            return {}

    
    # def analyze_query(self, query: str, profile: Dict = None) -> List[Dict]:
    #     """
    #     Lightweight query analysis using keyword matching instead of heavy LLM.
    #     This prevents loading a second 8B model just for intent classification.
        
    #     Args:
    #         query: The user's question
    #         profile: Optional user profile information

    #     Returns:
    #         List of dicts, each with query analysis for a sub-query
    #     """
    #     logger.info(f"Analyzing query (lightweight): {query}")
        
    #     query_lower = query.lower()
    #     profile = profile or {}
        
    #     # Weather-related keywords
    #     weather_keywords = ['weather', 'temperature', 'rain', 'forecast', 'climate', 'humidity', 'wind', 'sunny', 'cloudy']
    #     if any(keyword in query_lower for keyword in weather_keywords):
    #         return [{
    #             "query_type": "weather_info",
    #             "tools_needed": ["weather_api"],
    #             "search_query": query,
    #             "metadata_filters": {}
    #         }]
        
    #     # Market/price-related keywords  
    #     price_keywords = ['price', 'market', 'cost', 'sell', 'mandi', 'rate', 'buying', 'selling', 'profit', 'money']
    #     if any(keyword in query_lower for keyword in price_keywords):
    #         return [{
    #             "query_type": "market_prices",
    #             "tools_needed": ["price_api"],
    #             "search_query": query,
    #             "metadata_filters": {}
    #         }]
        
    #     # Scheme-related keywords
    #     scheme_keywords = ['scheme', 'subsidy', 'government', 'loan', 'credit', 'support', 'benefit', 'eligibility', 'pm kisan', 'insurance']
    #     if any(keyword in query_lower for keyword in scheme_keywords):
    #         return [{
    #             "query_type": "scheme_search", 
    #             "tools_needed": ["elasticsearch"],
    #             "search_query": query,
    #             "metadata_filters": {"state": profile.get("state")} if profile.get("state") else {}
    #         }]
        
    #     # Default: treat as general farming question (search schemes)
    #     return [{
    #         "query_type": "general_question",
    #         "tools_needed": ["elasticsearch"], 
    #         "search_query": query,
    #         "metadata_filters": {}
    #     }]

    def analyze_query(self, query: str, profile: Dict = None) -> List[Dict]:
        """
        Lightweight query analysis using keyword matching instead of heavy LLM.
        This prevents loading a second 8B model just for intent classification.
        
        Args:
            query: The user's question
            profile: Optional user profile information

        Returns:
            List of dicts, each with query analysis for a sub-query
        """
        logger.info(f"Analyzing query (lightweight): {query}")
        
        query_lower = query.lower()
        profile = profile or {}

        # Knowledge Graph specific keywords
        kg_keywords = ['pest', 'disease', 'fertilizer', 'soil health', 'nutrient', 'deficiency', 
                    'cultivation', 'farming practice', 'crop rotation', 'irrigation', 
                    'harvest', 'planting', 'seeds', 'variety', 'treatment', 'control',
                    'management', 'symptoms', 'organic', 'chemical', 'spray']
        
        # Problem-solution keywords
        problem_keywords = ['problem', 'issue', 'disease', 'pest', 'infestation', 'yellowing', 
                        'wilting', 'spots', 'fungus', 'bacteria', 'virus', 'treatment',
                        'how to treat', 'how to control', 'what to do']
        
        if any(keyword in query_lower for keyword in problem_keywords):
            tools = ["knowledge_graph", "FARMING_KB"]
            if self.kg_available:
                return [{
                    "query_type": "problem_solution",
                    "tools_needed": tools,
                    "search_query": query_lower,
                    "metadata_filters": {}
                }]
        
        if any(keyword in query_lower for keyword in kg_keywords):
            tools = ["knowledge_graph", "FARMING_KB"] if self.kg_available else ["FARMING_KB"]
            return [{
                "query_type": "knowledge_graph_search",
                "tools_needed": tools,
                "search_query": query_lower,
                "metadata_filters": {}
            }]
        
        # Crop-specific queries
        crop_names = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'potato', 'tomato', 'onion']
        if any(crop in query_lower for crop in crop_names):
            tools = ["knowledge_graph", "FARMING_KB"] if self.kg_available else ["FARMING_KB"]
            return [{
                "query_type": "crop_specific",
                "tools_needed": tools,
                "search_query": query_lower,
                "metadata_filters": {}
            }]
            # Weather-related keywords
        weather_keywords = ['weather', 'temperature', 'rain', 'forecast', 'climate', 'humidity', 'wind', 'sunny', 'cloudy']
        if any(keyword in query_lower for keyword in weather_keywords):
            return [{
                "query_type": "weather_info",
                "tools_needed": ["weather_api"],
                "search_query": query_lower,
                "metadata_filters": {}
            }]
        
        # Market/price-related keywords  
        price_keywords = ['price', 'market', 'cost', 'sell', 'mandi', 'rate', 'buying', 'selling', 'profit', 'money']
        if any(keyword in query_lower for keyword in price_keywords):
            return [{
                "query_type": "market_prices",
                "tools_needed": ["price_api"],
                "search_query": query_lower,
                "metadata_filters": {}
            }]
        
        # Scheme-related keywords
        scheme_keywords = ['scheme', 'subsidy', 'government', 'loan', 'credit', 'support', 'benefit', 'eligibility', 'pm kisan', 'insurance']
        if any(keyword in query_lower for keyword in scheme_keywords):
            return [{
                "query_type": "scheme_search", 
                "tools_needed": ["elasticsearch"],
                "search_query": query,
                "metadata_filters": {"state": profile.get("state")} if profile.get("state") else {}
            }]
        
    # Default: include KG if available
        if self.kg_available:
            return [{
                "query_type": "hybrid_search",
                "tools_needed": ["knowledge_graph", "SCHEMES_KB"], 
                "search_query": query,
                "metadata_filters": {}
            }]
        else:
            return [{
                "query_type": "general_question",
                "tools_needed": ["SCHEMES_KB"], 
                "search_query": query,
                "metadata_filters": {}
            }]


    def get_weather_data(self, location: str) -> Dict:
        """Get weather data for a specific location using wttr.in API"""
        try:
            logger.info(f"Getting weather data for: {location}")
            
            # Using wttr.in - completely free, no API key needed
            base_url = f'http://wttr.in/{location}'
            params = {
                'format': 'j1'  # JSON format
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                weather_data = response.json()
                
                # Extract relevant information
                current = weather_data['current_condition'][0]
                forecast_days = weather_data.get('weather', [])[:3]
                
                result = {
                    "location": location,
                    "temperature": f"{current['temp_C']}°C",
                    "feels_like": f"{current['FeelsLikeC']}°C",
                    "humidity": f"{current['humidity']}%",
                    "weather_description": current['weatherDesc'][0]['value'],
                    "wind_speed": f"{current['windspeedKmph']} km/h",
                    "forecast": [
                        {
                            "date": day['date'],
                            "max_temp": f"{day['maxtempC']}°C",
                            "min_temp": f"{day['mintempC']}°C",
                            "description": day['hourly'][0]['weatherDesc'][0]['value'] if day['hourly'] else "N/A"
                        } for day in forecast_days
                    ]
                }
                
                logger.info(f"✅ Weather data retrieved for {location}")
                return result
                
            else:
                logger.error(f"Weather API error: Status {response.status_code}")
                return {
                    "location": location,
                    "error": f"Weather API returned status code: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"Weather API timeout for {location}")
            return {
                "location": location,
                "error": "Weather API request timed out"
            }
        except Exception as e:
            logger.error(f"Weather API exception: {e}")
            return {
                "location": location,
                "error": f"Error fetching weather data: {str(e)}"
            }
    
    # def search_metadata(self, query: str, index_to_search: str = "scheme_metadata_filter") -> List[Dict]:
    #     """
    #     Search for metadata_filters based on query from metadata index in ES

    #     Args:
    #         query: The search query
    #         index_to_search: Index to search for metadata from
        
    #     Returns:
    #         List of relevant Metadata filters
    #     """
    #     logger.info(f"Searching metadata for query: '{query}' in index: '{index_to_search}'")
        
    #     if not self.es_available:
    #         logger.warning("Elasticsearch is not available.")
    #         return []
            
    #     try:
    #         results = self.es_rag.vector_search(
    #             query=query,
    #             top_k=10,
    #             search_type="metadata",
    #             index_name=index_to_search
    #         )
    #     except Exception as e:
    #         logger.error(f"No metadata found 😔 Exception: {e}")
    #         results = []
            
    #     if not results:
    #         logger.warning("No metadata results found or returned from Elasticsearch.")

    #     filtered = [
    #         {
    #             "scheme_id": item.get("scheme_id"),
    #         }
    #         for item in results
    #     ]
    #     return filtered
        
    # def search_schemes(self, query: str, model='qwen', metadata_list: Optional[list] = None) -> List[Dict]:
    #     """
    #     Search for agricultural schemes in Elasticsearch
        
    #     Args:
    #         query: The search query
    #         model: Model to use for search
    #         metadata_list: Optional metadata filters
            
    #     Returns:
    #         List of relevant schemes
    #     """
    #     logger.info(f"Searching schemes for: '{query}'")
    #     metadata_list = self.search_metadata(query)
    #     if not self.es_available:
    #         logger.warning("Elasticsearch not available, returning mock results")
    #         return [
    #             {
    #                 "name": "PM Kisan Samman Nidhi",
    #                 "objective": "Direct income support to farmers",
    #                 "benefits": "₹6000 per year in three installments",
    #                 "eligibility": "Small and marginal farmers with up to 2 hectares"
    #             },
    #             {
    #                 "name": "Pradhan Mantri Fasal Bima Yojana",
    #                 "objective": "Crop insurance scheme",
    #                 "benefits": "Protection against crop loss due to natural disasters",
    #                 "eligibility": "All farmers growing notified crops"
    #             }
    #         ]

    #     try:
    #         results = self.es_rag.hybrid_search_with_metadata(
    #             query_text=query,
    #             top_k=3,
    #             model=model,
    #             metadata_list=metadata_list
    #         )
            
    #         logger.info(f"✅ Found {len(results)} scheme results")
    #         return results
            
    #     except Exception as e:
    #         logger.error(f"Error searching schemes: {e}")
    #         return []


    def search_metadata(self, query: str, index_to_search: str = "scheme_metadata_filter", query_embedding=None) -> List[Dict]:
        """
        Search for metadata_filters based on query from metadata index in ES

        Args:
            query: The search query
            index_to_search: Index to search for metadata from
        
        Returns:
            List of relevant Metadata filters
        """
        logger.info(f"Searching metadata for query: '{query}' in index: '{index_to_search}'")
        if not self.es_available:
            logger.warning("Elasticsearch is not available.")
            return []
        try:
            # Use cached embedding if provided
            if query_embedding is not None:
                results = self.es_rag.vector_search(
                    query=query,
                    top_k=3,
                    search_type="metadata",
                    index_name=index_to_search,
                    query_embedding=query_embedding
                )
            else:
                results = self.es_rag.vector_search(
                    query=query,
                    top_k=3,
                    search_type="metadata",
                    index_name=index_to_search
                )
        except Exception as e:
            logger.error(f"No metadata found 😔 Exception: {e}")
            results = []
        if not results:
            logger.warning("No metadata results found or returned from Elasticsearch.")

        filtered = [
            {
                "scheme_id": item.get("scheme_id"),
            }
            for item in results
        ]
        logger.info(f"🥰🥰🥰❤️❤️❤️🥰🥰🥰❤️❤️❤️🥰🥰🥰❤️❤️❤️🥰🥰🥰❤️❤️❤️ metadata retreived {filtered}\n\n\n")
        return filtered
        
    def search_keys(self, query: str, model='minilm', index="schemes", query_embedding=None):
        if index == "schemes":
            keys = self.keys_schemes
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            if query_embedding is None:
                query_embedding = st_model.encode(query)
            text_embeddings = st_model.encode(keys)

            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            scores = []
            for i, text_embedding in enumerate(text_embeddings):
                score = cosine_similarity(query_embedding, text_embedding)
                scores.append((keys[i], score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_keys = [k for k, _ in scores[:5]]

            
            if "embedding_text" not in top_keys:
                top_keys.append("embedding_text")
            if "references or sources" not in top_keys:
                top_keys.append("references or sources")
            if "eligibility" in top_keys and "excluded or ineligibility" not in top_keys:
                top_keys.append("excluded or ineligibility")
            if "excluded or ineligibility" in top_keys and "eligibility" not in top_keys:
                top_keys.append("eligibility")
            if "name" not in top_keys:
                top_keys.append("name")

            return top_keys

    def search_schemes(self, query: str, model='qwen', metadata_list: Optional[list] = None) -> List[Dict]:
        """
        Search for agricultural schemes in Elasticsearch

        Args:
            query: The search query
            metadata_filters: Optional filters for the search

        Returns:
            List of relevant schemes (filtered to only relevant keys)
        """
        if not self.es_available:
            return []

        # Compute and cache MiniLM embedding ONCE
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.query_embedding_mini_lm = st_model.encode(query)

        metadata_filters = self.search_metadata(query, query_embedding=self.query_embedding_mini_lm)
        try:
            # Use the ElasticsearchRAG utility for hybrid search
            results = self.es_rag.hybrid_search_with_metadata(
                query_text=query,
                top_k=5,
                model=model,
                metadata_list=metadata_filters
            )

            # Get relevant keys for this query
            relevant_keys = self.search_keys(query, model="minilm", index="schemes", query_embedding=self.query_embedding_mini_lm)
            logger.info(f"\n\n🥰🥰🥰😘😘🚑❤️❤️❤️🥰🥰🥰❤️❤️❤️ relevant keys {relevant_keys}\n\n\n")
            def filter_scheme_keys(scheme, allowed_keys):
                # If ES returns _source, use that, else use scheme directly
                src = scheme.get("_source", scheme)
                return {k: v for k, v in src.items() if k in allowed_keys}

            # Filter each scheme to only include relevant keys
            filtered_results = [filter_scheme_keys(s, relevant_keys) for s in results]
            # Print and save as rag_context.json
            # rag_context = {"rag_context": filtered_results}
            # print(json.dumps(rag_context, indent=2, ensure_ascii=False))
            # with open("rag_context.json", "w", encoding="utf-8") as f:
            #     json.dump(rag_context, f, indent=2, ensure_ascii=False)

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching schemes: {e}")
            return []
    
    
    def search_farming_info(self, query: str, model='qwen', top_k: int = 5) -> list:
        """
        Search for general farming advice or information in Elasticsearch.

        Args:
            query: The search query
            model: Embedding/model to use for search
            top_k: Number of results to return

        Returns:
            List of relevant farming info documents
        """
        if not self.es_available:
            logger.warning("Elasticsearch not available, returning empty results.")
            return []

        try:
            index_name = "paddy"  # Change to your actual index name if different

            # Use cached embedding model
            st_model = self.get_embedding_model()
            query_embedding = st_model.encode(query)

            results = self.es_rag.farming_search(
                query_text=query,
                top_k=top_k,
                model=model,
                index_name=index_name,
                query_embedding=query_embedding
            )

            filtered_results = []
            for doc in results:
                src = doc.get("_source", doc)
                filtered = {
                    "section": src.get("section", ""),
                    "subsection": src.get("subsection", ""),
                    "text": src.get("text", "")
                }
                filtered_results.append(filtered)


            logger.info(f"✅ Found 🥰🥰🥰😘  {len(filtered_results)} farming info results")
            # logger.info(f"🥰🥰🥰😘 farming info retreived {filtered_results}\n\n\n")
            return filtered_results

        except Exception as e:
            logger.error(f"Error searching farming info: {e}")
            return []


    # 
    def get_market_prices(self, crop: str = None, location: str = None) -> Dict:
        """
        Return the entire prices.csv file as a list of dicts.
        Args:
            crop: Ignored, kept for compatibility.
            location: Ignored, kept for compatibility.
        Returns:
            Dict with all CSV data.
        """
        logger.info("Returning entire prices.csv file")
        csv_path = "./prices.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Could not load prices.csv: {e}")
            return {
                "error": "Could not load price data"
            }

        all_data = df.to_dict(orient="records")
        return {
            "data": all_data,
            "count": len(all_data),
            "last_updated": "2025-08-15"
        }

    def generate_rag_prompt(self, query: str, search_results: List[Dict]) -> str:
        """
        Generate a RAG prompt with retrieved context
        
        Args:
            query: User query
            search_results: Elasticsearch search results
            
        Returns:
            Formatted prompt with context
        """
        if not search_results:
            return query
            
        # Format search results as context
        context = "I found the following agricultural schemes that may be relevant:\n\n"
        
        for i, scheme in enumerate(search_results, 1):
            context += f"{i}. {scheme.get('name', 'Unnamed Scheme')}\n"
            
            if 'objective' in scheme and scheme['objective']:
                context += f"   Objective: {scheme['objective']}\n"
                
            if 'benefits' in scheme and scheme['benefits']:
                context += f"   Benefits: {scheme['benefits']}\n"
                
            if 'eligibility' in scheme and scheme['eligibility']:
                context += f"   Eligibility: {scheme['eligibility']}\n"
            
            context += "\n"
            
        # Create RAG prompt
        rag_prompt = f"""Context information:
{context}

Using the context information provided above, please answer the following question:
{query}

If the context doesn't contain the information needed to answer the question, say so clearly
and do not generate any information from your memory.
"""
        return rag_prompt

if __name__ == "__main__":
    # Test the tools
    logger.info("Testing FarmerTools...")
    tools = FarmerTools()
    
    # Test weather API
    weather = tools.get_weather_data("Karnataka, India")
    print("Weather test:", weather.get('location'), weather.get('temperature'))
    
    # Test query analysis
    analysis = tools.analyze_query("What's the weather today?", {"state": "Karnataka"})
    print("Analysis test:", analysis)