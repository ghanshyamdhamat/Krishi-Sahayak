"""
Farmer Agent - Clean LangGraph-based agent for helping farmers
This agent integrates Elasticsearch RAG, Qwen3-8B for reasoning, and various APIs
Features chain-of-thought reasoning to decide between API calls and knowledge base
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Optional, Literal
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j_files.store_farmer_profile import StoreFarmerProfile

# Try to import transformers with error handling
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Import custom tools
from agent_tools_clean import FarmerTools

# --- SETUP ---

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangGraph-compatible Qwen3-8B LLM with chain-of-thought reasoning
class Qwen3LLM(Runnable):
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        """Initialize Qwen3-8B model using transformers with quantization"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen3-8B model with quantization"""
        try:
            logger.info(f"Loading Qwen3-8B model with quantization: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            if torch.cuda.is_available():
                try:
                    # Try 4-bit quantization for memory efficiency
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16
                    )
                    
                    logger.info("✅ Model loaded with 4-bit quantization on GPU")
                    
                except ImportError:
                    logger.warning("BitsAndBytesConfig not available, using float16")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={0: "10GB"}
                    )
                    logger.info("✅ Model loaded with float16 on GPU")
            else:
                # CPU loading
                dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info(f"✅ Model loaded with {dtype} on CPU")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def invoke(self, input_data, config=None, **kwargs):
        """Generate response using the Qwen3-8B model - LangGraph compatible"""
        try:
            # Handle different input formats from LangGraph
            if isinstance(input_data, dict):
                if 'messages' in input_data:
                    messages = input_data['messages']
                    if messages and hasattr(messages[-1], 'content'):
                        prompt = messages[-1].content
                    else:
                        prompt = str(messages)
                else:
                    prompt = str(input_data)
            else:
                prompt = str(input_data)
            
            # Prepare the prompt for chat format
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant for farmers in India. Provide clear, concise, and practical advice."},
                    {"role": "user", "content": str(prompt)}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = f"<|im_start|>system\nYou are a helpful AI assistant for farmers in India. Provide clear, concise, and practical advice.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize input
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=100000)
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
            else:
                # Remove the original prompt
                response = response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            return response.strip() if response.strip() else "I'm here to help with your farming questions."
            
        except Exception as e:
            logger.error(f"Error generating response with Qwen3-8B: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

    @property
    def InputType(self):
        return dict

    @property  
    def OutputType(self):
        return str

# Fallback LLM for when transformers is not available
class MockLLM(Runnable):
    """Mock LLM that provides farming responses when Qwen3 is not available"""
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def invoke(self, input_data, config=None, **kwargs):
        """Generate contextual response based on input"""
        if isinstance(input_data, dict):
            if 'messages' in input_data:
                messages = input_data['messages']
                if messages and hasattr(messages[-1], 'content'):
                    query = messages[-1].content.lower()
                else:
                    query = str(messages).lower()
            else:
                query = str(input_data).lower()
        else:
            query = str(input_data).lower()
        
        # Generate contextual responses based on query content
        if any(word in query for word in ['weather', 'temperature', 'forecast', 'rain', 'climate']):
            return """🌤️ **Weather Information:**

Based on the available weather data for your location, here are the current conditions and farming recommendations:

**Current Weather:**
- Temperature: As per the weather data retrieved
- Humidity: Current humidity levels
- Conditions: Current weather conditions
- Wind Speed: Current wind information

**Farming Advice:**
- Plan your farming activities based on current weather conditions
- Consider irrigation needs based on humidity and rainfall
- Monitor weather forecasts for planning field operations
- Take protective measures during adverse weather conditions"""

        elif any(word in query for word in ['price', 'market', 'cost', 'sell', 'mandi']):
            return """💰 **Market Price Information:**

**Current Market Trends:**
- Prices are showing stability in most regions
- Consider local demand and transportation costs
- Monitor market trends for optimal selling time

**Recommendations:**
- Check local mandi prices regularly
- Consider storage options vs immediate sale
- Factor in quality premiums for better crops
- Plan harvest timing based on market demand"""

        elif any(word in query for word in ['scheme', 'subsidy', 'government', 'loan', 'support']):
            return """🏛️ **Government Schemes Information:**

Based on the scheme information found, here are the key details:

**Available Schemes:**
- Various government schemes are available for farmers
- Income support and crop insurance options
- Subsidies for agricultural inputs and equipment

**Next Steps:**
- Visit your nearest Krishi Vigyan Kendra
- Contact local agriculture extension officer
- Check eligibility criteria for specific schemes
- Apply through proper channels with required documents"""

        else:
            return """🧑‍🌾 **Farming Assistant:**

I'm here to help you with comprehensive farming support including:

🌾 **Weather Services** - Current conditions and forecasts
💰 **Market Information** - Crop prices and trends  
🏛️ **Government Schemes** - Subsidies and support programs
🌱 **Farming Guidance** - Best practices and recommendations

Please ask specific questions about weather, prices, schemes, or farming practices for detailed assistance."""

    @property
    def InputType(self):
        return dict

    @property  
    def OutputType(self):
        return str

# Initialize LLM
if TRANSFORMERS_AVAILABLE:
    try:
        llm = Qwen3LLM()
        logger.info("✅ Qwen3-8B model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qwen3-8B: {e}")
        llm = MockLLM()
        logger.info("✅ Using MockLLM as fallback")
else:
    llm = MockLLM()
    logger.info("✅ Using MockLLM (transformers not available)")

# Initialize tools
farmer_tools = FarmerTools()
logger.info("✅ Farmer tools initialized")

# --- MODELS ---

class QueryType(str, Enum):
    SCHEME_SEARCH = "scheme_search"
    CROP_ADVICE = "crop_advice"
    WEATHER_INFO = "weather_info"
    MARKET_PRICES = "market_prices"
    GENERAL_QUESTION = "general_question"
    HYBRID = "hybrid"

class FarmerProfile(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    contact: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    country: Optional[str] = "India"
    # taluka: Optional[str] = None
    # village: Optional[str] = None
    crops: List[str] = Field(default_factory=list)
    land_size_acres: Optional[float] = None
    preferred_language: Optional[str] = "English"

# Add knowledge_graph_results to your AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[Any], "Conversation messages"]
    profile: Annotated[Optional[FarmerProfile], "Farmer's profile"]
    query_analysis: Annotated[Dict, "Analysis of the query"]
    scheme_results: Annotated[List[Dict], "Results from search tools"]
    farming_info_results: Annotated[List[Dict], "Farming information from Elasticsearch"]
    knowledge_graph_results: Annotated[List[Dict], "Results from Neo4j knowledge graph"]  # Add this line
    crop_recommendation: Annotated[Optional[Dict], "Crop recommendation if available"]
    weather_data: Annotated[Optional[Dict], "Weather data if requested"]
    market_data: Annotated[Optional[Dict], "Market price data if requested"]
    reasoning_chain: Annotated[str, "Chain of thought reasoning"]
    final_response: Annotated[str, "Final response to user"]
    next_step: Annotated[str, "Next step in the workflow"]


def log_tool_usage_summary(state: AgentState):
    """Log a summary of which tools were used for debugging"""
    query_analysis = state.get("query_analysis", {})
    tools_needed = query_analysis.get("tools_needed", [])
    
    logger.info("=== TOOL USAGE SUMMARY ===")
    logger.info(f"Query: {state.get('messages', [])[-1].content if state.get('messages') else 'Unknown'}")
    logger.info(f"Tools Selected: {tools_needed}")
    
    if "KNOWLEDGE_GRAPH" in tools_needed:
        kg_results = state.get("knowledge_graph_results", [])
        logger.info(f"🧠 KNOWLEDGE GRAPH: {'✅ USED' if kg_results else '❌ NO RESULTS'}")
        if kg_results:
            logger.info(f"   - Found {len(kg_results)} results")
            for i, result in enumerate(kg_results, 1):
                logger.info(f"   - Result {i}: {result.get('type', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
    else:
        logger.info("🧠 KNOWLEDGE GRAPH: ❌ NOT SELECTED")
    
    logger.info("========================")

def analyze_query_with_reasoning(state: AgentState) -> AgentState:
    """Analyze query using chain-of-thought reasoning to decide on tools"""
    logger.info("Node: analyze_query_with_reasoning")
    
    # Extract the latest user message
    latest_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            latest_message = message.content
            break
    
    if not latest_message:
        return {
            **state,
            "query_analysis": {},
            "reasoning_chain": "",
            "next_step": "generate_response"
        }
    
    query_lower = latest_message.lower()

    # Add crop recommendation detection
    crop_keywords = ['crop recommendation', 'what crop', 'which crop', 'best crop', 'suitable crop', 
                    'crop suggestion', 'recommend crop', 'soil test', 'soil health card', 
                    'what to grow', 'what should i plant', 'crop selection']
    if any(keyword in query_lower for keyword in crop_keywords):
        analysis = {
            "classification": "CROP_RECOMMENDATION",
            "query_type": "crop_recommendation",
            "tools_needed": ["CROP_RECOMMENDATION"],
            "priority_tool": "crop_model",
            "search_query": query_lower,
            "metadata_filters": {},
            "reasoning_chain": "Detected crop recommendation keywords. Prioritizing crop model."
        }
        logger.info(f"Query analysis result: {analysis}")
        return {
            **state,
            "query_analysis": analysis,
            "reasoning_chain": analysis["reasoning_chain"],
            "next_step": "execute_tools_intelligently"
        }
    
    # Get farmer profile context
    profile_dict = state["profile"].model_dump() if state.get("profile") else {}
    profile_context = f"""
    Farmer Profile:
    - Name: {profile_dict.get('name', 'Unknown')}
    - State: {profile_dict.get('state', 'Unknown')}
    - District: {profile_dict.get('district', 'Unknown')}
    - Country: {profile_dict.get('country', 'India')}
    - Preferred Language: {profile_dict.get('preferred_language', 'English')}
    - Crops: {', '.join(profile_dict.get('crops', []))}
    - Land Size: {profile_dict.get('land_size_acres', 'Unknown')} acres
    """
    
    # Enhanced Chain-of-thought reasoning prompt with Knowledge Graph
    reasoning_prompt = f"""You are an intelligent farming assistant. Analyze this query using step-by-step reasoning:
This is the farmers profile context to help you:
{profile_context}

USER QUERY: "{latest_message}"

REASONING PROCESS:
Step 1: What is the farmer asking about?
Step 2: What type of information do they need?
Step 3: What data sources would be most helpful?
Step 4: Should I use live APIs, knowledge base, or knowledge graph?

DECISION CRITERIA:
- Weather queries → Use Weather API (live data needed)
- Market prices → Use Price API (live data needed) 
- Government schemes → Use Knowledge Base (comprehensive info needed)
- Agricultural problems/diseases/pests → Use KNOWLEDGE_GRAPH (expert agricultural knowledge)
- Farming practices/cultivation methods → Use KNOWLEDGE_GRAPH (technical knowledge)
- Fertilizer/treatment recommendations → Use KNOWLEDGE_GRAPH (specific guidance)
- General farming advice → Combine KNOWLEDGE_GRAPH + Knowledge Base
- Location-specific info → Combine APIs + Knowledge Base + KNOWLEDGE_GRAPH

Based on your analysis, determine the tools needed to answer the query with utmost accuracy:

Available tools:
1. WEATHER_LIVE - Needs current weather data from API
2. PRICES_LIVE - Needs current market prices from API  
3. SCHEMES_KB - Needs government scheme info from knowledge base
4. FARMING_KB - Needs general farming advice from knowledge base
5. KNOWLEDGE_GRAPH - Needs expert agricultural knowledge from Neo4j graph (USE THIS for agricultural problems, diseases, pests, treatments, cultivation practices, fertilizers, crop management)
6. CROP_RECOMMENDATION - Needs crop recommendation model and weather data

# Also classify the query type into any two of the following categories to best describe the query:
"FarmingActivity": "Procedural knowledge, best practices, or management strategies for a specific farming task (e.g., sowing, land preparation, fertilizer application timing, water management).",
"AgriculturalProblem": "Information describing a problem affecting a crop, including its symptoms and causes. This includes pests, diseases, weeds, nutrient deficiencies, or physiological disorders like 'straighthead'.",
"SolutionOrTreatment": "Actionable advice, products, or methods for solving an agricultural problem. This includes specific fertilizers (urea, ammonium sulfate), pesticides, organic controls, or corrective actions like draining a field.",
"MarketAndFinance": "Data related to market prices, government schemes, crop insurance, subsidies, loans, or the economics of farming.",
"AgronomicFact": "Fundamental scientific information about crop botany, soil science, plant nutrition, soil chemistry (e.g., nitrification, pH), or weather patterns. This describes the 'why' behind farming practices.",
"Other": "Relevant agricultural information that does not fit into the other categories."

IMPORTANT: If the query is about agricultural problems, diseases, pests, treatments, fertilizers, or farming practices, ALWAYS include KNOWLEDGE_GRAPH in your tools list with some other tool as well as you can't just rely on knowledge graph

Respond with your reasoning and then end with:
TOOLS_NEEDED: [list of tools]
CATEGORY: [FarmingActivity, AgriculturalProblem, SolutionOrTreatment, MarketAndFinance, AgronomicFact, Other]

Example response:
TOOLS_NEEDED: [KNOWLEDGE_GRAPH, FARMING_KB]
CATEGORY: [AgriculturalProblem, SolutionOrTreatment]

"""
    
    # Get reasoning from LLM
    reasoning_response = llm.invoke({"messages": [HumanMessage(content=reasoning_prompt)]})
    logger.info(f"Chain-of-thought reasoning: {reasoning_response}")
    
    # Parse the reasoning response
    analysis = parse_reasoning_response(reasoning_response, latest_message, profile_dict)
    
    logger.info(f"🔍 Query analysis result: {analysis}")
    
    return {
        **state,
        "query_analysis": analysis,
        "reasoning_chain": reasoning_response,
        "next_step": "execute_tools_intelligently"
    }

def parse_reasoning_response(reasoning_text: str, original_query: str, profile: Dict) -> Dict:
    """Parse the chain-of-thought reasoning response for tools and categories."""
    # Extract TOOLS_NEEDED
    tools_match = re.search(r"TOOLS_NEEDED:\s*\[([^\]]*)\]", reasoning_text, re.IGNORECASE)
    categories_match = re.search(r"CATEGORY:\s*\[([^\]]*)\]", reasoning_text, re.IGNORECASE)

    tools_needed = []
    if tools_match:
        tools_needed = [t.strip() for t in tools_match.group(1).split(",") if t.strip()]
    # Fallback to FARMING_KB if nothing found
    if not tools_needed:
        tools_needed = ["FARMING_KB"]

    categories = []
    if categories_match:
        categories = [c.strip() for c in categories_match.group(1).split(",") if c.strip()]

    # Use the first tool as the main classification for backward compatibility
    classification = tools_needed[0] if tools_needed else "FARMING_KB"

    # Map classification to query_type and priority_tool
    classification_map = {
        "WEATHER_LIVE": {
            "query_type": "weather_info",
            "priority_tool": "weather_api"
        },
        "PRICES_LIVE": {
            "query_type": "market_prices",
            "priority_tool": "price_api"
        },
        "SCHEMES_KB": {
            "query_type": "scheme_search",
            "priority_tool": "elasticsearch"
        },
        "FARMING_KB": {
            "query_type": "general_question",
            "priority_tool": "elasticsearch"
        },
        "CROP_RECOMMENDATION": {
            "query_type": "crop_recommendation",
            "priority_tool": "crop_model"
        },
        "HYBRID": {
            "query_type": "hybrid_search",
            "priority_tool": "weather_api"
        }
    }
    config = classification_map.get(classification, classification_map["FARMING_KB"])

    return {
        "classification": classification,
        "query_type": config["query_type"],
        "tools_needed": tools_needed,
        "priority_tool": config["priority_tool"],
        "search_query": original_query,
        "metadata_filters": {"state": profile.get("state")} if profile.get("state") else {},
        "categories": categories,
        "reasoning_chain": reasoning_text
    }

def extract_soil_data_from_query(query: str) -> Dict[str, float]:
    """
    Extract soil parameters from user query using regex patterns.
    Supports formats like: N=90, P=42, K=43, ph=6.5
    """
    import re
    
    soil_data = {}
    
    # Patterns for different parameter formats
    patterns = [
        r'N\s*[=:]\s*(\d+\.?\d*)',
        r'P\s*[=:]\s*(\d+\.?\d*)', 
        r'K\s*[=:]\s*(\d+\.?\d*)',
        r'ph\s*[=:]\s*(\d+\.?\d*)',
        r'pH\s*[=:]\s*(\d+\.?\d*)',
        r'nitrogen\s*[=:]\s*(\d+\.?\d*)',
        r'phosphorus\s*[=:]\s*(\d+\.?\d*)',
        r'potassium\s*[=:]\s*(\d+\.?\d*)'
    ]
    
    param_map = {
        'N': 'N', 'nitrogen': 'N',
        'P': 'P', 'phosphorus': 'P',
        'K': 'K', 'potassium': 'K',
        'ph': 'ph', 'pH': 'ph'
    }
    
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            param_name = pattern.split('\\')[0].lower()
            if param_name in param_map:
                try:
                    soil_data[param_map[param_name]] = float(matches[0])
                except ValueError:
                    continue
    
    return soil_data

def execute_tools_intelligently(state: AgentState) -> AgentState:
    """Execute tools based on chain-of-thought analysis - now with KG support"""
    logger.info("Node: execute_tools_intelligently")
    
    query_analysis = state.get("query_analysis", {})
    tools_needed = query_analysis.get("tools_needed", [])
    search_query = query_analysis.get("search_query", "")
    metadata_filters = query_analysis.get("metadata_filters", {})
    profile_dict = state["profile"].model_dump() if state.get("profile") else {}

    updated_state = {
        **state,
        "scheme_results": [],
        "crop_recommendation": None,
        "farming_info_results": [],
        "knowledge_graph_results": [],  # Add KG results
        "weather_data": None,
        "market_data": None,
        "farmer_context": {} 
    }

    try:
        for tool in tools_needed:
            logger.info(f"🔧 Executing tool: {tool}")
            
            if tool == "KNOWLEDGE_GRAPH":
                logger.info("🧠 Using Knowledge Graph for agricultural expertise")
                if search_query:
                    kg_results = farmer_tools.search_knowledge_graph_intelligent(search_query, profile_dict, top_k=3)
                    updated_state["knowledge_graph_results"] = kg_results
                    logger.info(f"✅ Found {len(kg_results)} results from Knowledge Graph")
                    
                    # Log what was found for debugging
                    for i, result in enumerate(kg_results[:2], 1):
                        logger.info(f"   KG Result {i}: {result.get('type', 'unknown')} - {result.get('content', '')[:100]}...")

            elif tool == "WEATHER_LIVE":
                logger.info("🌤️ Using Weather API for live data")
                location = get_farmer_location(profile_dict)
                weather_data = farmer_tools.get_weather_data(location)
                updated_state["weather_data"] = weather_data
                logger.info(f"✅ Retrieved live weather data for {location}")

            elif tool == "PRICES_LIVE":
                logger.info("💰 Using Price API for live market data")
                location = get_farmer_location(profile_dict)
                crops = profile_dict.get("crops", [])
                if crops:
                    market_data = farmer_tools.get_market_prices(crops[0], location)
                    updated_state["market_data"] = market_data
                    logger.info(f"✅ Retrieved live market data for {crops[0]}")

            elif tool == "CROP_RECOMMENDATION":
                logger.info("🌱 Using Crop Recommendation Model")
                soil_data = extract_soil_data_from_query(search_query)
                location = get_farmer_location(profile_dict)
                if soil_data:
                    crop_result = farmer_tools.recommend_crop(soil_data, location)
                    updated_state["crop_recommendation"] = crop_result
                    logger.info(f"✅ Crop recommendation completed for {location}")
                else:
                    updated_state["crop_recommendation"] = {
                        "error": "Soil parameters needed for crop recommendation",
                        "required_parameters": ["N", "P", "K", "ph"],
                        "example": "Please provide: N=90, P=42, K=43, ph=6.5"
                    }

            elif tool == "SCHEMES_KB":
                logger.info("📚 Using Schemes Knowledge Base")
                if search_query:
                    results = farmer_tools.search_schemes(search_query, model='qwen')
                    updated_state.setdefault("scheme_results", []).extend(results)
                    logger.info(f"✅ Found {len(results)} schemes from ES")

            elif tool == "FARMING_KB":
                logger.info("🌾 Using Farming Knowledge Base")
                if search_query:
                    results = farmer_tools.search_farming_info(search_query, model='qwen', top_k=10)
                    updated_state.setdefault("farming_info_results", []).extend(results)
                    logger.info(f"✅ Found {len(results)} farming info results from ES")

    except Exception as e:
        logger.error(f"Error executing tools: {e}")

    updated_state["next_step"] = "generate_intelligent_response"
    log_tool_usage_summary(updated_state)
    return updated_state

def get_farmer_location(profile_dict: Dict) -> str:
    """Get the best location string from farmer profile"""
    if profile_dict.get("district") and profile_dict.get("state"):
        return f"{profile_dict['district']}, {profile_dict['state']}"
    elif profile_dict.get("state"):
        return profile_dict["state"]
    else:
        return "India"


def extract_final_answer(response_text: str) -> str:
    """Extract only the final answer from LLM response, removing thinking parts"""
    if not response_text:
        return response_text
    
    # Remove explicit thinking markers
    thinking_markers = [
        "<think>", "</think>", "Let me think", "I need to think", 
        "My thinking:", "Analysis:", "Step by step:", "First,", "Next,",
        "Looking at this", "Considering", "Based on my analysis"
    ]
    
    cleaned_response = response_text
    for marker in thinking_markers:
        if marker in cleaned_response:
            # Remove everything before the marker if it appears at the start
            parts = cleaned_response.split(marker)
            if len(parts) > 1:
                # Take the last part after the thinking marker
                cleaned_response = parts[-1].strip()
    
    # Split by common answer markers
    answer_markers = [
        "**Answer:**", "**Response:**", "**Recommendation:**", 
        "**Advice:**", "**Final Answer:**", "Here's my advice:",
        "My recommendation:", "Based on this information:",
        "🌤️", "💰", "📚", "🧑‍🌾"  # Start with emoji markers
    ]
    
    for marker in answer_markers:
        if marker in cleaned_response:
            parts = cleaned_response.split(marker, 1)
            if len(parts) > 1:
                return marker + parts[1].strip()
    
    # Remove thinking patterns with regex
    import re
    thinking_patterns = [
        r"<think>.*?</think>",
        r"Let me.*?[\.\n]",
        r"I need to.*?[\.\n]", 
        r"First.*?then.*?[\.\n]",
        r"Step \d+.*?[\.\n]",
        r"My thinking.*?[\.\n]",
        r"Analysis.*?[\.\n]",
        r"Looking at.*?I.*?[\.\n]",
        r"Considering.*?[\.\n]",
        r"Based on my analysis.*?[\.\n]"
    ]
    
    for pattern in thinking_patterns:
        cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE | re.DOTALL)
    
    # If no explicit markers, look for paragraph breaks and take substantial content
    paragraphs = [p.strip() for p in cleaned_response.split('\n\n') if p.strip()]
    
    if paragraphs:
        # Return paragraphs that don't start with analysis words
        analysis_starters = ['based on', 'looking at', 'analyzing', 'considering', 'given that', 'let me', 'i need']
        final_paragraphs = []
        
        for para in paragraphs:
            if not any(para.lower().startswith(starter) for starter in analysis_starters):
                final_paragraphs.append(para)
        
        if final_paragraphs:
            return '\n\n'.join(final_paragraphs)
    
    return cleaned_response.strip()


def generate_intelligent_response(state: AgentState) -> AgentState:
    """Generate response using reasoning chain and all data sources collected from tools"""
    logger.info("Node: generate_intelligent_response")
    
    try:
        # Get user query and analysis
        messages = state.get("messages", [])
        if not messages:
            return {**state, "final_response": "No query provided.", "next_step": END}
        
        user_query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        query_analysis = state.get("query_analysis", {})
        tools_needed = query_analysis.get("tools_needed", [])
        categories = query_analysis.get("categories", [])
        
        # Build context based on all collected data from various tools
        context_parts = []
        
        # Add weather data if available
        if state.get("weather_data"):
            weather_data = state["weather_data"]
            if "error" not in weather_data:
                context_parts.append(f"""
🌤️ **LIVE WEATHER DATA for {weather_data.get('location')}:**
- Current Temperature: {weather_data.get('temperature')}
- Feels Like: {weather_data.get('feels_like')} 
- Humidity: {weather_data.get('humidity')}
- Conditions: {weather_data.get('weather_description')}
- Wind Speed: {weather_data.get('wind_speed')}

📅 **3-Day Forecast:**""")
                for forecast in weather_data.get('forecast', []):
                    context_parts.append(f"- {forecast.get('date')}: {forecast.get('max_temp')}/{forecast.get('min_temp')} - {forecast.get('description')}")
        
        # Add market data if available
        if state.get("market_data"):
            market_data = state["market_data"]
            if "error" not in market_data:
                context_parts.append(f"""
💰 **LIVE MARKET DATA:**
- Crop: {market_data.get('crop')}
- Location: {market_data.get('location')}
- Price Range: {market_data.get('price_range')}
- Trend: {market_data.get('trend')}""")
        
        # Add crop recommendation if available
        if state.get("crop_recommendation"):
            crop_data = state["crop_recommendation"]
            if "error" in crop_data:
                context_parts.append(f"""
🌱 **CROP RECOMMENDATION:**
To provide a crop recommendation, I need soil test data:
{crop_data.get('example', 'Please provide: N=90, P=42, K=43, ph=6.5')}""")
            else:
                context_parts.append(f"""
🌱 **CROP RECOMMENDATION:**
- Recommended Crop: {crop_data.get('recommended_crop')}
- Confidence: {crop_data.get('confidence', 'High')}
- Soil Parameters: N={crop_data.get('soil_data', {}).get('N', 'N/A')}, P={crop_data.get('soil_data', {}).get('P', 'N/A')}, K={crop_data.get('soil_data', {}).get('K', 'N/A')}, pH={crop_data.get('soil_data', {}).get('ph', 'N/A')}""")
        
        # Add scheme results if available (from SCHEMES_KB)
        if state.get("scheme_results"):
            scheme_results = state["scheme_results"]
            if scheme_results:
                context_parts.append(f"\n📚 **GOVERNMENT SCHEME RESULTS:** ({len(scheme_results)} schemes found)")
                for i, result in enumerate(scheme_results[:3], 1):
                    context_parts.append(f"{i}. **{result.get('name', 'Scheme')}**")
                    if result.get('purpose or objective of scheme'):
                        context_parts.append(f"   - Objective: {result.get('purpose or objective of scheme')}")
                    elif result.get('objective'):
                        context_parts.append(f"   - Objective: {result.get('objective')}")
                    
                    if result.get('benefits'):
                        context_parts.append(f"   - Benefits: {result.get('benefits')}")
                    
                    if result.get('eligibility'):
                        context_parts.append(f"   - Eligibility: {result.get('eligibility')}")
        
        # Add farming information results if available (from FARMING_KB)
        if state.get("farming_info_results"):
            farming_info_results = state["farming_info_results"]
            # logger.info(f"farming info results: {farming_info_results}")
            if farming_info_results:
                context_parts.append(f"\n🌾 **FARMING INFORMATION:** ({len(farming_info_results)} results found)")
                for i, result in enumerate(farming_info_results):
                    title = result.get('source_file') or result.get('name') or "Farming Information"
                    context_parts.append(f"{i}. **{title}**")
                    
                    # Extract the most relevant content - could be in various fields
                    content = result.get('chunk_text') or result.get('description') or result.get('text')
                    if content:
                        # Limit content length for readability
                        # context_parts.append(f"   {content}..." if len(content) > 250 else f"   {content}")
                        context_parts.append(f"{content}")

        
        # Add Knowledge Graph results if available
        if state.get("knowledge_graph_results"):
            kg_results = state["knowledge_graph_results"]
            if kg_results:
                context_parts.append(f"\n🧠 **KNOWLEDGE GRAPH INSIGHTS:** ({len(kg_results)} results found)")
                for i, result in enumerate(kg_results[:3], 1):
                    result_type = result.get('type', 'knowledge').replace('_', ' ').title()
                    context_parts.append(f"{i}. **{result_type}**")
                    context_parts.append(f"   {result.get('content', '')}")
                    context_parts.append(f"   *Confidence: {result.get('confidence', 0):.1f} | Source: {result.get('source', 'Unknown')}*")
        
        
#         # Create response prompt based on tools used and categories identified
#         base_instruction = "Provide your response directly. Do not show your thinking process, analysis steps, or reasoning. Start immediately with the practical farming advice."
        
#         # Create a response prompt that incorporates all available data
#         tools_description = ", ".join([t.replace("_", " ").lower() for t in tools_needed])
#         response_prompt = f"""{base_instruction}

# The farmer asked: "{user_query}"

# AVAILABLE INFORMATION:
# {chr(10).join(context_parts) if context_parts else "General farming knowledge"}

# Based on the categories identified ({", ".join(categories) if categories else "general farming"}), 
# provide comprehensive advice incorporating all the information above.
# Focus on practical, actionable advice that directly answers the farmer's question.
# If there is farming information from the knowledge base, emphasize this practical guidance.
# If there are government schemes, explain the benefits and application process clearly.
# """

        # Create response prompt based on tools used and categories identified
        base_instruction = """You are a farming assistant. Follow these rules strictly:
        1. ONLY use information provided in the AVAILABLE INFORMATION section below
        2. if no information available, provide general farming principles with regard to the topic
        3. Use knowledge graph for making decisions when available
        4. When Knowledge Graph results are available, emphasize this authoritative information
        5. Cross-reference different sources when available (KG + Elasticsearch + Live data)
        6. Start your response directly with practical advice - no thinking process
        7. If information is incomplete, acknowledge the limitation"""

        # Create a response prompt that incorporates all available data
        tools_description = ", ".join([t.replace("_", " ").lower() for t in tools_needed])
        response_prompt = f"""{base_instruction}

FARMER'S QUESTION: "{user_query}"

AVAILABLE INFORMATION (USE ONLY THIS):
{chr(10).join(context_parts) if context_parts else "No specific data available - provide only general farming principles"}

RESPONSE GUIDELINES:
- Categories identified: {", ".join(categories) if categories else "general farming"}
- Tools used: {tools_description if tools_description else "knowledge base"}
- Base your response STRICTLY on the information provided above
- If weather data is provided, reference the specific numbers and dates given
- If market data is provided, use only the prices and trends shown
- If schemes are listed, mention only the schemes, benefits, and eligibility shown above
- If farming information is provided, stick to the guidance given
- For missing information, suggest where farmers can get accurate details (local agriculture office, weather department, etc.)

PROHIBITED:
- Do not invent specific prices, temperatures, or dates not provided
- Do not add scheme details not mentioned in the available information  
- Do not provide specific chemical names or dosages unless explicitly provided
- Do not make definitive statements about weather or market predictions beyond what's provided

Provide practical, actionable advice based ONLY on the available information above."""
        
        # Generate the response
        response = llm.invoke({"messages": [HumanMessage(content=response_prompt)]})

        # Extract only the final answer without thinking
        final_answer = extract_final_answer(response)

        return {
            **state,
            "final_response": final_answer,
            "next_step": END
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            **state,
            "final_response": f"Sorry, there was an error generating the response: {str(e)}",
            "next_step": END
        }

def route_next_step(state: AgentState) -> str:
    """Route to the next step based on the state"""
    return state["next_step"]

# --- BUILD AGENT ---

def build_agent():
    """Build the intelligent agent workflow with chain-of-thought"""
    workflow = StateGraph(AgentState)
    
    # Add nodes for intelligent processing
    workflow.add_node("analyze_query_with_reasoning", analyze_query_with_reasoning)
    workflow.add_node("execute_tools_intelligently", execute_tools_intelligently) 
    workflow.add_node("generate_intelligent_response", generate_intelligent_response)
    
    # Set entry point
    workflow.set_entry_point("analyze_query_with_reasoning")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze_query_with_reasoning",
        route_next_step,
        {
            "execute_tools_intelligently": "execute_tools_intelligently",
            "generate_response": "generate_intelligent_response"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_tools_intelligently",
        route_next_step,
        {
            "generate_intelligent_response": "generate_intelligent_response"
        }
    )
    
    workflow.add_edge("generate_intelligent_response", END)
    
    return workflow.compile()

def create_agent():
    """Create the agent with the workflow"""
    return build_agent()

def process_query(agent, user_query: str, profile: Optional[FarmerProfile] = None) -> str:
    """Process a user query using the agent"""
    messages = [HumanMessage(content=user_query)]
    
    initial_state = {
        "messages": messages,
        "profile": profile,
        "query_analysis": {},
        "scheme_results": [],
        "crop_recommendation": None,
        "farming_info_results": [],
        "knowledge_graph_results": [],
        "weather_data": None,
        "market_data": None,
        "farmer_context": {},
        "reasoning_chain": "",
        "final_response": "",
        "next_step": ""
    }
    
    try:
        result_state = agent.invoke(initial_state)
        final_response = result_state.get("final_response", "Sorry, I couldn't process your request.")
        
        # Store the interaction if farmer ID exists
        if profile and profile.id:
            # Determine problem type from query
            problem_type = None
            query_lower = user_query.lower()
            if any(word in query_lower for word in ['disease', 'pest', 'problem', 'issue']):
                problem_type = 'agricultural_problem'
            elif any(word in query_lower for word in ['fertilizer', 'nutrition', 'nutrient']):
                problem_type = 'nutrition_advice'
            elif any(word in query_lower for word in ['weather', 'climate']):
                problem_type = 'weather_inquiry'
            elif any(word in query_lower for word in ['price', 'market', 'sell']):
                problem_type = 'market_inquiry'
            else:
                problem_type = 'general_farming'
            
            farmer_tools.store_farmer_interaction(
                farmer_id=profile.id,
                query=user_query,
                response=final_response,
                problem_type=problem_type
            )
            
            logger.info(f"✅ Stored interaction for farmer {profile.id}")
        
        return final_response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Sorry, there was an error processing your request: {str(e)}"

def run_cli():
    """Run the agent as a CLI application"""
    print("🧑‍🌾 Welcome to the Intelligent Farmer Assistant!")
    print("This assistant uses chain-of-thought reasoning to provide the best help.")
    print("\nFirst, let's set up your profile.")
    
    name = input("Enter your name: ")
    contact = input("Enter your contact (optional): ")
    state = input("Enter your state (e.g., Karnataka): ")
    district = input("Enter your district (optional): ")
    crops_str = input("Enter your crops (comma-separated): ")
    
    land_size = None
    while True:
        try:
            land_size_str = input("Enter your land size in acres (optional, press Enter to skip): ")
            if not land_size_str:
                break
            land_size = float(land_size_str)
            break
        except ValueError:
            print("Please enter a valid number.")
    
    profile = FarmerProfile(
        name=name,
        contact=contact,
        state=state,
        district=district,
        crops=[c.strip() for c in crops_str.split(',') if c.strip()],
        land_size_acres=land_size
    )
    
    print("\n✅ Profile saved. You can now start asking questions.")
    print("The assistant will reason about your questions and decide whether to use:")
    print("🌤️  Live weather APIs")
    print("💰 Live market price APIs") 
    print("📚 Knowledge base of schemes")
    print("🔄 Or a combination of sources")
    print("\nType 'exit' or 'quit' to end.\n")
    
    agent = create_agent()
    
    while True:
        user_query = input("Ask about farming, schemes, weather, or prices: ")
        
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye! Happy farming! 🌾")
            break
        
        if not user_query.strip():
            continue
        
        print("\n🧠 Analyzing your question and deciding on the best data sources...")
        
        response = process_query(agent, user_query, profile)
        
        print("\n" + "="*50)
        print("🤖 Assistant's Response:")
        print("="*50)
        print(response)
        print("="*50 + "\n")


if __name__ == "__main__":
    run_cli()