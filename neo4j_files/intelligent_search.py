from typing import List, Dict, Any, Set, Tuple
import re
from dataclasses import dataclass
from enum import Enum

class SearchStrategy(Enum):
    DIRECT_ENTITY_MATCH = "direct_entity_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PROBLEM_SOLUTION_CHAIN = "problem_solution_chain"
    ENTITY_RELATIONSHIP_TRAVERSAL = "entity_relationship_traversal"
    CROP_LIFECYCLE_SEARCH = "crop_lifecycle_search"
    SYMPTOM_DIAGNOSIS = "symptom_diagnosis"

@dataclass
class QueryPlan:
    """Plan for how to search the knowledge graph"""
    primary_strategy: SearchStrategy
    entities_to_search: List[str]
    relationships_to_follow: List[str]
    node_types_to_focus: List[str]
    confidence_threshold: float
    max_hops: int
    search_params: Dict[str, Any]

class IntelligentKGQueryPlanner:
    """
    Plans knowledge graph searches based on query analysis.
    Uses agricultural domain knowledge to make smart decisions about what to search.
    """
    
    def __init__(self):
        # Agricultural domain knowledge for intelligent planning
        self.agricultural_entities = {
            'crops': {
                'rice': ['paddy', 'basmati', 'jasmine rice', 'brown rice'],
                'wheat': ['durum wheat', 'spring wheat', 'winter wheat'],
                'maize': ['corn', 'sweet corn', 'field corn'],
                'cotton': ['bt cotton', 'organic cotton'],
                'sugarcane': ['sugar cane', 'cane'],
                'soybean': ['soya', 'soy'],
                'tomato': ['tomatoes'],
                'potato': ['potatoes', 'aloo']
            },
            'problems': {
                'diseases': ['blast', 'blight', 'rust', 'wilt', 'rot', 'spot', 'mosaic', 'yellowing'],
                'pests': ['aphid', 'thrips', 'borer', 'caterpillar', 'mite', 'weevil', 'hopper', 'fly'],
                'deficiencies': ['nitrogen', 'phosphorus', 'potassium', 'iron', 'zinc', 'magnesium'],
                'environmental': ['drought', 'flood', 'heat stress', 'cold damage', 'wind damage']
            },
            'treatments': {
                'chemical': ['fungicide', 'pesticide', 'herbicide', 'fertilizer', 'spray'],
                'biological': ['neem', 'bt spray', 'beneficial insects', 'bio pesticide'],
                'cultural': ['crop rotation', 'intercropping', 'mulching', 'pruning'],
                'nutritional': ['urea', 'dap', 'potash', 'micronutrient', 'organic manure']
            },
            'activities': {
                'planting': ['sowing', 'transplanting', 'seeding', 'planting'],
                'maintenance': ['weeding', 'irrigation', 'fertilizing', 'pruning'],
                'harvest': ['harvesting', 'threshing', 'drying', 'storage'],
                'preparation': ['land preparation', 'tilling', 'plowing']
            }
        }
        
        # Relationship patterns for traversal
        self.relationship_patterns = {
            'problem_solution': ['TREATS', 'CONTROLS', 'PREVENTS', 'CURES', 'MANAGES'],
            'crop_requirements': ['REQUIRES', 'NEEDS', 'BENEFITS_FROM', 'GROWS_IN'],
            'temporal': ['BEFORE', 'AFTER', 'DURING', 'FOLLOWS'],
            'causal': ['CAUSES', 'LEADS_TO', 'RESULTS_IN', 'TRIGGERS'],
            'spatial': ['GROWS_IN', 'FOUND_IN', 'AFFECTS', 'SPREADS_TO']
        }

    def plan_search(self, query: str, farmer_profile: Dict = None) -> QueryPlan:
        """
        Create an intelligent search plan based on query analysis.
        
        Args:
            query: User's agricultural question
            farmer_profile: Farmer's context (crops, location, etc.)
            
        Returns:
            QueryPlan with strategy and parameters
        """
        query_lower = query.lower()
        
        # Extract entities from query
        entities = self._extract_agricultural_entities(query)
        
        # Determine query intent
        intent = self._classify_query_intent(query_lower, entities)
        
        # Create search plan based on intent
        if intent == 'problem_diagnosis':
            return self._plan_problem_diagnosis_search(query_lower, entities, farmer_profile)
        elif intent == 'solution_seeking':
            return self._plan_solution_search(query_lower, entities, farmer_profile)
        elif intent == 'crop_management':
            return self._plan_crop_management_search(query_lower, entities, farmer_profile)
        elif intent == 'general_knowledge':
            return self._plan_general_knowledge_search(query_lower, entities, farmer_profile)
        else:
            return self._plan_fallback_search(query_lower, entities, farmer_profile)

    def _extract_agricultural_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract agricultural entities from the query using domain knowledge."""
        query_lower = query.lower()
        found_entities = {
            'crops': [],
            'problems': [],
            'treatments': [],
            'activities': []
        }
        
        # Search for crops and their aliases
        for crop, aliases in self.agricultural_entities['crops'].items():
            if crop in query_lower or any(alias in query_lower for alias in aliases):
                found_entities['crops'].append(crop)
        
        # Search for problems
        for category, problems in self.agricultural_entities['problems'].items():
            for problem in problems:
                if problem in query_lower:
                    found_entities['problems'].append(problem)
        
        # Search for treatments
        for category, treatments in self.agricultural_entities['treatments'].items():
            for treatment in treatments:
                if treatment in query_lower:
                    found_entities['treatments'].append(treatment)
        
        # Search for activities
        for category, activities in self.agricultural_entities['activities'].items():
            for activity in activities:
                if activity in query_lower:
                    found_entities['activities'].append(activity)
        
        return found_entities

    def _classify_query_intent(self, query_lower: str, entities: Dict[str, List[str]]) -> str:
        """Classify the intent of the agricultural query."""
        
        # Problem diagnosis keywords
        problem_keywords = ['problem', 'disease', 'pest', 'issue', 'symptoms', 'yellowing', 'wilting', 
                           'spots', 'damage', 'dying', 'infected', 'infestation', 'what is wrong']
        
        # Solution seeking keywords
        solution_keywords = ['how to', 'treat', 'control', 'cure', 'prevent', 'manage', 'solution', 
                           'remedy', 'fix', 'stop', 'get rid of']
        
        # Crop management keywords
        management_keywords = ['when to', 'how much', 'fertilizer', 'irrigation', 'sowing', 'harvest',
                             'planting', 'cultivation', 'growing', 'care', 'maintenance']
        
        if any(keyword in query_lower for keyword in problem_keywords) or entities['problems']:
            return 'problem_diagnosis'
        elif any(keyword in query_lower for keyword in solution_keywords):
            return 'solution_seeking'
        elif any(keyword in query_lower for keyword in management_keywords) or entities['activities']:
            return 'crop_management'
        else:
            return 'general_knowledge'

    def _plan_problem_diagnosis_search(self, query_lower: str, entities: Dict, 
                                     farmer_profile: Dict) -> QueryPlan:
        """Plan search for problem diagnosis queries."""
        
        # Focus on crops mentioned + farmer's crops
        search_entities = entities['crops'].copy()
        if farmer_profile and 'crops' in farmer_profile:
            search_entities.extend([crop.lower() for crop in farmer_profile['crops']])
        
        # Add problem entities
        search_entities.extend(entities['problems'])
        
        return QueryPlan(
            primary_strategy=SearchStrategy.SYMPTOM_DIAGNOSIS,
            entities_to_search=list(set(search_entities)),
            relationships_to_follow=['AFFECTS', 'CAUSES', 'SYMPTOMS', 'FOUND_IN'],
            node_types_to_focus=['AgronomicFact', 'CultivationPractice', 'Entity'],
            confidence_threshold=0.7,
            max_hops=2,
            search_params={
                'focus_on_problems': True,
                'include_symptoms': True,
                'search_treatments': True,
                'regional_filter': farmer_profile.get('state') if farmer_profile else None
            }
        )

    def _plan_solution_search(self, query_lower: str, entities: Dict, 
                            farmer_profile: Dict) -> QueryPlan:
        """Plan search for solution-seeking queries."""
        
        search_entities = entities['crops'] + entities['problems'] + entities['treatments']
        
        return QueryPlan(
            primary_strategy=SearchStrategy.PROBLEM_SOLUTION_CHAIN,
            entities_to_search=search_entities,
            relationships_to_follow=['TREATS', 'CONTROLS', 'PREVENTS', 'MANAGES', 'CURES'],
            node_types_to_focus=['AgronomicFact', 'CultivationPractice'],
            confidence_threshold=0.8,
            max_hops=3,
            search_params={
                'prioritize_solutions': True,
                'include_dosage': True,
                'include_timing': True,
                'include_application_method': True
            }
        )

    def _plan_crop_management_search(self, query_lower: str, entities: Dict,
                                   farmer_profile: Dict) -> QueryPlan:
        """Plan search for crop management queries."""
        
        search_entities = entities['crops'] + entities['activities']
        if farmer_profile and 'crops' in farmer_profile:
            search_entities.extend([crop.lower() for crop in farmer_profile['crops']])
        
        return QueryPlan(
            primary_strategy=SearchStrategy.CROP_LIFECYCLE_SEARCH,
            entities_to_search=list(set(search_entities)),
            relationships_to_follow=['REQUIRES', 'BENEFITS_FROM', 'GROWS_IN', 'NEEDS'],
            node_types_to_focus=['AgronomicFact', 'CultivationPractice'],
            confidence_threshold=0.6,
            max_hops=2,
            search_params={
                'include_seasonal_info': True,
                'include_best_practices': True,
                'regional_specific': True
            }
        )

    def _plan_general_knowledge_search(self, query_lower: str, entities: Dict,
                                     farmer_profile: Dict) -> QueryPlan:
        """Plan search for general knowledge queries."""
        
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        return QueryPlan(
            primary_strategy=SearchStrategy.SEMANTIC_SIMILARITY,
            entities_to_search=all_entities,
            relationships_to_follow=['RELATED_TO', 'IS_A', 'PART_OF', 'CONTAINS'],
            node_types_to_focus=['AgronomicFact', 'CultivationPractice', 'Entity'],
            confidence_threshold=0.5,
            max_hops=2,
            search_params={
                'broad_search': True,
                'include_related_concepts': True
            }
        )

    def _plan_fallback_search(self, query_lower: str, entities: Dict,
                            farmer_profile: Dict) -> QueryPlan:
        """Fallback search plan."""
        
        return QueryPlan(
            primary_strategy=SearchStrategy.SEMANTIC_SIMILARITY,
            entities_to_search=[],
            relationships_to_follow=[],
            node_types_to_focus=['AgronomicFact'],
            confidence_threshold=0.4,
            max_hops=1,
            search_params={'fallback_mode': True}
        )