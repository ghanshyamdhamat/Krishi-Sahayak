from neo4j import GraphDatabase
from typing import List, Dict, Optional
from enum import Enum
import logging
from neo4j_files.intelligent_search import IntelligentKGQueryPlanner, QueryPlan, SearchStrategy
from typing import Any
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KGResult:
    """Standardized result format from Knowledge Graph queries"""
    content: str
    source: str
    confidence: float
    metadata: Dict[str, Any]
    relationships: List[Dict[str, str]] = None

class EnhancedKnowledgeGraphService:
    """
    Enhanced KG service that uses intelligent query planning
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.query_planner = IntelligentKGQueryPlanner()
        self._verify_connection()
    
    def _verify_connection(self):
        # Simple check: try to run a trivial query
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            print(f"Knowledge Graph connection failed: {e}")
            return False
        
    def _search_farmer_specific_knowledge(self, farmer_id: str, query: str, limit: int = 3) -> List[KGResult]:
        """Search for farmer-specific knowledge and experiences"""
        with self.driver.session() as session:
            results = []
            
            try:
                # Search for farmer's previous interactions/experiences
                cypher_query = """
                MATCH (farmer:Farmer {id: $farmer_id})-[:HAS_EXPERIENCE]->(exp:Experience)
                WHERE toLower(exp.description) CONTAINS toLower($query_term)
                OR toLower(exp.topic) CONTAINS toLower($query_term)
                OR toLower(exp.problem) CONTAINS toLower($query_term)
                RETURN exp.description as content,
                    exp.topic as topic,
                    exp.problem as problem,
                    exp.solution as solution,
                    exp.outcome as outcome,
                    exp.created_date as date
                ORDER BY exp.created_date DESC
                LIMIT $limit
                """
                
                records = session.run(cypher_query, farmer_id=farmer_id, query_term=query, limit=limit)
                
                for record in records:
                    content = f"Your Previous Experience: {record['content']}"
                    if record['solution']:
                        content += f"\nSolution Applied: {record['solution']}"
                    if record['outcome']:
                        content += f"\nOutcome: {record['outcome']}"
                    
                    results.append(KGResult(
                        content=content,
                        source='Your Personal Experience',
                        confidence=0.95,  # High confidence for personal experience
                        metadata={
                            'type': 'farmer_specific_experience',
                            'farmer_id': farmer_id,
                            'topic': record['topic'],
                            'date': record['date']
                        }
                    ))
            
                # Search for location/crop specific advice for this farmer
                farmer_context_query = """
                MATCH (farmer:Farmer {id: $farmer_id})
                WITH farmer
                MATCH (advice:AgronomicFact)
                WHERE (toLower(advice.details) CONTAINS toLower(farmer.state)
                    OR toLower(advice.details) CONTAINS toLower(farmer.district))
                AND (toLower(advice.details) CONTAINS toLower($query_term))
                RETURN advice.details as content,
                    advice.source_file as source,
                    advice.fact_type as fact_type
                LIMIT $limit
                """
                
                records = session.run(farmer_context_query, farmer_id=farmer_id, query_term=query, limit=limit)
                
                for record in records:
                    results.append(KGResult(
                        content=record['content'],
                        source=record['source'] or 'Localized Knowledge',
                        confidence=0.85,
                        metadata={
                            'type': 'location_specific_advice',
                            'farmer_id': farmer_id,
                            'fact_type': record['fact_type']
                        }
                    ))
                    
            except Exception as e:
                logger.error(f"Error in farmer-specific search: {e}")
            
            return results

    def intelligent_search(self, query: str, farmer_profile: Dict = None, 
                          top_k: int = 5) -> List[KGResult]:
        """
        Intelligent search using query planning
        """
        # Create search plan
        plan = self.query_planner.plan_search(query, farmer_profile)
        farmer_id = farmer_profile.get('id') if farmer_profile else None
    
        logger.info(f"🧠 Search Strategy: {plan.primary_strategy.value}")
        logger.info(f"🔍 Entities to search: {plan.entities_to_search}")
        logger.info(f"👤 Farmer ID: {farmer_id}")

        results = []

        if farmer_id:
            personalized_results = self._search_farmer_specific_knowledge(farmer_id, query, top_k//2)
            results.extend(personalized_results)
            logger.info(f"✅ Found {len(personalized_results)} farmer-specific results")

        # Execute search based on strategy
        if plan.primary_strategy == SearchStrategy.SYMPTOM_DIAGNOSIS:
            general_results = self._execute_symptom_diagnosis_search(plan, query)
        elif plan.primary_strategy == SearchStrategy.PROBLEM_SOLUTION_CHAIN:
            general_results = self._execute_problem_solution_search(plan, query)
        elif plan.primary_strategy == SearchStrategy.CROP_LIFECYCLE_SEARCH:
            general_results = self._execute_crop_management_search(plan, query)
        elif plan.primary_strategy == SearchStrategy.ENTITY_RELATIONSHIP_TRAVERSAL:
            general_results = self._execute_relationship_traversal_search(plan, query)
        else:
            general_results = self._execute_semantic_search(plan, query)
        
        results.extend(general_results)
        
        # Remove duplicates and sort by confidence
        seen_content = set()
        unique_results = []
        for result in results:
            if result.content not in seen_content:
                seen_content.add(result.content)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.confidence, reverse=True)
        return unique_results[:top_k]
    
    def store_farmer_interaction(self, farmer_id: str, query: str, response: str, 
                           problem_type: str = None, solution_applied: str = None):
        """Store farmer's interaction for future personalized responses"""
        with self.driver.session() as session:
            try:
                # Create or update farmer node
                cypher_query = """
                MERGE (farmer:Farmer {id: $farmer_id})
                SET farmer.last_interaction = datetime()
                
                CREATE (interaction:Interaction {
                    id: randomUUID(),
                    query: $query,
                    response: $response,
                    problem_type: $problem_type,
                    solution_applied: $solution_applied,
                    created_date: datetime()
                })
                
                MERGE (farmer)-[:HAS_INTERACTION]->(interaction)
                """
                
                session.run(cypher_query,
                        farmer_id=farmer_id,
                        query=query,
                        response=response,
                        problem_type=problem_type,
                        solution_applied=solution_applied)
                
                logger.info(f"✅ Stored interaction for farmer {farmer_id}")
                
            except Exception as e:
                logger.error(f"Error storing farmer interaction: {e}")

    def create_farmer_profile_in_kg(self, farmer_profile: Dict):
        """Create or update farmer profile in knowledge graph"""
        with self.driver.session() as session:
            try:
                cypher_query = """
                MERGE (farmer:Farmer {id: $farmer_id})
                SET farmer.name = $name,
                    farmer.contact = $contact,
                    farmer.state = $state,
                    farmer.district = $district,
                    farmer.village = $village,
                    farmer.country = $country,
                    farmer.taluka = $taluka,
                    farmer.land_size_acres = $land_size_acres,
                    farmer.preferred_language = $preferred_language,
                    farmer.updated_at = datetime()
                
                // Remove existing crop relationships
                MATCH (farmer)-[r:GROWS]->(crop:Crop)
                DELETE r
                
                // Add crop relationships
                WITH farmer
                UNWIND $crops as crop_name
                MERGE (crop:Crop {name: crop_name})
                MERGE (farmer)-[:GROWS]->(crop)
                """
                
                session.run(cypher_query,
                        farmer_id=farmer_profile.get('id'),
                        name=farmer_profile.get('name'),
                        contact=farmer_profile.get('contact'),
                        state=farmer_profile.get('state'),
                        district=farmer_profile.get('district'),
                        village=farmer_profile.get('village'),
                        country=farmer_profile.get('country', 'India'),
                        taluka=farmer_profile.get('taluka'),
                        land_size_acres=farmer_profile.get('land_size_acres'),
                        preferred_language=farmer_profile.get('preferred_language', 'English'),
                        crops=farmer_profile.get('crops', []))
                
                logger.info(f"✅ Created/updated farmer profile in KG for {farmer_profile.get('id')}")
                
            except Exception as e:
                logger.error(f"Error creating farmer profile in KG: {e}")

    def _execute_symptom_diagnosis_search(self, plan: QueryPlan, query: str) -> List[KGResult]:
        """Execute symptom-based diagnosis search"""
        with self.driver.session() as session:
            results = []
            
            # Search for symptoms and related problems
            for entity in plan.entities_to_search:
                cypher_query = """
                MATCH (problem:AgronomicFact)-[:DESCRIBES]->(crop:Entity)
                WHERE toLower(crop.name) CONTAINS toLower($entity)
                  AND (toLower(problem.details) CONTAINS 'symptom' 
                       OR toLower(problem.details) CONTAINS 'disease'
                       OR toLower(problem.details) CONTAINS 'pest')
                WITH problem, crop
                OPTIONAL MATCH (solution:AgronomicFact)-[:DESCRIBES]->(crop)
                WHERE toLower(solution.details) CONTAINS 'treatment'
                   OR toLower(solution.details) CONTAINS 'control'
                   OR toLower(solution.details) CONTAINS 'manage'
                RETURN problem.details as problem_description,
                       solution.details as solution_description,
                       crop.name as affected_crop,
                       problem.source_file as source
                LIMIT 3
                """
                
                records = session.run(cypher_query, entity=entity)
                
                for record in records:
                    content = record['problem_description']
                    if record['solution_description']:
                        content += f"\nTreatment: {record['solution_description']}"
                    
                    results.append(KGResult(
                        content=content,
                        source=record['source'] or 'Knowledge Base',
                        confidence=0.85,
                        metadata={
                            'type': 'symptom_diagnosis',
                            'affected_crop': record['affected_crop'],
                            'search_entity': entity
                        }
                    ))
            
            return results
        
    def get_farmer_history(self, farmer_id: str, limit: int = 10) -> List[Dict]:
        """Get farmer's interaction history"""
        with self.driver.session() as session:
            try:
                cypher_query = """
                MATCH (farmer:Farmer {id: $farmer_id})-[:HAS_INTERACTION]->(interaction:Interaction)
                RETURN interaction.query as query,
                    interaction.response as response,
                    interaction.problem_type as problem_type,
                    interaction.solution_applied as solution_applied,
                    interaction.created_date as date
                ORDER BY interaction.created_date DESC
                LIMIT $limit
                """
                
                records = session.run(cypher_query, farmer_id=farmer_id, limit=limit)
                
                history = []
                for record in records:
                    history.append({
                        'query': record['query'],
                        'response': record['response'],
                        'problem_type': record['problem_type'],
                        'solution_applied': record['solution_applied'],
                        'date': str(record['date'])
                    })
                
                return history
                
            except Exception as e:
                logger.error(f"Error getting farmer history: {e}")
                return []

    def _execute_problem_solution_search(self, plan: QueryPlan, query: str) -> List[KGResult]:
        """Execute problem-solution chain search"""
        with self.driver.session() as session:
            results = []
            
            # Build dynamic relationship query
            relationship_filter = " OR ".join([f"type(r) = '{rel}'" for rel in plan.relationships_to_follow])
            
            for entity in plan.entities_to_search:
                cypher_query = f"""
                MATCH (problem)-[r]->(solution)
                WHERE ({relationship_filter})
                  AND (toLower(problem.details) CONTAINS toLower($entity)
                       OR toLower(solution.details) CONTAINS toLower($entity))
                  AND problem.class_name CONTAINS 'Fact'
                  AND solution.class_name CONTAINS 'Fact'
                RETURN problem.details as problem_info,
                       solution.details as solution_info,
                       type(r) as relationship_type,
                       problem.source_file as source
                ORDER BY size(solution.details) DESC
                LIMIT 3
                """
                
                records = session.run(cypher_query, entity=entity)
                
                for record in records:
                    content = f"Problem: {record['problem_info']}\n"
                    content += f"Solution: {record['solution_info']}\n"
                    content += f"Relationship: {record['relationship_type']}"
                    
                    results.append(KGResult(
                        content=content,
                        source=record['source'] or 'Knowledge Base',
                        confidence=0.9,
                        metadata={
                            'type': 'problem_solution_chain',
                            'relationship': record['relationship_type'],
                            'search_entity': entity
                        }
                    ))
            
            return results

    def _execute_crop_management_search(self, plan: QueryPlan, query: str) -> List[KGResult]:
        """Execute crop management search"""
        with self.driver.session() as session:
            results = []
            
            for entity in plan.entities_to_search:
                cypher_query = """
                MATCH (practice:CultivationPractice)-[:DESCRIBES]->(crop:Entity)
                WHERE toLower(crop.name) CONTAINS toLower($entity)
                  OR toLower(practice.details) CONTAINS toLower($entity)
                WITH practice, crop
                OPTIONAL MATCH (practice)-[:REQUIRES]->(resource:Entity)
                RETURN practice.details as practice_info,
                       crop.name as crop_name,
                       practice.fact_type as practice_type,
                       collect(resource.name) as required_resources,
                       practice.source_file as source
                ORDER BY size(practice.details) DESC
                LIMIT 3
                """
                
                records = session.run(cypher_query, entity=entity)
                
                for record in records:
                    content = record['practice_info']
                    if record['required_resources'] and any(record['required_resources']):
                        content += f"\nRequired resources: {', '.join(filter(None, record['required_resources']))}"
                    
                    results.append(KGResult(
                        content=content,
                        source=record['source'] or 'Knowledge Base',
                        confidence=0.8,
                        metadata={
                            'type': 'crop_management',
                            'crop': record['crop_name'],
                            'practice_type': record['practice_type'],
                            'search_entity': entity
                        }
                    ))
            
            return results

    def _execute_semantic_search(self, plan: QueryPlan, query: str) -> List[KGResult]:
        """Execute semantic similarity search as fallback"""
        with self.driver.session() as session:
            # Use text similarity on all agronomic facts
            cypher_query = """
            MATCH (fact:AgronomicFact)
            WHERE size(fact.details) > 50
              AND (toLower(fact.details) CONTAINS toLower($search_term)
                   OR toLower(fact.subject) CONTAINS toLower($search_term))
            RETURN fact.details as content,
                   fact.subject as subject,
                   fact.fact_type as fact_type,
                   fact.source_file as source
            ORDER BY size(fact.details) DESC
            LIMIT 5
            """
            
            # Extract key terms from query for search
            search_term = self._extract_key_terms(query)
            
            records = session.run(cypher_query, search_term=search_term)
            
            results = []
            for record in records:
                results.append(KGResult(
                    content=record['content'],
                    source=record['source'] or 'Knowledge Base',
                    confidence=0.6,
                    metadata={
                        'type': 'semantic_search',
                        'subject': record['subject'],
                        'fact_type': record['fact_type']
                    }
                ))
            
            return results

    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms from query for semantic search"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'how', 'what', 'when', 'where', 'why', 'the', 'is', 'are', 'can', 'should', 'to', 'for', 'in', 'on', 'of'}
        words = query.lower().split()
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(key_terms[:3])  # Use top 3 key terms


# Initialize as singleton
_kg_service = None

def get_kg_service() -> Optional[EnhancedKnowledgeGraphService]:
    """Get the global knowledge graph service instance"""
    global _kg_service
    if _kg_service is None:
        try:
            _kg_service = EnhancedKnowledgeGraphService(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "test1234")
            )
        except Exception as e:
            logger.warning(f"Knowledge Graph service not available: {e}")
            return None
    return _kg_service