from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphQuery:
    """Interactive querying tool for the agricultural knowledge graph."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: Dict = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def get_agro_climatic_zones(self) -> List[Dict[str, Any]]:
        """Find all agro-climatic zones mentioned in the knowledge base."""
        query = """
        MATCH (zone)
        WHERE zone.subject CONTAINS 'Zone' OR zone.details CONTAINS 'zone'
        OR zone.name CONTAINS 'Zone' OR zone.fact_type = 'Geography'
        RETURN DISTINCT 
            zone.name as zone_name,
            zone.subject as subject,
            zone.details as details,
            zone.fact_type as fact_type
        ORDER BY zone_name
        LIMIT 20
        """
        return self.run_query(query)

    def search_cultivation_practices(self, crop: str = "rice") -> List[Dict[str, Any]]:
        """Find cultivation practices for a specific crop."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE toLower(fact.subject) CONTAINS toLower($crop)
        AND (fact.fact_type = 'CultivationPractice' 
             OR fact.details CONTAINS 'cultivation'
             OR fact.details CONTAINS 'practice'
             OR fact.details CONTAINS 'management')
        RETURN DISTINCT
            fact.subject as crop,
            fact.details as practice_details,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 15
        """
        return self.run_query(query, {"crop": crop})

    def get_soil_management_advice(self) -> List[Dict[str, Any]]:
        """Find soil management and fertility advice."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE fact.fact_type = 'SoilScience'
        OR fact.details CONTAINS 'soil'
        OR fact.details CONTAINS 'fertilizer'
        OR fact.details CONTAINS 'nutrient'
        RETURN DISTINCT
            fact.subject as crop_or_context,
            fact.details as soil_advice,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 20
        """
        return self.run_query(query)

    def find_crop_varieties(self, crop: str = "paddy") -> List[Dict[str, Any]]:
        """Find information about crop varieties."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE (toLower(fact.subject) CONTAINS toLower($crop)
               OR toLower(fact.details) CONTAINS toLower($crop))
        AND (fact.details CONTAINS 'variety'
             OR fact.details CONTAINS 'cultivar'
             OR fact.fact_type = 'CropVariety')
        RETURN DISTINCT
            fact.subject as variety_info,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 15
        """
        return self.run_query(query, {"crop": crop})

    def search_irrigation_techniques(self) -> List[Dict[str, Any]]:
        """Find irrigation and water management techniques."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE fact.details CONTAINS 'irrigation'
        OR fact.details CONTAINS 'water'
        OR fact.details CONTAINS 'AWMD'
        OR fact.details CONTAINS 'moisture'
        RETURN DISTINCT
            fact.subject as context,
            fact.details as irrigation_info,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 15
        """
        return self.run_query(query)

    def get_relationships_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """Find all relationships involving a specific entity."""
        query = """
        MATCH (e:Entity {name: $entity_name})-[r]-(other)
        RETURN 
            e.name as main_entity,
            type(r) as relationship_type,
            other.name as related_entity,
            labels(other) as related_labels
        ORDER BY relationship_type
        LIMIT 20
        """
        return self.run_query(query, {"entity_name": entity_name})

    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search across all agricultural facts by keyword."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE toLower(fact.details) CONTAINS toLower($keyword)
        OR toLower(fact.subject) CONTAINS toLower($keyword)
        RETURN DISTINCT
            fact.subject as subject,
            fact.details as details,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 20
        """
        return self.run_query(query, {"keyword": keyword})

    def get_seasonal_information(self, season: str = "monsoon") -> List[Dict[str, Any]]:
        """Find seasonal farming information."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE toLower(fact.details) CONTAINS toLower($season)
        OR fact.details CONTAINS 'season'
        OR fact.details CONTAINS 'Kharif'
        OR fact.details CONTAINS 'Rabi'
        RETURN DISTINCT
            fact.subject as context,
            fact.details as seasonal_info,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 15
        """
        return self.run_query(query, {"season": season})

    def find_disease_and_pest_info(self) -> List[Dict[str, Any]]:
        """Find disease and pest management information."""
        query = """
        MATCH (fact:AgronomicFact)
        WHERE fact.details CONTAINS 'disease'
        OR fact.details CONTAINS 'pest'
        OR fact.details CONTAINS 'pathogen'
        OR fact.details CONTAINS 'insect'
        OR fact.fact_type = 'PlantProtection'
        RETURN DISTINCT
            fact.subject as crop_context,
            fact.details as protection_info,
            fact.fact_type as category,
            fact.source_file as source
        ORDER BY fact.subject
        LIMIT 15
        """
        return self.run_query(query)

    def interactive_query_session(self):
        """Run an interactive query session."""
        print("🌾 Agricultural Knowledge Graph Query System")
        print("=" * 50)
        
        queries = {
            "1": ("Agro-climatic Zones", self.get_agro_climatic_zones),
            "2": ("Cultivation Practices", lambda: self.search_cultivation_practices("rice")),
            "3": ("Soil Management", self.get_soil_management_advice),
            "4": ("Crop Varieties", lambda: self.find_crop_varieties("paddy")),
            "5": ("Irrigation Techniques", self.search_irrigation_techniques),
            "6": ("Seasonal Information", lambda: self.get_seasonal_information("monsoon")),
            "7": ("Disease & Pest Management", self.find_disease_and_pest_info),
            "8": ("Custom Keyword Search", self._custom_search)
        }
        
        while True:
            print("\nAvailable Queries:")
            for key, (desc, _) in queries.items():
                print(f"  {key}. {desc}")
            print("  q. Quit")
            
            choice = input("\nEnter your choice (1-8 or q): ").strip()
            
            if choice.lower() == 'q':
                break
                
            if choice in queries:
                desc, func = queries[choice]
                print(f"\n--- {desc} ---")
                try:
                    results = func()
                    self._display_results(results)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Invalid choice. Please try again.")

    def _custom_search(self):
        """Handle custom keyword search."""
        keyword = input("Enter search keyword: ").strip()
        if keyword:
            return self.search_by_keyword(keyword)
        return []

    def _display_results(self, results: List[Dict[str, Any]], max_results: int = 10):
        """Display query results in a formatted way."""
        if not results:
            print("No results found.")
            return
            
        print(f"\nFound {len(results)} results (showing first {min(len(results), max_results)}):\n")
        
        for i, result in enumerate(results[:max_results], 1):
            print(f"{i}. ", end="")
            for key, value in result.items():
                if value and str(value).strip():
                    # Truncate long values
                    display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"{key}: {display_value}")
            print("-" * 80)

def main():
    """Main function to run the query system."""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "test1234"  # Replace with your password
    
    querier = KnowledgeGraphQuery(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        querier.interactive_query_session()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        querier.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()