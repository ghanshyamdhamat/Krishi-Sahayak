from neo4j import GraphDatabase
import json
import os
import re
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StoreJSONData:
    """Enhanced class to store PDF extraction data into a Neo4j knowledge graph.
    
    Creates a hierarchical structure:
    Document -> Chunk -> Extraction (with rich metadata) -> Relationships
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_indexes()

    def close(self):
        self.driver.close()

    def _create_indexes(self):
        """Create database indexes for better query performance."""
        indexes = [
            "CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source_file)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX fact_type IF NOT EXISTS FOR (f:AgronomicFact) ON (f.fact_type)",
            "CREATE INDEX fact_subject IF NOT EXISTS FOR (f:AgronomicFact) ON (f.subject)",
            "CREATE INDEX zone_name IF NOT EXISTS FOR (z:AgroClimaticZone) ON (z.name)"
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created/verified index: {index.split('FOR')[1].split('ON')[0].strip()}")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")

    def store_data_from_file(self, json_file_path: str):
        """
        Reads a JSON file containing extracted data and stores it in Neo4j.
        Creates hierarchical structure: Document -> Chunks -> Extractions -> Relationships
        """
        if not os.path.exists(json_file_path):
            logger.error(f"JSON file not found at: {json_file_path}")
            return

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("JSON file should contain a list of objects.")
            return

        # Extract document name from first chunk
        document_name = data[0].get("source_file", "unknown_document") if data else "unknown_document"
        
        with self.driver.session() as session:
            # Create document node
            session.execute_write(self._create_document, document_name)
            
            # Process each chunk
            for i, chunk in enumerate(data):
                logger.info(f"Processing chunk {i+1}/{len(data)} from {os.path.basename(json_file_path)}")
                try:
                    session.execute_write(self._create_chunk_and_extractions, chunk, document_name)
                except Exception as e:
                    logger.error(f"Failed to process chunk {i}: {e}")
        
        logger.info(f"✅ Successfully processed and stored data from {json_file_path}")

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        with self.driver.session() as session:
            stats = {}
            
            # Node counts by type
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
            """).data()
            
            stats['node_counts'] = {str(record['labels']): record['count'] for record in node_counts}
            
            # Relationship counts
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            stats['relationship_counts'] = {record['relationship_type']: record['count'] for record in rel_counts}
            
            # Fact type distribution
            fact_types = session.run("""
                MATCH (f:AgronomicFact)
                RETURN f.fact_type as fact_type, count(f) as count
                ORDER BY count DESC
            """).data()
            
            stats['fact_types'] = {record['fact_type'] or 'Unknown': record['count'] for record in fact_types}
            
            return stats

    @staticmethod
    def _create_document(tx, document_name: str):
        """Create a document node."""
        tx.run("""
            MERGE (d:Document {title: $title})
            SET d.created_at = datetime()
        """, title=document_name)

    @staticmethod
    def _create_chunk_and_extractions(tx, chunk_data: dict, document_name: str):
        """
        Creates a chunk node and all its extractions with rich metadata.
        Links: Document -> Chunk -> Extractions -> Relationships
        """
        source_file = chunk_data.get("source_file")
        chunk_index = chunk_data.get("chunk_index")
        
        if source_file is None or chunk_index is None:
            logger.warning("Skipping chunk due to missing 'source_file' or 'chunk_index'.")
            return

        chunk_id = f"{os.path.basename(source_file)}_{chunk_index}"

        # Create chunk and link to document
        tx.run("""
            MATCH (d:Document {title: $document_name})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.source_file = $source_file,
                c.chunk_index = $chunk_index,
                c.text = $text,
                c.classification = $classification
            MERGE (d)-[:HAS_CHUNK]->(c)
        """, 
        document_name=document_name,
        chunk_id=chunk_id,
        source_file=source_file,
        chunk_index=chunk_index,
        text=chunk_data.get("chunk_text"),
        classification=chunk_data.get("chunk_classification"))

        # Process extractions
        for extraction in chunk_data.get("extractions", []):
            StoreJSONData._create_extraction_node(tx, extraction, chunk_id)

    @staticmethod
    def _create_extraction_node(tx, extraction: dict, chunk_id: str):
        """
        Create extraction nodes with rich metadata and their relationships.
        """
        class_name = extraction.get("class_name", "Entity")
        fact_type = extraction.get("fact_type", "General")
        subject = extraction.get("subject", "Unknown")
        details = extraction.get("details", "")
        source_file = extraction.get("source_file", "")
        content_type = extraction.get("content_type", "text")
        
        # Sanitize class name for Neo4j label
        node_label = StoreJSONData._sanitize_label(class_name)
        
        # Create extraction node with rich properties
        extraction_id = f"{chunk_id}_{hash(details) % 10000}"
        
        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})
        MERGE (e:{node_label} {{id: $extraction_id}})
        SET e.fact_type = $fact_type,
            e.subject = $subject,
            e.details = $details,
            e.source_file = $source_file,
            e.content_type = $content_type,
            e.class_name = $class_name
        MERGE (c)-[:CONTAINS_EXTRACTION]->(e)
        """
        
        tx.run(query,
               chunk_id=chunk_id,
               extraction_id=extraction_id,
               fact_type=fact_type,
               subject=subject,
               details=details,
               source_file=source_file,
               content_type=content_type,
               class_name=class_name)

        # Create relationships from this extraction
        for triple in extraction.get("relationships", []):
            StoreJSONData._create_relationship(tx, triple, extraction_id, node_label)

    @staticmethod
    def _create_relationship(tx, triple: dict, extraction_id: str, extraction_label: str):
        """Create relationships between entities."""
        subject = triple.get("subject")
        predicate = triple.get("predicate")
        obj = triple.get("object")

        if not all([subject, predicate, obj]):
            logger.warning(f"Skipping incomplete triple: {triple}")
            return

        rel_type = StoreJSONData._sanitize_relationship_type(predicate)

        # Create subject and object entities, then relationship
        query = f"""
        MATCH (extraction:{extraction_label} {{id: $extraction_id}})
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[r:{rel_type}]->(o)
        MERGE (extraction)-[:DESCRIBES]->(s)
        MERGE (extraction)-[:DESCRIBES]->(o)
        """
        
        tx.run(query, 
               extraction_id=extraction_id,
               subject=subject, 
               object=obj)

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Sanitizes a string to be a valid Neo4j label."""
        # Replace common patterns
        label = label.replace("-", "_").replace(" ", "_")
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', label)
        
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "Entity_" + sanitized
        return sanitized

    @staticmethod
    def _sanitize_relationship_type(rel_type: str) -> str:
        """Sanitizes a string to be a valid Neo4j relationship type."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type).upper()
        return sanitized

    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("🗑️  Database cleared!")

if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "test1234"  # Replace with your Neo4j password
    
    JSON_DIR = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Capital_One/capital_one_agent_ai/backend/neo4j_files"

    storer = StoreJSONData(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Find all extraction JSON files
        json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('_extractions.json')]
        if not json_files:
            logger.warning(f"No extraction JSON files found in {JSON_DIR}")
        else:
            logger.info(f"Found {len(json_files)} JSON files to process")
            
            for json_file in json_files:
                file_path = os.path.join(JSON_DIR, json_file)
                logger.info(f"--- Processing file: {json_file} ---")
                storer.store_data_from_file(file_path)
            
            # Display statistics
            logger.info("=== KNOWLEDGE GRAPH STATISTICS ===")
            stats = storer.get_graph_statistics()
            
            print("\n📊 Node Counts:")
            for label, count in stats['node_counts'].items():
                print(f"  {label}: {count}")
            
            print("\n🔗 Relationship Counts:")
            for rel_type, count in stats['relationship_counts'].items():
                print(f"  {rel_type}: {count}")
            
            print("\n🌾 Agricultural Fact Types:")
            for fact_type, count in stats['fact_types'].items():
                print(f"  {fact_type}: {count}")
                
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        storer.close()
        logger.info("Neo4j connection closed.")