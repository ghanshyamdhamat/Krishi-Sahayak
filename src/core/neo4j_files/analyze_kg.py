from neo4j import GraphDatabase
import json
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphAnalyzer:
    """Comprehensive analysis tool for the agricultural knowledge graph."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: Dict = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        stats = {}
        
        # Basic node and relationship counts
        basic_stats = self.run_query("""
        MATCH (n)
        RETURN 
            count(n) as total_nodes,
            count(DISTINCT labels(n)) as unique_node_types
        """)[0]
        
        rel_stats = self.run_query("""
        MATCH ()-[r]->()
        RETURN 
            count(r) as total_relationships,
            count(DISTINCT type(r)) as unique_relationship_types
        """)[0]
        
        stats.update(basic_stats)
        stats.update(rel_stats)
        
        # Node type distribution
        node_distribution = self.run_query("""
        MATCH (n)
        UNWIND labels(n) as label
        RETURN label, count(*) as count
        ORDER BY count DESC
        """)
        stats['node_distribution'] = {item['label']: item['count'] for item in node_distribution}
        
        # Relationship type distribution  
        rel_distribution = self.run_query("""
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """)
        stats['relationship_distribution'] = {item['relationship_type']: item['count'] for item in rel_distribution}
        
        return stats

    def analyze_agricultural_content(self) -> Dict[str, Any]:
        """Analyze the agricultural content structure."""
        content_analysis = {}
        
        # Fact type distribution
        fact_types = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.fact_type IS NOT NULL
        RETURN f.fact_type as fact_type, count(f) as count
        ORDER BY count DESC
        """)
        content_analysis['fact_types'] = {item['fact_type']: item['count'] for item in fact_types}
        
        # Subject analysis (what crops/topics are most covered)
        subjects = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.subject IS NOT NULL
        RETURN f.subject as subject, count(f) as count
        ORDER BY count DESC
        LIMIT 20
        """)
        content_analysis['top_subjects'] = {item['subject']: item['count'] for item in subjects}
        
        # Source file coverage
        sources = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.source_file IS NOT NULL
        RETURN f.source_file as source, count(f) as fact_count
        ORDER BY fact_count DESC
        """)
        content_analysis['source_coverage'] = {item['source']: item['fact_count'] for item in sources}
        
        # Content type distribution
        content_types = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.content_type IS NOT NULL
        RETURN f.content_type as content_type, count(f) as count
        ORDER BY count DESC
        """)
        content_analysis['content_types'] = {item['content_type']: item['count'] for item in content_types}
        
        return content_analysis

    def analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze the connectivity patterns in the knowledge graph."""
        connectivity = {}
        
        # Most connected entities
        most_connected = self.run_query("""
        MATCH (e:Entity)-[r]-()
        RETURN e.name as entity, count(r) as connection_count
        ORDER BY connection_count DESC
        LIMIT 20
        """)
        connectivity['most_connected_entities'] = [(item['entity'], item['connection_count']) for item in most_connected]
        
        # Isolated entities (no relationships)
        isolated = self.run_query("""
        MATCH (e:Entity)
        WHERE NOT (e)-[]-()
        RETURN count(e) as isolated_count
        """)[0]
        connectivity.update(isolated)
        
        # Average connectivity
        avg_connectivity = self.run_query("""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as connections
        RETURN avg(connections) as avg_connections, max(connections) as max_connections
        """)[0]
        connectivity.update(avg_connectivity)
        
        # Relationship pattern analysis
        patterns = self.run_query("""
        MATCH (a)-[r1]->(b)-[r2]->(c)
        RETURN type(r1) + " -> " + type(r2) as pattern, count(*) as frequency
        ORDER BY frequency DESC
        LIMIT 15
        """)
        connectivity['common_patterns'] = {item['pattern']: item['frequency'] for item in patterns}
        
        return connectivity

    def identify_knowledge_gaps(self) -> Dict[str, Any]:
        """Identify potential gaps and data quality issues."""
        gaps = {}
        
        # Chunks without extractions
        empty_chunks = self.run_query("""
        MATCH (c:Chunk)
        WHERE NOT (c)-[:CONTAINS_EXTRACTION]->()
        RETURN count(c) as empty_chunks_count
        """)[0]
        gaps.update(empty_chunks)
        
        # Facts without relationships
        isolated_facts = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE NOT (f)-[:DESCRIBES]->()
        RETURN count(f) as isolated_facts_count
        """)[0]
        gaps.update(isolated_facts)
        
        # Missing fact types
        missing_fact_types = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.fact_type IS NULL OR f.fact_type = ""
        RETURN count(f) as missing_fact_types_count
        """)[0]
        gaps.update(missing_fact_types)
        
        # Poor quality details - COMPLETELY SAFE VERSION
        poor_details = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.details IS NULL OR f.details = ""
        RETURN count(f) as poor_details_count
        """)[0]
        gaps.update(poor_details)
        
        # Separate query to count short details (only for strings)
        short_details = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.details IS NOT NULL 
        AND f.details <> "" 
        AND apoc.meta.type(f.details) = "STRING"
        AND length(f.details) < 20
        RETURN count(f) as short_details_count
        """)[0]
        gaps['short_details_count'] = short_details.get('short_details_count', 0)
        
        return gaps
    
    def get_domain_specific_insights(self) -> Dict[str, Any]:
        """Get agricultural domain-specific insights."""
        insights = {}
        
        # Crop coverage analysis
        crop_mentions = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.details CONTAINS 'rice' OR f.details CONTAINS 'paddy' 
           OR f.details CONTAINS 'wheat' OR f.details CONTAINS 'cotton'
        WITH 
            CASE 
                WHEN f.details CONTAINS 'rice' OR f.details CONTAINS 'paddy' THEN 'Rice/Paddy'
                WHEN f.details CONTAINS 'wheat' THEN 'Wheat'
                WHEN f.details CONTAINS 'cotton' THEN 'Cotton'
                ELSE 'Other'
            END as crop,
            f
        RETURN crop, count(f) as mentions
        ORDER BY mentions DESC
        """)
        insights['crop_coverage'] = {item['crop']: item['mentions'] for item in crop_mentions}
        
        # Zone/region analysis
        zone_mentions = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.details CONTAINS 'zone' OR f.details CONTAINS 'Tamil Nadu' 
           OR f.details CONTAINS 'district' OR f.details CONTAINS 'region'
        RETURN count(f) as geographic_references
        """)[0]
        insights.update(zone_mentions)
        
        # Technical vs practical content
        technical_content = self.run_query("""
        MATCH (f:AgronomicFact)
        WHERE f.fact_type IN ['SoilScience', 'PlantPhysiology', 'Genetics']
        RETURN count(f) as technical_facts
        """)[0]
        
        practical_content = self.run_query("""
        MATCH (f:AgronomicFact)  
        WHERE f.fact_type IN ['CultivationPractice', 'FarmManagement', 'PlantProtection']
        RETURN count(f) as practical_facts
        """)[0]
        
        insights.update(technical_content)
        insights.update(practical_content)
        
        return insights

    def export_analysis_report(self, filename: str = "knowledge_graph_analysis.json"):
        """Export comprehensive analysis to JSON file."""
        report = {
            "basic_statistics": self.get_comprehensive_statistics(),
            "agricultural_content": self.analyze_agricultural_content(),
            "connectivity_analysis": self.analyze_connectivity(), 
            "knowledge_gaps": self.identify_knowledge_gaps(),
            "domain_insights": self.get_domain_specific_insights()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report exported to {filename}")
        return report

    def create_visualizations(self, save_plots: bool = True):
        """Create visualization charts for the knowledge graph analysis."""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Agricultural Knowledge Graph Analysis', fontsize=16, fontweight='bold')
            
            # Get data for visualizations
            stats = self.get_comprehensive_statistics()
            content = self.analyze_agricultural_content()
            connectivity = self.analyze_connectivity()
            
            # 1. Node Type Distribution (Pie Chart)
            if stats.get('node_distribution'):
                node_data = stats['node_distribution']
                axes[0, 0].pie(node_data.values(), labels=node_data.keys(), autopct='%1.1f%%')
                axes[0, 0].set_title('Node Type Distribution')
            
            # 2. Fact Type Distribution (Bar Chart)
            if content.get('fact_types'):
                fact_data = content['fact_types']
                axes[0, 1].bar(range(len(fact_data)), list(fact_data.values()))
                axes[0, 1].set_xticks(range(len(fact_data)))
                axes[0, 1].set_xticklabels(list(fact_data.keys()), rotation=45, ha='right')
                axes[0, 1].set_title('Agricultural Fact Types')
                axes[0, 1].set_ylabel('Count')
            
            # 3. Source Coverage (Horizontal Bar Chart)
            if content.get('source_coverage'):
                source_data = content['source_coverage']
                # Take top 10 sources
                top_sources = dict(list(source_data.items())[:10])
                y_pos = range(len(top_sources))
                axes[1, 0].barh(y_pos, list(top_sources.values()))
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels([s.split('/')[-1][:20] + '...' if len(s) > 20 else s.split('/')[-1] 
                                           for s in top_sources.keys()], fontsize=8)
                axes[1, 0].set_title('Top Source Documents (by Fact Count)')
                axes[1, 0].set_xlabel('Number of Facts')
            
            # 4. Connectivity Distribution
            if connectivity.get('most_connected_entities'):
                conn_data = connectivity['most_connected_entities'][:10]
                entities = [item[0][:15] + '...' if len(item[0]) > 15 else item[0] for item in conn_data]
                connections = [item[1] for item in conn_data]
                
                axes[1, 1].bar(range(len(entities)), connections)
                axes[1, 1].set_xticks(range(len(entities)))
                axes[1, 1].set_xticklabels(entities, rotation=45, ha='right', fontsize=8)
                axes[1, 1].set_title('Most Connected Entities')
                axes[1, 1].set_ylabel('Connection Count')
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('knowledge_graph_analysis.png', dpi=300, bbox_inches='tight')
                logger.info("Visualization saved as 'knowledge_graph_analysis.png'")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

    def print_comprehensive_report(self):
        """Print a comprehensive analysis report to console."""
        print("\n" + "="*80)
        print("🌾 AGRICULTURAL KNOWLEDGE GRAPH COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Basic Statistics
        stats = self.get_comprehensive_statistics()
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total Nodes: {stats.get('total_nodes', 0):,}")
        print(f"   Total Relationships: {stats.get('total_relationships', 0):,}")
        print(f"   Unique Node Types: {stats.get('unique_node_types', 0)}")
        print(f"   Unique Relationship Types: {stats.get('unique_relationship_types', 0)}")
        
        # Node Distribution
        if stats.get('node_distribution'):
            print(f"\n📋 NODE TYPE DISTRIBUTION:")
            for node_type, count in list(stats['node_distribution'].items())[:10]:
                print(f"   {node_type}: {count:,}")
        
        # Agricultural Content Analysis
        content = self.analyze_agricultural_content()
        if content.get('fact_types'):
            print(f"\n🌾 AGRICULTURAL FACT TYPES:")
            for fact_type, count in list(content['fact_types'].items())[:10]:
                print(f"   {fact_type}: {count:,}")
        
        if content.get('top_subjects'):
            print(f"\n🎯 TOP AGRICULTURAL SUBJECTS:")
            for subject, count in list(content['top_subjects'].items())[:10]:
                print(f"   {subject}: {count:,}")
        
        # Connectivity Analysis
        connectivity = self.analyze_connectivity()
        print(f"\n🔗 CONNECTIVITY ANALYSIS:")
        print(f"   Isolated Entities: {connectivity.get('isolated_count', 0)}")
        print(f"   Average Connections per Entity: {connectivity.get('avg_connections', 0):.2f}")
        print(f"   Maximum Connections: {connectivity.get('max_connections', 0)}")
        
        if connectivity.get('most_connected_entities'):
            print(f"\n🌟 MOST CONNECTED ENTITIES:")
            for entity, conn_count in connectivity['most_connected_entities'][:5]:
                print(f"   {entity[:50]}{'...' if len(entity) > 50 else ''}: {conn_count} connections")
        
        # Knowledge Gaps
        gaps = self.identify_knowledge_gaps()
        print(f"\n⚠️  KNOWLEDGE GAPS & QUALITY ISSUES:")
        print(f"   Empty Chunks: {gaps.get('empty_chunks_count', 0)}")
        print(f"   Isolated Facts: {gaps.get('isolated_facts_count', 0)}")
        print(f"   Missing Fact Types: {gaps.get('missing_fact_types_count', 0)}")
        print(f"   Poor Quality Details: {gaps.get('poor_details_count', 0)}")
        
        # Domain Insights
        insights = self.get_domain_specific_insights()
        if insights.get('crop_coverage'):
            print(f"\n🌱 CROP COVERAGE ANALYSIS:")
            for crop, mentions in insights['crop_coverage'].items():
                print(f"   {crop}: {mentions:,} mentions")
        
        print(f"\n📍 GEOGRAPHIC REFERENCES: {insights.get('geographic_references', 0):,}")
        print(f"🧪 TECHNICAL FACTS: {insights.get('technical_facts', 0):,}")
        print(f"👨‍🌾 PRACTICAL FACTS: {insights.get('practical_facts', 0):,}")
        
        print("\n" + "="*80)
        print("Analysis complete! Use create_visualizations() for charts.")
        print("="*80)

def main():
    """Main function to run the analysis."""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "test1234"  # Replace with your password
    
    analyzer = KnowledgeGraphAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        print("Starting comprehensive knowledge graph analysis...")
        
        # Print comprehensive report
        analyzer.print_comprehensive_report()
        
        # Export detailed analysis
        analyzer.export_analysis_report("kg_analysis_report.json")
        
        # Create visualizations
        create_viz = input("\nCreate visualizations? (y/n): ").strip().lower()
        if create_viz == 'y':
            analyzer.create_visualizations(save_plots=True)
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    finally:
        analyzer.close()
        print("Analysis connection closed.")

if __name__ == "__main__":
    main()