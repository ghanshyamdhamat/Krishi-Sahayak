from neo4j import GraphDatabase
import logging
from datetime import datetime
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoreFarmerProfile:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()
        
    def close(self):
        if self.driver:
            self.driver.close()
    
    def _verify_connection(self):
        """Verify Neo4j connection"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1").single()
                logger.info("✅ Connected to Neo4j for farmer profile storage")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def store_profile(self, farmer_id: str, profile_data: Dict):
        """Store or update farmer profile in Neo4j"""
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_or_update_profile, farmer_id, profile_data)
                logger.info(f"✅ Profile stored/updated for farmer {farmer_id}")
            except Exception as e:
                logger.error(f"Error storing profile for {farmer_id}: {e}")
                raise
    
    @staticmethod
    def _create_or_update_profile(tx, farmer_id: str, profile_data: Dict):
        """Create or update farmer profile with proper data handling"""
        
        # Handle different possible key names for compatibility
        name = profile_data.get('name', '')
        contact = profile_data.get('contact', '')
        village = profile_data.get('village', '')
        taluka = profile_data.get('taluka', '')
        district = profile_data.get('district', '')
        state = profile_data.get('state', '')
        country = profile_data.get('country', 'India')
        
        # Handle land size (could be 'land_size' or 'land_size_acres')
        land_size = profile_data.get('land_size', profile_data.get('land_size_acres', 0.0))
        if land_size is None:
            land_size = 0.0
        
        # Handle crops (ensure it's a list)
        crops = profile_data.get('crops', [])
        if isinstance(crops, str):
            crops = [c.strip() for c in crops.split(',') if c.strip()]
        
        preferred_language = profile_data.get('preferred_language', 'en')
        
        # Create/update main farmer node
        query = """
        MERGE (f:Farmer {id: $farmer_id})
        SET f.name = $name,
            f.contact = $contact,
            f.village = $village,
            f.taluka = $taluka,
            f.district = $district,
            f.state = $state,
            f.country = $country,
            f.land_size_acres = $land_size,
            f.preferred_language = $preferred_language,
            f.updated_at = datetime(),
            f.created_at = CASE WHEN f.created_at IS NULL THEN datetime() ELSE f.created_at END
        """
        
        tx.run(query, 
               farmer_id=farmer_id,
               name=name,
               contact=contact,
               district=district,
               state=state,
               village=village,
               taluka=taluka,
               country=country,
               land_size=float(land_size) if land_size else 0.0,
               preferred_language=preferred_language)
        
        # Handle crops separately with relationships
        if crops:
            # First, remove existing crop relationships
            tx.run("""
                MATCH (f:Farmer {id: $farmer_id})-[r:GROWS]->(c:Crop)
                DELETE r
            """, farmer_id=farmer_id)
            
            # Add new crop relationships
            for crop in crops:
                if crop.strip():
                    tx.run("""
                        MATCH (f:Farmer {id: $farmer_id})
                        MERGE (c:Crop {name: $crop_name})
                        MERGE (f)-[:GROWS]->(c)
                    """, farmer_id=farmer_id, crop_name=crop.strip())
    
    def get_profile(self, farmer_id: str) -> Optional[Dict]:
        """Retrieve farmer profile from Neo4j"""
        with self.driver.session() as session:
            try:
                result = session.execute_read(self._fetch_profile, farmer_id)
                if result:
                    logger.info(f"✅ Profile retrieved for farmer {farmer_id}")
                else:
                    logger.info(f"❌ No profile found for farmer {farmer_id}")
                return result
            except Exception as e:
                logger.error(f"Error retrieving profile for {farmer_id}: {e}")
                return None
    
    @staticmethod
    def _fetch_profile(tx, farmer_id: str) -> Optional[Dict]:
        """Fetch farmer profile with crops"""
        # Get main profile data
        query = """
        MATCH (f:Farmer {id: $farmer_id})
        RETURN f.id as id,
               f.name AS name, 
               f.contact AS contact,
               f.village AS village, 
               f.taluka AS taluka, 
               f.district AS district, 
               f.state AS state, 
               f.country AS country, 
               f.land_size_acres AS land_size_acres, 
               f.preferred_language AS preferred_language,
               f.created_at AS created_at,
               f.updated_at AS updated_at
        """
        
        record = tx.run(query, farmer_id=farmer_id).single()
        
        if record:
            profile_data = dict(record)
            
            # Get crops separately
            crops_query = """
            MATCH (f:Farmer {id: $farmer_id})-[:GROWS]->(c:Crop)
            RETURN collect(c.name) as crops
            """
            crops_result = tx.run(crops_query, farmer_id=farmer_id).single()
            
            if crops_result and crops_result['crops']:
                profile_data['crops'] = crops_result['crops']
            else:
                profile_data['crops'] = []
            
            return profile_data
        else:
            return None
    
    def get_all_farmers(self, limit: int = 100) -> List[Dict]:
        """Get all farmer profiles (for admin purposes)"""
        with self.driver.session() as session:
            try:
                result = session.execute_read(self._fetch_all_farmers, limit)
                logger.info(f"✅ Retrieved {len(result)} farmer profiles")
                return result
            except Exception as e:
                logger.error(f"Error retrieving all farmers: {e}")
                return []
    
    @staticmethod
    def _fetch_all_farmers(tx, limit: int) -> List[Dict]:
        """Fetch all farmers with basic info"""
        query = """
        MATCH (f:Farmer)
        OPTIONAL MATCH (f)-[:GROWS]->(c:Crop)
        RETURN f.id as id,
               f.name as name,
               f.state as state,
               f.district as district,
               f.village as village,
               f.contact as contact,
               f.preferred_language as preferred_language,
               f.land_size_acres as land_size_acres,
               collect(c.name) as crops,
               f.created_at as created_at,
               f.updated_at as updated_at
        ORDER BY f.updated_at DESC
        LIMIT $limit
        """
        
        farmers = []
        records = tx.run(query, limit=limit)
        
        for record in records:
            farmer_data = dict(record)
            # Clean up crops list (remove None values)
            farmer_data['crops'] = [crop for crop in farmer_data.get('crops', []) if crop]
            farmers.append(farmer_data)
        
        return farmers
    
    def delete_profile(self, farmer_id: str) -> bool:
        """Delete farmer profile and all related data"""
        with self.driver.session() as session:
            try:
                session.execute_write(self._delete_farmer_profile, farmer_id)
                logger.info(f"✅ Profile deleted for farmer {farmer_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting profile for {farmer_id}: {e}")
                return False
    
    @staticmethod
    def _delete_farmer_profile(tx, farmer_id: str):
        """Delete farmer and all relationships"""
        query = """
        MATCH (f:Farmer {id: $farmer_id})
        DETACH DELETE f
        """
        tx.run(query, farmer_id=farmer_id)
    
    def update_farmer_activity(self, farmer_id: str, activity_type: str, description: str):
        """Update farmer's last activity"""
        with self.driver.session() as session:
            try:
                session.execute_write(self._update_activity, farmer_id, activity_type, description)
                logger.info(f"✅ Activity updated for farmer {farmer_id}: {activity_type}")
            except Exception as e:
                logger.error(f"Error updating activity for {farmer_id}: {e}")
    
    @staticmethod
    def _update_activity(tx, farmer_id: str, activity_type: str, description: str):
        """Update farmer's last activity"""
        query = """
        MATCH (f:Farmer {id: $farmer_id})
        SET f.last_activity = datetime(),
            f.last_activity_type = $activity_type,
            f.last_activity_description = $description
        """
        tx.run(query, farmer_id=farmer_id, activity_type=activity_type, description=description)
    
    def get_farmer_stats(self, farmer_id: str) -> Dict:
        """Get farmer statistics"""
        with self.driver.session() as session:
            try:
                result = session.execute_read(self._fetch_farmer_stats, farmer_id)
                return result
            except Exception as e:
                logger.error(f"Error getting stats for {farmer_id}: {e}")
                return {}
    
    @staticmethod
    def _fetch_farmer_stats(tx, farmer_id: str) -> Dict:
        """Fetch farmer statistics"""
        query = """
        MATCH (f:Farmer {id: $farmer_id})
        OPTIONAL MATCH (f)-[:HAS_INTERACTION]->(i:Interaction)
        OPTIONAL MATCH (f)-[:GROWS]->(c:Crop)
        RETURN f.created_at as joined_date,
               f.last_activity as last_activity,
               f.last_activity_type as last_activity_type,
               count(DISTINCT i) as total_interactions,
               count(DISTINCT c) as total_crops
        """
        
        record = tx.run(query, farmer_id=farmer_id).single()
        
        if record:
            return {
                'joined_date': str(record['joined_date']) if record['joined_date'] else None,
                'last_activity': str(record['last_activity']) if record['last_activity'] else None,
                'last_activity_type': record['last_activity_type'],
                'total_interactions': record['total_interactions'] or 0,
                'total_crops': record['total_crops'] or 0
            }
        else:
            return {}
    
    def farmer_exists(self, farmer_id: str) -> bool:
        """Check if farmer exists in database"""
        with self.driver.session() as session:
            try:
                result = session.execute_read(self._check_farmer_exists, farmer_id)
                return result
            except Exception as e:
                logger.error(f"Error checking if farmer {farmer_id} exists: {e}")
                return False
    
    @staticmethod
    def _check_farmer_exists(tx, farmer_id: str) -> bool:
        """Check if farmer exists"""
        query = """
        MATCH (f:Farmer {id: $farmer_id})
        RETURN count(f) > 0 as exists
        """
        record = tx.run(query, farmer_id=farmer_id).single()
        return record['exists'] if record else False
    
    def search_farmers(self, search_term: str, limit: int = 20) -> List[Dict]:
        """Search farmers by name, state, or district"""
        with self.driver.session() as session:
            try:
                result = session.execute_read(self._search_farmers, search_term, limit)
                return result
            except Exception as e:
                logger.error(f"Error searching farmers: {e}")
                return []
    
    @staticmethod
    def _search_farmers(tx, search_term: str, limit: int) -> List[Dict]:
        """Search farmers"""
        query = """
        MATCH (f:Farmer)
        WHERE toLower(f.name) CONTAINS toLower($search_term)
           OR toLower(f.state) CONTAINS toLower($search_term)
           OR toLower(f.district) CONTAINS toLower($search_term)
           OR toLower(f.village) CONTAINS toLower($search_term)
        OPTIONAL MATCH (f)-[:GROWS]->(c:Crop)
        RETURN f.id as id,
               f.name as name,
               f.state as state,
               f.district as district,
               f.village as village,
               collect(c.name) as crops
        ORDER BY f.name
        LIMIT $limit
        """
        
        farmers = []
        records = tx.run(query, search_term=search_term, limit=limit)
        
        for record in records:
            farmer_data = dict(record)
            farmer_data['crops'] = [crop for crop in farmer_data.get('crops', []) if crop]
            farmers.append(farmer_data)
        
        return farmers

# Test and example usage
if __name__ == "__main__":
    # Example usage
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "test1234"  # Use your actual Neo4j password
    
    store = StoreFarmerProfile(uri, user, password)
    
    try:
        # Test profile creation
        farmer_id = "farmer123"
        profile_data = {
            'name': 'John Doe',
            'contact': '+91-9876543210',
            'district': 'Wardha',
            'state': 'Maharashtra',
            'country': 'India',
            'land_size_acres': 5.0,
            'crops': ['Wheat', 'Rice', 'Cotton'],
            'preferred_language': 'en',
        }
        
        print("🔄 Storing profile...")
        store.store_profile(farmer_id, profile_data)
        
        print("🔄 Retrieving profile...")
        retrieved_profile = store.get_profile(farmer_id)
        print(f"Retrieved profile: {retrieved_profile}")
        
        print("🔄 Getting farmer stats...")
        stats = store.get_farmer_stats(farmer_id)
        print(f"Farmer stats: {stats}")
        
        print("🔄 Checking if farmer exists...")
        exists = store.farmer_exists(farmer_id)
        print(f"Farmer exists: {exists}")
        
        print("🔄 Getting all farmers...")
        all_farmers = store.get_all_farmers(limit=5)
        print(f"Total farmers retrieved: {len(all_farmers)}")
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        store.close()