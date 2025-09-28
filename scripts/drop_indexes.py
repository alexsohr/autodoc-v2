#!/usr/bin/env python3
"""
Drop specific conflicting indexes.
"""

import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient

# Add src to path for imports
sys.path.insert(0, "src")

from utils.config_loader import get_settings


async def drop_specific_indexes():
    """Drop the specific indexes that are causing conflicts."""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_url)
    
    try:
        db = client[settings.mongodb_database]
        
        # Drop conflicting indexes
        indexes_to_drop = [
            ("repositories", "url_1"),
            ("wiki_structures", "title_text_description_text"),
        ]
        
        for collection_name, index_name in indexes_to_drop:
            try:
                collection = db[collection_name]
                await collection.drop_index(index_name)
                print(f"Dropped {index_name} from {collection_name}")
            except Exception as e:
                print(f"Error dropping {index_name} from {collection_name}: {e}")
        
        print("Index cleanup completed.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False
    finally:
        client.close()
    
    return True


if __name__ == "__main__":
    asyncio.run(drop_specific_indexes())