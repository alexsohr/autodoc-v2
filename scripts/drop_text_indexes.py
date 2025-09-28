#!/usr/bin/env python3
"""
Drop text indexes that are causing conflicts.
"""

import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient

# Add src to path for imports
sys.path.insert(0, "src")

from utils.config_loader import get_settings


async def drop_text_indexes():
    """Drop text indexes from wiki_structures collection."""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_url)
    
    try:
        db = client[settings.mongodb_database]
        collection = db["wiki_structures"]
        
        # List current indexes
        indexes = await collection.list_indexes().to_list(None)
        print("Current wiki_structures indexes:")
        for idx in indexes:
            print(f"  - {idx['name']}: {idx.get('key', 'N/A')}")
        
        # Drop all text indexes
        text_indexes = []
        for idx in indexes:
            if 'text' in idx['name'].lower() and idx['name'] != '_id_':
                text_indexes.append(idx['name'])
        
        for index_name in text_indexes:
            try:
                await collection.drop_index(index_name)
                print(f"Dropped text index: {index_name}")
            except Exception as e:
                print(f"Error dropping {index_name}: {e}")
        
        print("Text index cleanup completed.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False
    finally:
        client.close()
    
    return True


if __name__ == "__main__":
    asyncio.run(drop_text_indexes())