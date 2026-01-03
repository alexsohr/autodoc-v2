#!/usr/bin/env python3
"""
Fix MongoDB index conflicts during Beanie migration.

This script resolves index naming conflicts that occur when migrating
from direct MongoDB operations to Beanie ODM.
"""

import asyncio
import sys
from typing import List, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import OperationFailure

# Add src to path for imports
sys.path.insert(0, "src")

from utils.config_loader import get_settings


async def list_collection_indexes(collection) -> List[Dict[str, Any]]:
    """List all indexes for a collection."""
    try:
        indexes = await collection.list_indexes().to_list(None)
        return indexes
    except Exception as e:
        print(f"Error listing indexes: {e}")
        return []


async def drop_conflicting_indexes(collection, conflicting_names: List[str]) -> None:
    """Drop indexes that conflict with Beanie naming conventions."""
    for index_name in conflicting_names:
        try:
            await collection.drop_index(index_name)
            print(f"✓ Dropped conflicting index: {index_name}")
        except OperationFailure as e:
            if "index not found" in str(e).lower():
                print(f"  Index {index_name} not found (already dropped)")
            else:
                print(f"✗ Error dropping index {index_name}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error dropping index {index_name}: {e}")


async def fix_collection_indexes(collection_name: str):
    """Fix index conflicts in a specific collection."""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_url)
    
    try:
        db = client[settings.mongodb_database]
        collection = db[collection_name]
        
        print(f"Checking {collection_name} collection indexes...")
        indexes = await list_collection_indexes(collection)
        
        print(f"
Current indexes in {collection_name}:")
        for idx in indexes:
            print(f"  - {idx['name']}: {idx.get('key', 'N/A')}")
        
        # Identify conflicting indexes
        conflicting_indexes = []
        for idx in indexes:
            name = idx['name']
            # Look for indexes that might conflict with Beanie's naming
            if name != '_id_':
                # For text indexes, look for multiple text indexes
                if 'text' in name.lower() and collection_name == 'wiki_structures':
                    conflicting_indexes.append(name)
                # For other potential conflicts
                elif (('url' in name or 'unique' in name) and 
                      not name.startswith('beanie_')):
                    conflicting_indexes.append(name)
        
        if conflicting_indexes:
            print(f"
Found {len(conflicting_indexes)} potentially conflicting indexes:")
            for name in conflicting_indexes:
                print(f"  - {name}")
            
            print(f"
Dropping conflicting indexes from {collection_name}...")
            await drop_conflicting_indexes(collection, conflicting_indexes)
        else:
            print(f"
No conflicting indexes found in {collection_name}.")
        
        print(f"
{collection_name} index cleanup completed.")
        
    except Exception as e:
        print(f"Error during {collection_name} index cleanup: {e}")
        return False
    finally:
        client.close()
    
    return True


async def main():
    """Main function to fix index conflicts."""
    print("AutoDoc v2 - Index Conflict Resolver")
    print("=" * 40)
    
    # Collections that might have index conflicts
    collections_to_check = [
        "repositories",
        "wiki_structures", 
        "code_documents",
        "chat_sessions",
        "questions",
        "answers"
    ]
    
    all_success = True
    for collection_name in collections_to_check:
        print(f"
{'='*50}")
        success = await fix_collection_indexes(collection_name)
        if not success:
            all_success = False
    
    print(f"
{'='*50}")
    if all_success:
        print("✓ All index conflicts resolved successfully!")
        print("You can now start the application with: make run")
    else:
        print("✗ Failed to resolve some index conflicts.")
        print("Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())