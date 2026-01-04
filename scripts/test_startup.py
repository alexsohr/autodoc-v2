#!/usr/bin/env python3
"""
Test application startup without running the full server.
"""

import asyncio
import sys

# Add src to path for imports
sys.path.insert(0, "src")

# Import from the correct path
from src.services.data_access import init_mongodb


async def test_startup():
    """Test MongoDB initialization."""
    try:
        print("Testing MongoDB initialization...")
        await init_mongodb()
        print("SUCCESS: MongoDB initialization completed without errors!")
        return True
    except Exception as e:
        print(f"ERROR: MongoDB initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_startup())
    if success:
        print("\nApplication should now start successfully!")
        sys.exit(0)
    else:
        print("\nApplication startup will still fail.")
        sys.exit(1)