#!/usr/bin/env python3
"""
Cache cleaning script for AutoDoc v2 development environment.
Removes __pycache__ folders, .egg-info directories, and other build artifacts.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List


def find_and_remove_directories(root_path: Path, dir_patterns: List[str]) -> int:
    """Find and remove directories matching the given patterns."""
    removed_count = 0
    
    for pattern in dir_patterns:
        for dir_path in root_path.rglob(pattern):
            if dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed: {dir_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"Warning: Could not remove {dir_path}: {e}")
    
    return removed_count


def find_and_remove_files(root_path: Path, file_patterns: List[str]) -> int:
    """Find and remove files matching the given patterns."""
    removed_count = 0
    
    for pattern in file_patterns:
        for file_path in root_path.rglob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    print(f"Removed: {file_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
    
    return removed_count


def clean_cache(project_root: Path = None) -> None:
    """Clean all cache and build artifacts from the project."""
    if project_root is None:
        # Get the project root (assuming script is in scripts/ directory)
        project_root = Path(__file__).parent.parent
    
    print(f"Cleaning cache files in: {project_root}")
    print("-" * 50)
    
    # Directory patterns to remove
    dir_patterns = [
        "__pycache__",
        "*.egg-info",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        "htmlcov",
    ]
    
    # File patterns to remove
    file_patterns = [
        "*.pyc",
        "*.pyo",
        ".coverage",
    ]
    
    # Remove directories
    dirs_removed = find_and_remove_directories(project_root, dir_patterns)
    
    # Remove files
    files_removed = find_and_remove_files(project_root, file_patterns)
    
    print("-" * 50)
    print(f"Cache cleaning completed!")
    print(f"Removed {dirs_removed} directories and {files_removed} files")
    
    # Also clean up any .pyc files that might be in specific locations
    # This is more thorough than the glob patterns above
    pyc_count = 0
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = Path(root) / file
                try:
                    file_path.unlink()
                    pyc_count += 1
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
    
    if pyc_count > 0:
        print(f"Additionally removed {pyc_count} .pyc/.pyo files")


def main():
    """Main entry point."""
    try:
        # Check if we're in the right directory
        project_root = Path.cwd()
        
        # Look for pyproject.toml to confirm we're in the project root
        if not (project_root / "pyproject.toml").exists():
            # Try to find the project root
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                print("Error: Could not find project root (no pyproject.toml found)")
                sys.exit(1)
        
        clean_cache(project_root)
        
    except KeyboardInterrupt:
        print("\nCache cleaning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during cache cleaning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
