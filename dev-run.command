#!/bin/bash
# AutoDoc v2 Development Server Launcher
# This script cleans cache and starts the development server

# Change to script directory (for double-click execution)
cd "$(dirname "$0")"

echo "AutoDoc v2 Development Server"
echo "=============================="

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: This script must be run from the project root directory"
    echo "Make sure you're in the autodoc-v2 folder"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not detected"
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Run the bash script
./scripts/dev-run.sh "$@"
