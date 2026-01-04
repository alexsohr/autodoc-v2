#!/bin/bash
# Bash script for Mac/Unix development environment
# Equivalent to 'make dev-run' - cleans cache and starts the development server

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Flags
CLEAN_ONLY=false
SKIP_CLEAN=false

show_help() {
    echo -e "${GREEN}AutoDoc v2 Development Server Script${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./scripts/dev-run.sh              # Clean cache and start server"
    echo "  ./scripts/dev-run.sh -c           # Only clean cache"
    echo "  ./scripts/dev-run.sh --clean-only # Only clean cache"
    echo "  ./scripts/dev-run.sh -s           # Start server without cleaning"
    echo "  ./scripts/dev-run.sh --skip-clean # Start server without cleaning"
    echo "  ./scripts/dev-run.sh -h           # Show this help"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  -c, --clean-only    Clean cache files only, don't start server"
    echo "  -s, --skip-clean    Start server without cleaning cache first"
    echo "  -h, --help          Show this help message"
    echo ""
}

test_python_environment() {
    # Check if we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${YELLOW}Warning: Virtual environment not detected. Make sure to activate your venv first:${NC}"
        echo -e "  ${CYAN}source venv/bin/activate${NC}"
        echo ""
    fi

    # Check if Python is available
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        echo -e "${GREEN}Using Python: $PYTHON_VERSION${NC}"
    else
        echo -e "${RED}Error: Python not found. Please ensure Python is installed and in your PATH.${NC}"
        exit 1
    fi
}

invoke_cache_clean() {
    echo -e "${YELLOW}Cleaning cache files...${NC}"
    if python scripts/clean_cache.py; then
        echo -e "${GREEN}Cache cleaning completed successfully!${NC}"
    else
        EXIT_CODE=$?
        echo -e "${RED}Error: Cache cleaning failed with exit code $EXIT_CODE${NC}"
        exit $EXIT_CODE
    fi
}

start_dev_server() {
    echo -e "${GREEN}Starting AutoDoc v2 development server...${NC}"
    echo -e "Server will be available at: ${CYAN}http://localhost:8000${NC}"
    echo -e "API documentation at: ${CYAN}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""

    python -m src.api.main
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}Error: Failed to start development server (exit code $EXIT_CODE)${NC}"
        exit $EXIT_CODE
    fi
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean-only)
            CLEAN_ONLY=true
            shift
            ;;
        -s|--skip-clean)
            SKIP_CLEAN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Main execution

# Check Python environment
test_python_environment

# Clean cache unless skipped
if [ "$SKIP_CLEAN" = false ]; then
    invoke_cache_clean
fi

# Start server unless clean-only mode
if [ "$CLEAN_ONLY" = false ]; then
    echo ""
    start_dev_server
else
    echo -e "${GREEN}Cache cleaning completed. Use './scripts/dev-run.sh -s' to start the server.${NC}"
fi
