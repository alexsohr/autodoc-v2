# PowerShell script for Windows development environment
# Equivalent to 'make dev-run' - cleans cache and starts the development server

param(
    [switch]$Help,
    [switch]$CleanOnly,
    [switch]$SkipClean
)

function Show-Help {
    Write-Host "AutoDoc v2 Development Server Script" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\dev-run.ps1           # Clean cache and start server"
    Write-Host "  .\scripts\dev-run.ps1 -CleanOnly # Only clean cache"
    Write-Host "  .\scripts\dev-run.ps1 -SkipClean # Start server without cleaning"
    Write-Host "  .\scripts\dev-run.ps1 -Help     # Show this help"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -CleanOnly    Clean cache files only, don't start server"
    Write-Host "  -SkipClean    Start server without cleaning cache first"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
}

function Test-PythonEnvironment {
    # Check if we're in a virtual environment
    if (-not $env:VIRTUAL_ENV) {
        Write-Warning "Virtual environment not detected. Make sure to activate your venv first:"
        Write-Host "  venv\Scripts\activate" -ForegroundColor Cyan
        Write-Host ""
    }
    
    # Check if Python is available
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "Using Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Error "Python not found. Please ensure Python is installed and in your PATH."
        exit 1
    }
}

function Invoke-CacheClean {
    Write-Host "Cleaning cache files..." -ForegroundColor Yellow
    try {
        python scripts/clean_cache.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Cache cleaning completed successfully!" -ForegroundColor Green
        } else {
            Write-Error "Cache cleaning failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    catch {
        Write-Error "Failed to run cache cleaning script: $_"
        exit 1
    }
}

function Start-DevServer {
    Write-Host "Starting AutoDoc v2 development server..." -ForegroundColor Green
    Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "API documentation at: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        python -m src.api.main
    }
    catch {
        Write-Error "Failed to start development server: $_"
        exit 1
    }
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

# Check Python environment
Test-PythonEnvironment

# Clean cache unless skipped
if (-not $SkipClean) {
    Invoke-CacheClean
}

# Start server unless clean-only mode
if (-not $CleanOnly) {
    Write-Host ""
    Start-DevServer
} else {
    Write-Host "Cache cleaning completed. Use '.\scripts\dev-run.ps1 -SkipClean' to start the server." -ForegroundColor Green
}
