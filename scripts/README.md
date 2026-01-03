# AutoDoc v2 Development Scripts

This directory contains utility scripts for AutoDoc v2 development and maintenance.

## Cache Cleaning

### Overview
During development, Python creates cache files (`__pycache__` folders, `.egg-info` directories, etc.) that can sometimes cause issues with imports, testing, or deployment. The cache cleaning functionality automatically removes these files before starting the development server.

### Available Scripts

#### 1. Python Cache Cleaner (`clean_cache.py`)
Cross-platform Python script that removes all cache and build artifacts.

**Usage:**
```bash
python scripts/clean_cache.py
```

**What it removes:**
- `__pycache__` directories (recursively)
- `*.egg-info` directories
- `.mypy_cache` directories
- `.pytest_cache` directories
- `build/` and `dist/` directories
- `htmlcov/` directory
- `.coverage` files
- `*.pyc` and `*.pyo` files

#### 2. PowerShell Development Script (`dev-run.ps1`)
Windows-specific PowerShell script that combines cache cleaning with server startup.

**Usage:**
```powershell
# Clean cache and start server
.\scripts\dev-run.ps1

# Only clean cache
.\scripts\dev-run.ps1 -CleanOnly

# Start server without cleaning
.\scripts\dev-run.ps1 -SkipClean

# Show help
.\scripts\dev-run.ps1 -Help
```

#### 3. Batch File Launcher (`dev-run.bat`)
Simple Windows batch file for easy access.

**Usage:**
```cmd
# From project root
dev-run.bat
```

### Makefile Integration

If you have `make` available (Linux/macOS/WSL), you can use these targets:

```bash
# Clean cache using Python script
make clean-cache

# Clean cache and start development server
make dev-run

# Start server without cleaning (original behavior)
make run
```

### When to Use Cache Cleaning

**Recommended scenarios:**
- Starting a new development session
- After switching Git branches
- When experiencing import errors
- Before running tests
- After installing/updating dependencies
- When preparing for deployment

**Not necessary for:**
- Quick code changes during active development
- Running individual test files
- Production deployments (handled by build process)

### Environment Setup

Before using any development scripts:

1. **Activate Virtual Environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Verify Python Version:**
   ```bash
   python --version  # Should be 3.12+
   ```

3. **Install Dependencies (if needed):**
   ```bash
   pip install -e ".[dev]"
   ```

### Troubleshooting

**Script not found errors:**
- Ensure you're running from the project root directory
- Check that the scripts have proper permissions

**Permission errors:**
- On Windows: Run PowerShell as Administrator if needed
- On Linux/macOS: Check file permissions with `ls -la scripts/`

**Python not found:**
- Ensure Python is in your PATH
- Activate your virtual environment
- Verify Python installation with `python --version`

**Cache cleaning fails:**
- Some files might be in use by your IDE or other processes
- Close your IDE and try again
- Check if any Python processes are still running

### Integration with IDEs

**VS Code / Cursor:**
- Add tasks to `.vscode/tasks.json` for easy access
- Use the integrated terminal to run scripts

**PyCharm:**
- Add external tools configuration
- Use the built-in terminal

### Automation

For continuous integration or automated workflows, you can integrate cache cleaning:

```yaml
# Example GitHub Actions step
- name: Clean Python cache
  run: python scripts/clean_cache.py
```

### Contributing

When adding new cache cleaning functionality:
1. Update the Python script for cross-platform compatibility
2. Test on both Windows and Unix-like systems
3. Update this documentation
4. Add appropriate error handling
