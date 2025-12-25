# Document Agent Simplification Design

**Date:** 2025-12-25
**Status:** Implemented
**Branch:** 003-updating-mongodb-to-beanie

## Overview

The Document Agent is being simplified to produce only two outputs:
1. **Documentation files content** - README, CLAUDE.md, agent.md, etc.
2. **File tree structure** - ASCII tree representation of repository files

The agent will no longer read all file contents, generate embeddings, or store documents in MongoDB.

## Current vs New Architecture

### Current Flow
```
clone → discover_files → process_content (ALL files) → generate_embeddings → store_documents → cleanup
```

### New Flow
```
clone → load_patterns → discover_and_build_tree → extract_docs → END
```

## State Structure

```python
class DocumentProcessingState(TypedDict):
    """State for document processing workflow"""
    repository_id: str
    repository_url: str
    branch: Optional[str]
    clone_path: Optional[str]

    # New simplified outputs
    documentation_files: List[Dict[str, str]]  # [{path, content}, ...]
    file_tree: str  # ASCII tree structure

    # Patterns (loaded from config, possibly overridden by .autodoc/autodoc.json)
    excluded_dirs: List[str]
    excluded_files: List[str]

    # Workflow tracking
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
```

### Removed Fields
- `discovered_files` → replaced by `file_tree`
- `processed_documents` → replaced by `documentation_files`
- `embeddings_generated` → removed (no embeddings)

## Configuration

### Default Exclusions (in config_loader.py)

**Excluded Directories:**
```python
default_excluded_dirs: List[str] = [
    ".venv/", "venv/", "env/", "virtualenv/",
    "node_modules/", "bower_components/", "jspm_packages/",
    ".git/", ".svn/", ".hg/", ".bzr/",
    ".idea/", ".vscode/", ".vscode-server/", ".vscode-server-insiders/",
    ".pytest_cache/", ".pytest/", ".next/",
]
```

**Excluded Files:**
```python
default_excluded_files: List[str] = [
    # Lock files
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock",

    # OS files
    ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk",

    # Environment files
    ".env", ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv",

    # Git/CI files
    ".gitignore", ".gitattributes", ".gitmodules", ".github",
    ".gitlab-ci.yml",

    # Linter/formatter configs
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8",
    "mypy.ini", "pyproject.toml", "tsconfig.json",

    # Build configs
    "webpack.config.js", "babel.config.js", "rollup.config.js",
    "jest.config.js", "karma.conf.js", "vite.config.js", "next.config.js",

    # Minified/bundled files
    "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css", "*.map",

    # Archives
    "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z",
    "*.iso", "*.dmg", "*.img",

    # Installers/packages
    "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi",

    # Binaries
    "*.exe", "*.dll", "*.so", "*.dylib", "*.o", "*.obj",
    "*.jar", "*.war", "*.ear", "*.jsm", "*.class",

    # Python compiled
    "*.pyc", "*.pyd", "*.pyo", "__pycache__",
]
```

### Repository Override

If `.autodoc/autodoc.json` exists in the cloned repository, it **replaces** the defaults:

```json
{
  "excluded_dirs": ["tests/", "docs/"],
  "excluded_files": ["*.test.js", "*.spec.ts"]
}
```

## Documentation File Patterns

Hardcoded patterns for extracting documentation files (not configurable):

```python
DOC_FILE_PATTERNS = [
    # AI assistant instructions
    "CLAUDE.md",
    "claude.md",
    ".claude/CLAUDE.md",
    "agent.md",
    "AGENT.md",
    "llm.txt",
    "LLM.txt",
    "copilot-instructions.md",
    ".github/copilot-instructions.md",

    # Standard project docs
    "README.md",
    "README.txt",
    "README",
    "readme.md",
    "CONTRIBUTING.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "CODEOWNERS",
    ".github/CODEOWNERS",

    # Docs folder (recursive)
    "docs/**/*.md",
    "doc/**/*.md",
]
```

Doc files are always extracted regardless of exclusion patterns.

## Tree Output Format

ASCII tree format (like Unix `tree` command):

```
src/
├── api/
│   ├── main.py
│   └── routes/
│       └── health.py
├── services/
│   └── auth_service.py
└── models/
    └── user.py
```

## Workflow Nodes

### 1. clone_repository (existing)
- Clones repo to temp directory
- No changes needed

### 2. load_patterns (new)
- Load defaults from `config_loader.py`
- Check if `.autodoc/autodoc.json` exists in cloned repo
- If yes, replace patterns with values from that file
- Store in state: `excluded_dirs`, `excluded_files`

### 3. discover_and_build_tree (replaces discover_files + process_content)
- Walk directory tree starting from clone_path
- Apply exclusion filters (dirs and files)
- Build ASCII tree string
- Store in state: `file_tree`
- Does NOT read file contents

### 4. extract_docs (new)
- Match files against `DOC_FILE_PATTERNS`
- Read content of matched files only
- Store in state: `documentation_files` as `[{path, content}, ...]`

### 5. handle_error (existing)
- Error handling node
- No changes needed

## Removed Components

- `_generate_embeddings_node` - No embeddings
- `_store_documents_node` - No MongoDB storage
- `_cleanup_node` - Temp folder stays for Wiki Agent
- Embedding tool dependency

## Implementation Files

### Files to Modify

1. **`src/utils/config_loader.py`**
   - Add `default_excluded_dirs: List[str]`
   - Add `default_excluded_files: List[str]`

2. **`src/agents/document_agent.py`**
   - Update `DocumentProcessingState`
   - Add `DOC_FILE_PATTERNS` constant
   - Remove nodes: `_generate_embeddings_node`, `_store_documents_node`, `_cleanup_node`
   - Add node: `_load_patterns_node`
   - Replace: `_discover_files_node` → `_discover_and_build_tree_node`
   - Replace: `_process_content_node` → `_extract_docs_node`
   - Update `_create_workflow`

### Files to Review
- Orchestration layer / Wiki Agent that consumes Document Agent output

## Output Contract

The Document Agent returns a state containing:

```python
{
    "clone_path": "/tmp/repo-xyz",  # For Wiki Agent to access files
    "documentation_files": [
        {"path": "README.md", "content": "# Project..."},
        {"path": "CLAUDE.md", "content": "# Instructions..."},
    ],
    "file_tree": "src/\n├── api/\n│   └── main.py\n...",
}
```

The temp folder (`clone_path`) is NOT cleaned up - cleanup is the responsibility of the orchestration layer after Wiki Agent completes.
