# AGENTS.md - LLM Prompts Subsystem

## Overview

YAML-based LLM prompts for wiki generation with anti-hallucination rules. This subsystem defines structured prompts that guide AI agents through repository exploration and documentation generation. All prompts enforce grounded, evidence-based output with strict citation requirements.

## Setup

```bash
pip install pyyaml
```

Prompts are loaded via PyYAML in agent code:
```python
import yaml
with open("src/prompts/wiki_prompts.yaml") as f:
    prompts = yaml.safe_load(f)
```

## Build/Tests

Prompts are validated through agent integration tests:
```bash
pytest tests/integration/test_wiki_agents.py
pytest tests/unit/test_wiki_workflow.py
```

No standalone prompt compilation step - prompts are loaded at runtime.

## Code Style

### YAML Structure
- Top-level `metadata` section with extraction info and purpose
- Agent prompts grouped by function (e.g., `structure_agent`, `page_generation_full`)
- Each prompt has `name`, `description`, `system_prompt`, and optional `user_prompt`

### Variable Substitution
Variables use curly brace format and are substituted at runtime:
- `{clone_path}` - Absolute path to cloned repository
- `{file_tree}` - Complete file listing with paths
- `{readme_content}` - README file contents
- `{page_title}`, `{page_description}` - Page metadata
- `{seed_paths_list}` - Starting file hints for exploration

### Two-Phase Exploration
1. **TRIAGE** - Parse README, analyze file tree, identify candidates (NO file reads)
2. **TARGETED READING** - Read only key files, max 2 reads per file

### Citation Format
```
Sources: [repo/path/file.ext:start-end]()
Sources: [src/main.py:10-25](), [src/config.py:5]()
```

## Security

- Never expose absolute paths in generated output
- Citations must use repo-relative paths only
- Clone paths are internal tooling references, not documentation content
- Validate that `{clone_path}` prefix is stripped from all citations

## PR Checklist

- [ ] Variable names match agent code expectations
- [ ] Anti-hallucination rules preserved in all prompts
- [ ] Citation format follows `[repo/path:lines]()` pattern
- [ ] Exit strategy conditions documented
- [ ] Two-phase exploration structure maintained
- [ ] File read limits enforced (max 2 per file)

## Examples

### Prompt Variable Usage
```yaml
system_prompt: |
  Repository root: {clone_path}
  Files available: {file_tree}
  README: {readme_content}
```

### Citation Format in Output
```markdown
The API routes are defined in the router module.
Sources: [src/api/routes/main.py:15-45]()

Configuration uses environment variables for secrets.
Sources: [src/config.py:10-20](), [.env.example:1-15]()
```

### Phase Structure
```yaml
# Phase 1: Triage (no reads)
- Parse README claims
- Identify file candidates from tree
- Determine repository type

# Phase 2: Targeted Reading
- Read first 150 lines of key files
- One optional follow-up read per file
- Stop when coverage complete or diminishing returns
```

## When Stuck

1. **Variable substitution failing**: Check that all `{variable}` names match exactly between prompt YAML and agent code
2. **Prompt not loading**: Verify YAML syntax, check for unescaped special characters
3. **LLM output malformed**: Inspect system prompt for clear output format instructions
4. **Citations using absolute paths**: Ensure prompts specify repo-relative path requirements
5. **Infinite exploration loops**: Verify exit strategy conditions are present in prompt

## House Rules

1. **Anti-hallucination required**: All prompts must include rules against inventing architecture or behavior
2. **Repo-relative paths only**: File paths in generated content are always relative to repository root
3. **Max 2 file reads**: Each file may be read at most twice (initial 150 lines + one targeted follow-up)
4. **Exit strategy mandatory**: Every exploration prompt must define clear exit conditions:
   - Coverage complete (identified repo type, entrypoints, module boundaries, runtime flow)
   - Diminishing returns (last 3 files revealed no new information)
5. **Unknown handling**: Unclear information goes to "open-questions" page or "next files to inspect"
6. **Citation minimum**: Page content must cite at least 5 distinct files
