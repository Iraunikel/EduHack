# Edalo Project Workflow

## Overview

This document defines the sustainable workflow for the Edalo project to ensure clear communication, controlled changes, and maintainable code.

## Change Management Process

### 1. Before Making Changes

- **Understand the current state**: Review existing code and documentation
- **Define the goal**: Clearly state what you want to achieve
- **Plan the changes**: Break down changes into small, incremental steps
- **Get approval**: For significant changes, discuss and get approval before implementing

### 2. Making Changes

- **Small, focused commits**: Each commit should address a single concern
- **Clear commit messages**: Use descriptive commit messages following the format:
  ```
  Short summary (50 chars or less)

  More detailed explanation if needed. Explain what and why, not how.
  ```
- **Follow project constraints**:
  - Functions max 25 lines
  - Files max 5 functions
  - Use while loops instead of for loops
  - No ternaries
  - Local execution only (no external APIs)

### 3. After Making Changes

- **Update CHANGELOG.md**: Document all changes with rationale
- **Test changes**: Verify that changes work as expected
- **Update documentation**: Keep README and code documentation up to date

## Code Review Checklist

Before committing changes, verify:

- [ ] Code follows project constraints (25-line functions, 5 functions per file, while loops, no ternaries)
- [ ] Code is properly documented with docstrings
- [ ] Changes are tested (if applicable)
- [ ] CHANGELOG.md is updated
- [ ] No temporary files or debug code is included
- [ ] Commit message is clear and descriptive

## Branch Strategy

### Main Branch
- `main`: Stable, working code
- Only merge tested, reviewed changes
- Keep in sync with origin/main

### Feature Branches (Optional)
- Use for experimental or large changes
- Create from `main`
- Merge back after testing and review
- Delete after merging

### Backup Branches
- Create backup branches before major resets or refactoring
- Name format: `backup-YYYYMMDD-HHMMSS` or `backup-description`

## Documentation Requirements

### Code Documentation
- All functions must have docstrings
- Document parameters and return values
- Explain complex logic

### Project Documentation
- Keep README.md up to date
- Document architecture changes in CHANGELOG.md
- Update workflow docs if process changes

## Commit Message Standards

### Format
```
Type: Short description (50 chars or less)

Optional longer description explaining what and why.
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

### Examples
```
fix: Handle missing magic library gracefully

Added try/except block to handle cases where python-magic
is not available, falling back to file extension detection.

docs: Update README with installation instructions

Added system dependency installation steps for python-magic.
```

## Approval Process

### Requiring Approval
- Changes to core architecture (pipeline, models, agents)
- Changes to project constraints or workflow
- Large refactorings (affecting multiple files)
- Changes to dependencies

### How to Request Approval
1. Create a clear description of proposed changes
2. Explain the rationale
3. Show examples or prototypes if applicable
4. Wait for explicit approval before implementing

### Quick Changes (No Approval Needed)
- Bug fixes
- Documentation updates
- Test additions
- Small refactorings within single files

## Testing Strategy

### Before Committing
- Run existing tests to ensure nothing breaks
- Test new functionality manually
- Verify constraints are followed (function length, file structure)

### Test Data
- Use test data in `tests/test_data/`
- Generate test data using `tests/generate_test_data.py`
- Keep test data representative of real use cases

## Handling Mistakes

### If Changes Don't Work
1. Don't panic - we have git history
2. Revert the commit if needed: `git revert <commit-hash>`
3. Or reset to previous state: `git reset --hard <commit-hash>`
4. Document what went wrong in CHANGELOG.md

### If You Lose Control
1. Create a backup branch: `git branch backup-before-fix`
2. Reset to last known good state
3. Review what went wrong
4. Implement fixes incrementally

## Communication

### When Working with AI Assistants
- Be explicit about what you want to change
- Review all changes before committing
- Ask for explanations if something is unclear
- Don't accept changes you don't understand

### When Working Alone
- Follow the same process
- Review your own changes
- Document your rationale
- Keep commits small and focused

## Project-Specific Guidelines

### Architecture
- 4-agent system: Detector, Optimizer, Embeddings, Evaluator
- Pipeline orchestrator coordinates agents
- Output formats: JSONL, SQLite, ChromaDB

### Constraints
- Functions: Max 25 lines
- Files: Max 5 functions
- Loops: Use while loops, not for loops
- Conditionals: No ternaries
- Execution: Local only, no external APIs

### File Structure
```
EduHack/
├── agents/           # Agent implementations
├── pipeline.py       # Main orchestrator
├── models.py         # Data models
├── output.py         # Output writers
├── cli.py            # CLI interface
├── config.py         # Configuration
├── utils.py          # Utilities
├── tests/            # Test suite
└── output/           # Generated outputs
```

## Regular Maintenance

### Weekly
- Review CHANGELOG.md
- Check for outdated documentation
- Verify tests still pass

### Before Major Releases
- Review all documentation
- Run full test suite
- Update README.md
- Create release notes

## Getting Help

### If Stuck
1. Review this workflow document
2. Check CHANGELOG.md for recent changes
3. Review git history: `git log --oneline`
4. Check planning documents in `planning (md)/`

### If Workflow Needs Improvement
1. Document the issue
2. Propose improvement
3. Update this document after approval
4. Update CHANGELOG.md

