A (maybe) growing tool collection 

1. Project_mapper

Project Mapper for LLM Context
================================
Generate a comprehensive mapping of a Python project for AI coding agents.

Features:

- Full AST analysis (classes, functions, imports)
- Cross-file dependency graph
- SQLite schema extraction
- LLM-optimized Markdown output
- Multi-project support

Usage:
    python project_mapper.py [--output PROJECT_MAP.md] [--db path/to/db.sqlite]
    
    
2. Check YAML Keys

YAML keys utility
===================

Script to detect unused YAML keys in fr.yaml.

Features:

- Parses the fr.yaml file
- Scans all Python files in the project
- Detects all calls to translate()
- Compares and lists unused keys
- Optional: Generates a cleaned fr.yaml

Usage:
    python check_yaml_keys.py
    python check_yaml_keys.py --clean  # Generate fr_clean.yaml