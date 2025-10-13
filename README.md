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
    

3. project_mapper_qwenCoder_rev

Project Mapper for LLM Context 
================================

A revised Project_mapper by Qwen 3 Coder

Advanced project mapper that generates comprehensive codebase analysis for AI coding assistants.

Capabilities:
- Multi-language support: Python, JavaScript, TypeScript, Java, C++, and more
- Detailed AST analysis: functions, classes, methods, decorators, constants, exceptions
- Import and dependency tracking with classification (stdlib/third-party/internal)
- Multi-format output: Markdown, JSON, YAML, XML
- Smart caching system for performance optimization
- Incremental analysis to reduce processing time
- Configuration file support (YAML)
- Database schema extraction (SQLite)
- Memory-optimized processing for large files
- Progress tracking and detailed logging
- Relative import resolution
- Type annotation analysis
- Cross-file dependency graph construction

Features:
- Enhanced code analysis: detects decorators, constants, exception handling patterns
- Performance optimization: caching and incremental analysis
- Error resilience: comprehensive exception handling
- Output flexibility: multiple formats for different AI tools
- Language extensibility: framework for additional language support
- Configurability: customizable settings via config file
- Memory management: optimized for large projects

Usage:
    # Basic usage
    python project_mapper_qwenCoder_rev.py
    
    # Custom output and format
    python project_mapper_qwenCoder_rev.py --output docs/project_map.json --format json
    
    # With configuration file
    python project_mapper_qwenCoder_rev.py --config project_mapper.yml *
    
    # Include database schema
    python project_mapper_qwenCoder_rev.py --db data/app.db
    
    # Multi-language analysis with caching
    python project_mapper_qwenCoder_rev.py --project /path/to/project --cache-dir .cache --format json --verbose
 
 * see project_mapper.yml for example
