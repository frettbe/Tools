#!/usr/bin/env python3
"""
Project Mapper for LLM Context
================================
Génère une cartographie complète d'un projet Python pour agents de codage AI.

Features:
- Analyse AST complète (classes, fonctions, imports)
- Graphe de dépendances inter-fichiers
- Extraction schéma SQLite
- Format markdown optimisé pour LLMs
- Support multi-projets

Usage:
    python project_mapper.py [--output PROJECT_MAP.md] [--db path/to/db.sqlite]
"""

import ast
import os
import sqlite3
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
import argparse
import json


@dataclass
class FileAnalysis:
    """Résultats d'analyse d'un fichier"""
    filepath: str
    imports: List[str] = field(default_factory=list)
    classes: List[Dict] = field(default_factory=list)
    functions: List[Dict] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    loc: int = 0
    internal_deps: Set[str] = field(default_factory=set)  # Dépendances vers autres fichiers du projet
    entry_points: List[str] = field(default_factory=list)  # Points d'entrée détectés


class CodeAnalyzer(ast.NodeVisitor):
    """Analyse statique AST d'un fichier Python"""

    def __init__(self, filepath: str, project_root: Path):
        self.filepath = filepath
        self.project_root = project_root
        self.imports = []
        self.classes = []
        self.functions = []
        self.dependencies = set()
        self.internal_deps = set()

    def visit_Import(self, node):
        """Capture les imports standards"""
        for alias in node.names:
            module = alias.name
            self.imports.append(module)
            self.dependencies.add(module.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Capture les imports from"""
        if node.module:
            self.imports.append(node.module)
            base_module = node.module.split('.')[0]
            self.dependencies.add(base_module)

            # Détecter si c'est un import interne au projet
            # (heuristique: pas dans stdlib ou packages connus)
            if not self._is_stdlib(base_module):
                potential_file = self._resolve_internal_import(node.module)
                if potential_file:
                    self.internal_deps.add(potential_file)

        self.generic_visit(node)

    def _is_stdlib(self, module: str) -> bool:
        """Check si module est stdlib Python"""
        # Utilisation d'un frozenset pour les performances + mise à jour des modules
        stdlib = frozenset({
            'os', 'sys', 'json', 'ast', 're', 'pathlib', 'collections',
            'typing', 'datetime', 'logging', 'argparse', 'configparser',
            'sqlite3', 'csv', 'io', 'math', 'random', 'time', 'unittest',
            'functools', 'itertools', 'operator', 'enum', 'dataclasses',
            'asyncio', 'threading', 'multiprocessing', 'subprocess',
            'http', 'urllib', 'email', 'html', 'xml', 'pickle', 'copy',
            'socket', 'ssl', 'hashlib', 'zlib', 'gzip', 'bz2', 'lzma',
            'shutil', 'tempfile', 'sysconfig', 'traceback', 'inspect'
        })
        return module in stdlib or module.split('.')[0] in stdlib

    def _resolve_internal_import(self, module: str) -> Optional[str]:
        """Tente de résoudre un import vers un fichier du projet"""
        # Convertir module.path en fichier potentiel
        parts = module.split('.')

        # Essayer plusieurs chemins possibles
        for i in range(len(parts), 0, -1):
            potential_path = Path('/'.join(parts[:i]) + '.py')
            if (self.project_root / potential_path).exists():
                return str(potential_path)

        return None

    def visit_If(self, node):
        """Détecte les points d'entrée de type if __name__ == '__main__'"""
        if isinstance(node.test, ast.Compare) and \
           isinstance(node.test.left, ast.Name) and \
           node.test.left.id == '__name__' and \
           len(node.test.ops) == 1 and \
           isinstance(node.test.ops[0], ast.Eq) and \
           isinstance(node.test.comparators[0], ast.Constant) and \
           node.test.comparators[0].value == '__main__':
            self.analysis.entry_points.append(f"__main__ block (line {node.lineno})")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Capture les classes et leurs méthodes"""
        # Détection des applications Flask
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'Flask':
                self.analysis.entry_points.append(f"Flask app: {node.name} (line {node.lineno})")
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(f"{b.value.id if hasattr(b.value, 'id') else '?'}.{b.attr}")

        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    'name': item.name,
                    'args': [{
                        'name': arg.arg,
                        'type': ast.unparse(arg.annotation) if arg.annotation else None
                    } for arg in item.args.args],
                    'return_type': ast.unparse(item.returns) if item.returns else None,
                    'is_async': False
                })
            elif isinstance(item, ast.AsyncFunctionDef):
                methods.append({
                    'name': item.name,
                    'args': [arg.arg for arg in item.args.args],
                    'is_async': True
                })

        self.classes.append({
            'name': node.name,
            'bases': bases,
            'methods': methods,
            'lineno': node.lineno,
            'docstring': ast.get_docstring(node)
        })
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Capture les fonctions (hors méthodes de classe)"""
        # Vérifier qu'on n'est pas dans une classe
        parent = getattr(node, 'parent', None)
        if not isinstance(parent, ast.ClassDef):
            self.functions.append({
                'name': node.name,
                'args': [{
                    'name': arg.arg,
                    'type': ast.unparse(arg.annotation) if arg.annotation else None
                } for arg in node.args.args],
                'return_type': ast.unparse(node.returns) if node.returns else None,
                'lineno': node.lineno,
                'is_async': False,
                'docstring': ast.get_docstring(node)
            })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Capture les fonctions async"""
        parent = getattr(node, 'parent', None)
        if not isinstance(parent, ast.ClassDef):
            self.functions.append({
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'lineno': node.lineno,
                'is_async': True,
                'docstring': ast.get_docstring(node)
            })
        self.generic_visit(node)


def analyze_file(filepath: Path, project_root: Path) -> Optional[FileAnalysis]:
    """Analyse un fichier Python"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))

        # Ajouter parent nodes pour context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        analyzer = CodeAnalyzer(str(filepath), project_root)
        analyzer.visit(tree)

        return FileAnalysis(
            filepath=str(filepath.relative_to(project_root)),
            imports=analyzer.imports,
            classes=analyzer.classes,
            functions=analyzer.functions,
            dependencies=analyzer.dependencies,
            internal_deps=analyzer.internal_deps,
            loc=len(content.split('\n'))
        )
    except Exception as e:
        print(f"⚠️  Error parsing {filepath}: {e}")
        return None


def analyze_project(root_dir: Path, ignore_dirs: Optional[Set[str]] = None) -> Dict:
    """Analyse complète du projet"""
    if ignore_dirs is None:
        ignore_dirs = {
            '__pycache__', '.git', 'venv', 'env', 'node_modules',
            '.venv', 'build', 'dist', '.pytest_cache', '.mypy_cache',
            'htmlcov', '.tox', '.eggs', '*.egg-info'
        }

    project_data = {
        'files': {},
        'stats': {
            'total_files': 0,
            'total_loc': 0,
            'total_classes': 0,
            'total_functions': 0,
            'entry_points': [],
            'test_files': 0,
            'test_coverage': None
        },
        'dependency_graph': defaultdict(set),
        'external_deps': defaultdict(int)
    }

    python_files = []

    # Collecter tous les fichiers Python
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]

        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)

    print(f"📊 Found {len(python_files)} Python files\n")

    # Analyser chaque fichier
    for filepath in python_files:
        rel_path = filepath.relative_to(root_dir)
        print(f"📄 Analyzing {rel_path}...")

        analysis = analyze_file(filepath, root_dir)

        if analysis:
            project_data['files'][analysis.filepath] = analysis
            project_data['stats']['total_files'] += 1
            project_data['stats']['total_loc'] += analysis.loc
            project_data['stats']['total_classes'] += len(analysis.classes)
            project_data['stats']['total_functions'] += len(analysis.functions)

            # Construire graphe de dépendances internes
            for dep in analysis.internal_deps:
                project_data['dependency_graph'][analysis.filepath].add(dep)

            # Compter dépendances externes
            for dep in analysis.dependencies:
                project_data['external_deps'][dep] += 1

    return project_data


def extract_sqlite_schema(db_path: Path) -> Dict:
    """Extrait le schéma d'une base SQLite"""
    schema = {
        'tables': {},
        'views': [],
        'indexes': []
    }

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Obtenir toutes les tables
        cursor.execute("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        for table_name, create_sql in cursor.fetchall():
            # Obtenir colonnes et types
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Obtenir foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()

            schema['tables'][table_name] = {
                'create_sql': create_sql,
                'columns': [
                    {
                        'name': col[1],
                        'type': col[2],
                        'notnull': bool(col[3]),
                        'default': col[4],
                        'pk': bool(col[5])
                    }
                    for col in columns
                ],
                'foreign_keys': [
                    {
                        'column': fk[3],
                        'references': f"{fk[2]}({fk[4]})"
                    }
                    for fk in foreign_keys
                ]
            }

        # Obtenir les vues
        cursor.execute("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='view'
            ORDER BY name
        """)
        schema['views'] = [{'name': name, 'sql': sql} for name, sql in cursor.fetchall()]

        # Obtenir les indexes
        cursor.execute("""
            SELECT name, tbl_name, sql FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            ORDER BY tbl_name, name
        """)
        schema['indexes'] = [
            {'name': name, 'table': tbl, 'sql': sql} 
            for name, tbl, sql in cursor.fetchall()
        ]

        conn.close()
        return schema

    except Exception as e:
        print(f"⚠️  Error extracting SQLite schema: {e}")
        return schema


def generate_markdown(project_data: Dict, db_schema: Optional[Dict], output_file: Path):
    """Génère le markdown optimisé pour LLMs"""

    stats = project_data['stats']

    md = f"""# 🗺️ Project Map - AI Context

> **Auto-generated** project cartography for LLM coding agents
> 
> **Stats:** {stats['total_files']} files | {stats['total_loc']:,} LOC | {stats['total_classes']} classes | {stats['total_functions']} functions

---

## 📋 Quick Overview

This map provides complete project context for AI pair programming. Use it to understand:
- Project structure and file organization
- Dependencies between modules
- Key classes and their responsibilities
- Database schema (if applicable)

---

## 🏗️ Project Structure

"""

    # Organiser par dossiers
    folders = defaultdict(list)
    for filepath in sorted(project_data['files'].keys()):
        folder = str(Path(filepath).parent) if Path(filepath).parent != Path('.') else '.'
        folders[folder].append(filepath)

    for folder in sorted(folders.keys()):
        files = folders[folder]
        md += f"\n### 📁 `{folder}/`\n\n"

        for filepath in sorted(files):
            data = project_data['files'][filepath]
            filename = Path(filepath).name

            # Résumé du fichier
            parts = []
            if data.classes:
                parts.append(f"{len(data.classes)} class{'es' if len(data.classes) > 1 else ''}")
            if data.functions:
                parts.append(f"{len(data.functions)} func{'s' if len(data.functions) > 1 else ''}")
            summary = ", ".join(parts) if parts else "utility module"

            md += f"#### `{filename}` ({data.loc} LOC)\n\n"
            md += f"**Purpose:** {summary}\n\n"

            # Classes
            if data.classes:
                md += "**Classes:**\n"
                for cls in data.classes:
                    bases_str = f" extends {', '.join(cls['bases'])}" if cls['bases'] else ""
                    md += f"- `{cls['name']}`{bases_str}\n"
                    if cls['methods']:
                        method_names = [m['name'] for m in cls['methods'][:5]]
                        md += f"  - Methods: `{'()`, `'.join(method_names)}()`"
                        if len(cls['methods']) > 5:
                            md += f" (+{len(cls['methods'])-5} more)"
                        md += "\n"
                md += "\n"

            # Functions
            if data.functions:
                md += "**Functions:**\n"
                for func in data.functions[:8]:
                    async_prefix = "async " if func['is_async'] else ""
                    args_str = ", ".join(func['args']) if func['args'] else ""
                    md += f"- `{async_prefix}{func['name']}({args_str})`\n"
                if len(data.functions) > 8:
                    md += f"- ... +{len(data.functions)-8} more functions\n"
                md += "\n"

            # Imports externes clés
            external_imports = [imp for imp in data.imports if imp.split('.')[0] in data.dependencies]
            if external_imports:
                key_imports = external_imports[:5]
                md += f"**Key imports:** `{'`, `'.join(key_imports)}`\n"
                if len(external_imports) > 5:
                    md += f" (+{len(external_imports)-5} more)\n"
                md += "\n"

            md += "---\n\n"

    # Graphe de dépendances
    if project_data['dependency_graph']:
        md += "## 🔗 Internal Dependencies Graph\n\n"
        md += "**File relationships** (who imports whom):\n\n"
        md += "```\n"
        for source, targets in sorted(project_data['dependency_graph'].items()):
            if targets:
                md += f"{source}\n"
                for target in sorted(targets):
                    md += f"  └─> {target}\n"
        md += "```\n\n"

    # Dépendances externes
    md += "## 📦 External Dependencies\n\n"

    stdlib_modules = {
        'os', 'sys', 'json', 'ast', 're', 'pathlib', 'collections',
        'typing', 'datetime', 'logging', 'argparse', 'sqlite3',
        'configparser', 'csv', 'io', 'math', 'random', 'time', 'unittest'
    }

    external = {dep: count for dep, count in project_data['external_deps'].items() 
                if dep not in stdlib_modules}

    if external:
        md += "**Third-party packages** (by usage):\n\n"
        for dep, count in sorted(external.items(), key=lambda x: x[1], reverse=True):
            md += f"- `{dep}` — used in {count} file{'s' if count > 1 else ''}\n"
    else:
        md += "*No external dependencies detected (stdlib only)*\n"

    md += "\n"

    # Schéma DB si fourni
    if db_schema and db_schema['tables']:
        md += "## 🗄️ Database Schema (SQLite)\n\n"

        for table_name, table_info in sorted(db_schema['tables'].items()):
            md += f"### Table: `{table_name}`\n\n"

            md += "| Column | Type | Constraints |\n"
            md += "|--------|------|-------------|\n"

            for col in table_info['columns']:
                constraints = []
                if col['pk']:
                    constraints.append("PRIMARY KEY")
                if col['notnull']:
                    constraints.append("NOT NULL")
                if col['default']:
                    constraints.append(f"DEFAULT {col['default']}")

                constraints_str = ", ".join(constraints) if constraints else "-"
                md += f"| `{col['name']}` | {col['type']} | {constraints_str} |\n"

            md += "\n"

            # Foreign keys
            if table_info['foreign_keys']:
                md += "**Foreign Keys:**\n"
                for fk in table_info['foreign_keys']:
                    md += f"- `{fk['column']}` → {fk['references']}\n"
                md += "\n"

        # Views
        if db_schema['views']:
            md += "### Views\n\n"
            for view in db_schema['views']:
                md += f"- `{view['name']}`\n"
            md += "\n"

    # Fichiers les plus importants (hub)
    md += "## 🎯 Key Files (ranked by complexity)\n\n"

    file_scores = []
    for filepath, data in project_data['files'].items():
        # Score basé sur classes, fonctions, LOC et dépendances
        score = (
            len(data.classes) * 5 +
            len(data.functions) * 2 +
            data.loc / 50 +
            len(data.internal_deps) * 3
        )
        file_scores.append((filepath, score, data))

    file_scores.sort(key=lambda x: x[1], reverse=True)

    for i, (filepath, score, data) in enumerate(file_scores[:15], 1):
        md += f"{i}. **`{filepath}`** (complexity: {score:.1f})\n"
        parts = []
        if data.classes:
            parts.append(f"{len(data.classes)} classes")
        if data.functions:
            parts.append(f"{len(data.functions)} functions")
        if data.internal_deps:
            parts.append(f"{len(data.internal_deps)} internal deps")
        md += f"   - {', '.join(parts)}\n"

    md += "\n---\n\n"

    # Guide pour LLMs
    md += """## 💡 Usage Guide for AI Coding Agents

### For Initial Context
When starting a new task, always:
1. Read this entire map to understand project structure
2. Identify relevant files from the structure section
3. Check dependencies before modifying files
4. Consult database schema when working with data models

### For Specific Tasks
- **Adding features**: Check "Key Files" section for entry points
- **Refactoring**: Review "Internal Dependencies Graph" to understand impact
- **Debugging**: Look at file's class/function list to locate code
- **Database work**: Refer to "Database Schema" section

### Best Practices
- Always maintain this map's structure when adding files
- Update dependencies when creating new modules
- Keep database schema in sync with migrations
- Use this map to avoid circular dependencies

---

*Generated by Project Mapper v1.0 - Optimized for LLM context*
"""

    # Sauvegarder
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n✅ {output_file} generated successfully!")

    return output_file


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="Generate project map for AI coding agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Analyze current directory
  %(prog)s --output docs/PROJECT_MAP.md # Custom output location
  %(prog)s --db data/app.db             # Include SQLite schema
  %(prog)s --project /path/to/project   # Analyze specific directory
        """
    )

    parser.add_argument(
        '--project',
        type=Path,
        default=Path('.'),
        help='Project root directory (default: current directory)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('PROJECT_MAP.md'),
        help='Output markdown file (default: PROJECT_MAP.md)'
    )

    parser.add_argument(
        '--db',
        type=Path,
        help='Path to SQLite database file (optional)'
    )

    parser.add_argument(
        '--json',
        type=Path,
        help='Also export raw analysis as JSON (optional)'
    )

    args = parser.parse_args()

    print("🗺️  Project Mapper - AI Context Generator\n")
    print(f"📂 Analyzing: {args.project.absolute()}\n")

    # Analyser le projet
    project_data = analyze_project(args.project)

    # Extraire schéma DB si fourni
    db_schema = None
    if args.db and args.db.exists():
        print(f"\n🗄️  Extracting SQLite schema from {args.db}...")
        db_schema = extract_sqlite_schema(args.db)
        print(f"✅ Found {len(db_schema['tables'])} tables")

    # Générer markdown
    generate_markdown(project_data, db_schema, args.output)

    # Export JSON optionnel
    if args.json:
        json_data = {
            'files': {
                path: {
                    'imports': data.imports,
                    'classes': data.classes,
                    'functions': data.functions,
                    'loc': data.loc,
                    'dependencies': list(data.dependencies),
                    'internal_deps': list(data.internal_deps)
                }
                for path, data in project_data['files'].items()
            },
            'stats': project_data['stats'],
            'dependency_graph': {k: list(v) for k, v in project_data['dependency_graph'].items()},
            'external_deps': dict(project_data['external_deps']),
            'db_schema': db_schema
        }

        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        print(f"✅ {args.json} exported")

    print(f"\n🎉 Done! Use '{args.output}' with your LLM coding agent.")


if __name__ == '__main__':
    main()
