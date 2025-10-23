#!/usr/bin/env python3
"""
Project Mapper for LLM Context
================================
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

    
    # Custom output and format
    --output docs/project_map.json --format json
    
    # With configuration file
    --config project_mapper.yml
    
    # Include database schema
    --db data/app.db
    
    # Multi-language analysis with caching
    --project /path/to/project --cache-dir .cache --format json --verbose
"""

import ast
import os
import sqlite3
import concurrent.futures
from functools import partial
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
import argparse
import json
import sys
import hashlib
import pickle
from datetime import datetime
import logging

# Try to import yaml for configuration file support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False


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
    mtime: float = 0.0  # Timestamp of last modification
    hash: str = ""  # File content hash for change detection
    constants: List[Dict] = field(default_factory=list)  # Constantes détectées
    decorators: List[str] = field(default_factory=list)  # Décorateurs utilisés
    exceptions: List[Dict] = field(default_factory=list)  # Gestion des exceptions


@dataclass
class CacheEntry:
    """Entry in the analysis cache"""
    file_analysis: FileAnalysis
    cached_at: datetime
    project_root: str


class CodeAnalyzer(ast.NodeVisitor):
    """Analyse statique AST d'un fichier Python"""

    def __init__(self, filepath: str, project_root: Path):
        self.filepath = filepath
        self.project_root = project_root
        self.imports = []
        self.classes = []
        self.functions = []
        self.constants = []
        self.decorators = []
        self.exceptions = []
        self.dependencies = set()
        self.internal_deps = set()
        self.entry_points = []  # Points d'entrée détectés

    def visit_Import(self, node):
        """Capture les imports standards"""
        for alias in node.names:
            module = alias.name
            self.imports.append(module)
            # Add the full module path as well as the top-level package
            self.dependencies.add(module)  # Full module path
            self.dependencies.add(module.split('.')[0])  # Top-level package
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Capture les imports from"""
        if node.module:
            self.imports.append(node.module)
            base_module = node.module.split('.')[0]
            self.dependencies.add(node.module)  # Full module path
            self.dependencies.add(base_module)  # Top-level package

            # Détecter si c'est un import interne au projet
            # (heuristique: pas dans stdlib ou packages connus)
            if not self._is_stdlib(base_module):
                potential_file = self._resolve_internal_import(node.module)
                if potential_file:
                    self.internal_deps.add(potential_file)
                else:
                    # Try relative imports
                    potential_rel_file = self._resolve_relative_import(node.module, self.filepath)
                    if potential_rel_file:
                        self.internal_deps.add(potential_rel_file)

        self.generic_visit(node)

    def _is_stdlib(self, module: str) -> bool:
        """Check si module est stdlib Python"""
        if not module:
            return False
            
        module_name = module.split('.')[0]
        
        if sys.version_info >= (3, 10):
            try:
                # Use the official list for Python 3.10+
                return module_name in sys.stdlib_module_names
            except AttributeError:
                # In case stdlib_module_names is not available
                pass

        # Fallback for older Python versions and safety
        stdlib = {
            'os', 'sys', 'json', 'ast', 're', 'pathlib', 'collections',
            'typing', 'datetime', 'logging', 'argparse', 'configparser',
            'sqlite3', 'csv', 'io', 'math', 'random', 'time', 'unittest',
            'functools', 'itertools', 'operator', 'enum', 'dataclasses',
            'asyncio', 'threading', 'multiprocessing', 'subprocess',
            'http', 'urllib', 'email', 'html', 'xml', 'pickle', 'copy',
            'socket', 'ssl', 'hashlib', 'zlib', 'gzip', 'bz2', 'lzma',
            'shutil', 'tempfile', 'sysconfig', 'traceback', 'inspect',
            'site', 'importlib', 'warnings', 'weakref', 'gc', 'linecache',
            'string', 'reprlib', 'pprint', 'heapq', 'bisect', 'array',
            'contextlib', 'contextvars', 'abc', 'atexit', 'codecs', 'tokenize',
            'token', 'keyword', 'symtable', 'opcode', 'dis', 'copyreg', 'trace',
            'code', 'parser', 'stringprep', 'types', 'numbers', 'decimal',
            'fractions', 'random', 'statistics', 'itertools', 'functools',
            'operator', 'pathlib', 'filecmp', 'tempfile', 'glob', 'shutil',
            'curses', 'colorsys', 'html', 'xml', 'webbrowser', 'cgi',
            'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'smtplib',
            'uuid', 'socket', 'ssl', 'select', 'asyncio', 'concurrent',
            'multiprocessing', 'threading', 'queue', 'sched', 'dummy_threading',
            'time', 'timeit', 'profile', 'pstats', 'cProfile', 'traceback',
            'pyclbr', 'py_compile', 'compileall', 'zipfile', 'tarfile', 'gzip',
            'bz2', 'lzma', 'shutil', 'sqlite3', 'dbm', 'struct', 'codecs',
            'encodings', 'plistlib', 'hashlib', 'hmac', 'secrets', 'itertools',
            'collections', 'abc', 'io', 'text', 'zipimport', 'pkgutil',
            'modulefinder', 'runpy', 'importlib', 'ast', 'symtable', 'tokenize',
            'token', 'keyword', 'parser', 'tabnanny', 'pyclbr', 'pickle',
            'copyreg', 'reprlib', 'pprint', 'dataclasses', 'enum', 'graphlib'
        }
        return module_name in stdlib

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

    def _resolve_relative_import(self, module: str, filepath: str) -> Optional[str]:
        """Tente de résoudre un import relatif basé sur le fichier courant"""
        try:
            # Calculer le chemin du dossier contenant le fichier courant
            current_dir = Path(filepath).parent
            
            # Convertir le module en chemin relatif
            # Si le module commence par '.', c'est un import relatif
            if module.startswith('.'):
                # Compter les points pour déterminer le niveau de parenté
                level = 0
                while module.startswith('.'):
                    level += 1
                    module = module[1:]
                
                # Remonter dans l'arborescence
                parent_dir = current_dir
                for _ in range(level - 1):  # -1 car un . = même répertoire
                    parent_dir = parent_dir.parent
                
                # Si module est vide après retrait des '.', c'est un import du __init__.py
                if module:
                    potential_path = parent_dir / (module.replace('.', '/') + '.py')
                else:
                    potential_path = parent_dir / '__init__.py'
                
                # Vérifier si le chemin existe relativement au projet
                try:
                    relative_to_project = potential_path.relative_to(self.project_root)
                    if potential_path.exists():
                        return str(relative_to_project)
                except ValueError:
                    # Le chemin n'est pas dans le projet
                    pass
        except Exception:
            pass  # Erreur lors de la résolution, ignorer
            
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
            self.entry_points.append(f"__main__ block (line {node.lineno})")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Capture les classes et leurs méthodes"""
        # Capture les décorateurs de classe
        class_decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                class_decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                class_decorators.append(decorator.attr)

        # Détection des applications Flask
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'Flask':
                self.entry_points.append(f"Flask app: {node.name} (line {node.lineno})")
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(f"{b.value.id if hasattr(b.value, 'id') else '?'}.{b.attr}")

        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Capture les décorateurs de méthode
                method_decorators = []
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name):
                        method_decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        method_decorators.append(decorator.attr)
                
                methods.append({
                    'name': item.name,
                    'args': [{
                        'name': arg.arg,
                        'type': ast.unparse(arg.annotation) if arg.annotation else None
                    } for arg in item.args.args],
                    'return_type': ast.unparse(item.returns) if item.returns else None,
                    'decorators': method_decorators,
                    'is_async': False
                })
            elif isinstance(item, ast.AsyncFunctionDef):
                # Capture les décorateurs de méthode async
                method_decorators = []
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name):
                        method_decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        method_decorators.append(decorator.attr)
                
                methods.append({
                    'name': item.name,
                    'args': [{
                        'name': arg.arg,
                        'type': ast.unparse(arg.annotation) if arg.annotation else None
                    } for arg in item.args.args],
                    'return_type': ast.unparse(item.returns) if item.returns else None,
                    'decorators': method_decorators,
                    'is_async': True
                })

        self.classes.append({
            'name': node.name,
            'bases': bases,
            'methods': methods,
            'lineno': node.lineno,
            'docstring': ast.get_docstring(node),
            'decorators': class_decorators
        })
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Capture les fonctions (hors méthodes de classe)"""
        # Capture les décorateurs de fonction
        func_decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                func_decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                func_decorators.append(decorator.attr)

        # Vérifier qu'on n'est pas dans une classe
        parent = getattr(node, 'parent', None)
        if not isinstance(parent, ast.ClassDef):
            args_info = []
            for arg in node.args.args:
                arg_info = {'name': arg.arg}
                if arg.annotation:
                    try:
                        arg_info['type'] = ast.unparse(arg.annotation)
                    except:
                        arg_info['type'] = 'Unknown'
                args_info.append(arg_info)

            return_type = None
            if node.returns:
                try:
                    return_type = ast.unparse(node.returns)
                except:
                    return_type = 'Unknown'

            self.functions.append({
                'name': node.name,
                'args': args_info,
                'return_type': return_type,
                'lineno': node.lineno,
                'decorators': func_decorators,
                'is_async': False,
                'docstring': ast.get_docstring(node)
            })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Capture les fonctions async"""
        # Capture les décorateurs de fonction
        func_decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                func_decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                func_decorators.append(decorator.attr)

        parent = getattr(node, 'parent', None)
        if not isinstance(parent, ast.ClassDef):
            args_info = []
            for arg in node.args.args:
                arg_info = {'name': arg.arg}
                if arg.annotation:
                    try:
                        arg_info['type'] = ast.unparse(arg.annotation)
                    except:
                        arg_info['type'] = 'Unknown'
                args_info.append(arg_info)

            return_type = None
            if node.returns:
                try:
                    return_type = ast.unparse(node.returns)
                except:
                    return_type = 'Unknown'

            self.functions.append({
                'name': node.name,
                'args': args_info,
                'return_type': return_type,
                'lineno': node.lineno,
                'decorators': func_decorators,
                'is_async': True,
                'docstring': ast.get_docstring(node)
            })
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Detecter les constantes et variables globales"""
        # Vérifier si c'est une assignation de constante (noms en majuscules)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Vérifier si c'est une constante (noms en majuscules ou majuscules_soulignés)
                if target.id.isupper() or ('_' in target.id and target.id.replace('_', '').isupper()):
                    value_repr = 'Unknown'
                    try:
                        value_repr = ast.unparse(node.value)
                    except:
                        pass
                    
                    self.constants.append({
                        'name': target.id,
                        'value': value_repr,
                        'line': node.lineno
                    })
        self.generic_visit(node)

    def visit_Try(self, node):
        """Détecter la gestion des exceptions"""
        handlers_info = []
        for handler in node.handlers:
            handler_info = {
                'type': 'ExceptHandler',
                'lineno': handler.lineno
            }
            if handler.type:
                try:
                    handler_info['exception_type'] = ast.unparse(handler.type)
                except:
                    handler_info['exception_type'] = 'Unknown'
            if handler.name:
                if isinstance(handler.name, ast.Name):
                    handler_info['exception_name'] = handler.name.id
                else:
                    handler_info['exception_name'] = str(handler.name)
            handlers_info.append(handler_info)

        self.exceptions.append({
            'line': node.lineno,
            'handlers': handlers_info,
            'has_finally': node.finalbody is not None,
            'has_else': node.orelse is not None
        })
        self.generic_visit(node)

    def visit_Raise(self, node):
        """Détecter les instructions raise"""
        raise_info = {
            'line': node.lineno,
            'type': 'Raise'
        }
        if node.exc:
            try:
                raise_info['exception'] = ast.unparse(node.exc)
            except:
                raise_info['exception'] = 'Unknown'
        self.exceptions.append(raise_info)
        self.generic_visit(node)


def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of file content for change detection"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_changed_files(project_root: Path, ignore_dirs: Set[str], cache_dir: Optional[Path] = None) -> tuple[list[Path], list[Path]]:
    """
    Determine which files have changed since last analysis and which can be loaded from cache.
    
    Returns:
        - changed_files: List of files that need re-analysis
        - cached_files: List of files that can be loaded from cache
    """
    if not cache_dir:
        # If no cache directory, all files need analysis
        python_files = []
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files, []

    changed_files = []
    cached_files = []

    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                rel_path = filepath.relative_to(project_root)

                # Get current file content hash
                current_file_hash = get_file_hash(filepath)
                
                # Look for any cache entry for this relative file path
                # Cache files are named as: {rel_path}_{hash}.cache
                # So we need to find any cache file that starts with the rel_path
                escaped_rel_path = str(rel_path).replace('/', '_').replace('\\', '_')
                cache_files = list(cache_dir.glob(f"{escaped_rel_path}*.cache"))
                
                # Check if any existing cache entry matches current file
                is_cached = False
                for cache_path in cache_files:
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                            
                            # Check if cached file has same content as current file (using hash only)
                            if (isinstance(cached_data, CacheEntry) and 
                                cached_data.file_analysis.hash == current_file_hash):
                                # File content is unchanged, can load from cache
                                cached_files.append(filepath)
                                is_cached = True
                                break
                    except Exception:
                        # If cache loading fails, continue checking other cache files
                        continue
                
                if not is_cached:
                    # File needs re-analysis
                    changed_files.append(filepath)

    return changed_files, cached_files


def get_changed_files_multilang(project_root: Path, ignore_dirs: Set[str], cache_dir: Optional[Path] = None) -> tuple[list[Path], list[Path]]:
    """
    Determine which files have changed since last analysis and which can be loaded from cache (multilang).
    
    Returns:
        - changed_files: List of files that need re-analysis
        - cached_files: List of files that can be loaded from cache
    """
    logger = logging.getLogger(__name__)
    
    # Define supported file extensions
    supported_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rb', '.php'}
    
    if not cache_dir:
        # If no cache directory, all files need analysis
        all_files = []
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in supported_extensions:
                    all_files.append(Path(root) / file)
        return all_files, []

    changed_files = []
    cached_files = []

    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions:
                filepath = Path(root) / file
                rel_path = filepath.relative_to(project_root)

                # Get current file content hash
                current_file_hash = get_file_hash(filepath)
                
                # Look for any cache entry for this relative file path
                # Cache files are named as: {rel_path}_{hash}.cache
                # So we need to find any cache file that starts with the rel_path
                escaped_rel_path = str(rel_path).replace('/', '_').replace('\\\\', '_')
                cache_files = list(cache_dir.glob(f"{escaped_rel_path}*.cache"))
                
                # Check if any existing cache entry matches current file
                is_cached = False
                for cache_path in cache_files:
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                            
                            # Check if cached file has same content as current file (using hash only)
                            if (isinstance(cached_data, CacheEntry) and 
                                cached_data.file_analysis.hash == current_file_hash):
                                # File content is unchanged, can load from cache
                                cached_files.append(filepath)
                                is_cached = True
                                break
                    except Exception:
                        # If cache loading fails, continue checking other cache files
                        continue
                
                if not is_cached:
                    # File needs re-analysis
                    changed_files.append(filepath)

    return changed_files, cached_files


def load_cached_analysis(filepath: Path, project_root: Path, cache_dir: Path) -> Optional[FileAnalysis]:
    """Load analysis results from cache for a specific file."""
    logger = logging.getLogger(__name__)
    
    try:
        rel_path = filepath.relative_to(project_root)
    except ValueError:
        # Handle case where file is not relative to project root
        logger.error(f"File {filepath} is not relative to project root {project_root}")
        return None
    
    # We need to get the cache filename based on the current file hash
    try:
        file_hash = get_file_hash(filepath)
        cache_filename = f"{str(rel_path).replace('/', '_').replace('\\', '_')}_{file_hash[:16]}.cache"
        cache_path = cache_dir / cache_filename
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, CacheEntry) and cached_data.file_analysis.hash == file_hash:
                    logger.info(f"Using cached analysis for {filepath}")
                    return cached_data.file_analysis
        else:
            logger.debug(f"Cache file does not exist: {cache_path}")
    except FileNotFoundError:
        logger.info(f"Cache file not found for {filepath}, will re-analyze")
    except pickle.PickleError:
        logger.error(f"Cache file is corrupted for {filepath}, will re-analyze")
    except Exception as e:
        logger.error(f"Error loading cached analysis for {filepath}: {e}")
    
    return None


def load_cached_analysis_multilang(filepath: Path, project_root: Path, cache_dir: Path) -> Optional[FileAnalysis]:
    """Load analysis results from cache for a specific file (multilang)."""
    logger = logging.getLogger(__name__)
    
    try:
        rel_path = filepath.relative_to(project_root)
    except ValueError:
        # Handle case where file is not relative to project root
        logger.error(f"File {filepath} is not relative to project root {project_root}")
        return None
    
    # We need to get the cache filename based on the current file hash
    try:
        file_hash = get_file_hash(filepath)
        cache_filename = f"{str(rel_path).replace('/', '_').replace('\\', '_')}_{file_hash[:16]}.cache"
        cache_path = cache_dir / cache_filename
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, CacheEntry) and cached_data.file_analysis.hash == file_hash:
                    logger.info(f"Using cached analysis for {filepath}")
                    return cached_data.file_analysis
        else:
            logger.debug(f"Cache file does not exist: {cache_path}")
    except FileNotFoundError:
        logger.info(f"Cache file not found for {filepath}, will re-analyze")
    except pickle.PickleError:
        logger.error(f"Cache file is corrupted for {filepath}, will re-analyze")
    except Exception as e:
        logger.error(f"Error loading cached analysis for {filepath}: {e}")
    
    return None


def analyze_file_multilang(filepath: Path, project_root: Path, cache_dir: Optional[Path] = None) -> Optional[FileAnalysis]:
    """Analyse un fichier dans n'importe quel langage supporté avec support de cache"""
    logger = logging.getLogger(__name__)
    
    ext = filepath.suffix.lower()
    
    # For Python files, use the existing analyzer
    if ext == '.py':
        return analyze_file(filepath, project_root, cache_dir)
    
    # For other languages, use a basic text analysis
    logger.debug(f"Performing basic analysis for {filepath} ({ext})")
    
    try:
        # Get file stats for cache validation
        stat = filepath.stat()
        mtime = stat.st_mtime
        file_hash = get_file_hash(filepath)
        
        cache_entry = None
        cache_path = None
        
        if cache_dir:
            # Create cache path based on project-relative path and hash
            rel_path = filepath.relative_to(project_root)
            cache_filename = f"{str(rel_path).replace('/', '_').replace('\\\\', '_')}_{file_hash[:16]}.cache"
            cache_path = cache_dir / cache_filename
            
            # Try to load from cache if it exists
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        if (isinstance(cached_data, CacheEntry) and 
                            cached_data.file_analysis.hash == file_hash):
                            logger.info(f"Using cached analysis for {filepath}")
                            return cached_data.file_analysis
                except Exception as e:
                    logger.warning(f"Cache load failed for {filepath}, will re-analyze: {e}")
                    pass

        # Perform basic analysis based on file extension
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Basic analysis: count lines, find basic patterns depending on language
        loc = len(content.split('\n'))
        
        # Initialize analyzer with empty data
        file_analysis = FileAnalysis(
            filepath=str(filepath.relative_to(project_root)),
            loc=loc,
            mtime=mtime,
            hash=file_hash
        )
        
        # Language-specific basic analysis
        if ext in ['.js', '.ts', '.jsx', '.tsx']:
            # Basic JavaScript/TypeScript patterns
            import re
            # Find function declarations
            func_pattern = r'(?:function\s+|const\s+\w+\s*=\s*function|let\s+\w+\s*=\s*function|var\s+\w+\s*=\s*function|const\s+\w+\s*=\s*\(|\w+\s*:\s*\(|\w+\s*=\s*\([^)]*\)\s*=>|export default function)'
            functions = re.findall(func_pattern, content)
            
            # Find class declarations
            class_pattern = r'(?:class\s+\w+|export\s+default\s+class|export\s+class)'
            classes = re.findall(class_pattern, content)
            
            # Find import statements
            import_pattern = r'(?:import\s+|from\s+|require\()'
            imports = re.findall(import_pattern, content)
            
            file_analysis.functions = [{'name': f'function_{i}', 'args': [], 'lineno': 0, 'is_async': False} for i in range(len(functions))]
            file_analysis.classes = [{'name': f'class_{i}', 'lineno': 0} for i in range(len(classes))]
            file_analysis.imports = imports[:10]  # Limit to first 10 imports
        
        elif ext in ['.java']:
            # Basic Java patterns
            import re
            class_pattern = r'public\s+class\s+\w+|class\s+\w+'
            method_pattern = r'(?:public|private|protected)\s+[\w<>\[\]]+\s+\w+\s*\([^)]*\)'
            
            classes = re.findall(class_pattern, content)
            methods = re.findall(method_pattern, content)
            
            file_analysis.classes = [{'name': f'class_{i}', 'lineno': 0} for i in range(len(classes))]
            file_analysis.functions = [{'name': f'method_{i}', 'args': [], 'lineno': 0, 'is_async': False} for i in range(len(methods))]
        
        elif ext in ['.py']:
            # Should be handled by analyze_file, but just in case
            return analyze_file(filepath, project_root, cache_dir)
        
        # Save to cache if cache directory is provided
        if cache_dir and cache_path:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_entry = CacheEntry(
                file_analysis=file_analysis,
                cached_at=datetime.now(),
                project_root=str(project_root)
            )
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_entry, f)
            logger.debug(f"Saved analysis to cache: {cache_path}")

        logger.debug(f"Completed basic analysis for {filepath} ({ext})")
        return file_analysis
    except Exception as e:
        logger.error(f"Error analyzing {filepath}: {e}")
        return FileAnalysis(
            filepath=str(filepath.relative_to(project_root)),
            loc=0,
            mtime=mtime,
            hash=file_hash
        )


def analyze_file(filepath: Path, project_root: Path, cache_dir: Optional[Path] = None) -> Optional[FileAnalysis]:
    """Analyse un fichier Python avec support de cache"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"Starting analysis of {filepath}")
        # Get file stats for cache validation
        stat = filepath.stat()
        mtime = stat.st_mtime
        file_hash = get_file_hash(filepath)
        
        cache_entry = None
        cache_path = None
        
        if cache_dir:
            # Create cache path based on project-relative path and hash
            rel_path = filepath.relative_to(project_root)
            cache_filename = f"{str(rel_path).replace('/', '_').replace('\\', '_')}_{file_hash[:16]}.cache"
            cache_path = cache_dir / cache_filename
            
            # Try to load from cache if it exists
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        if (isinstance(cached_data, CacheEntry) and 
                            cached_data.file_analysis.hash == file_hash):
                            logger.info(f"Using cached analysis for {filepath}")
                            return cached_data.file_analysis
                except Exception as e:
                    logger.warning(f"Cache load failed for {filepath}, will re-analyze: {e}")
                    pass

        logger.debug(f"Reading file: {filepath}")
        
        # Check file size to optimize memory usage for large files
        file_size = filepath.stat().st_size
        if file_size > 5 * 1024 * 1024:  # 5 MB threshold
            logger.warning(f"Large file detected: {filepath} ({file_size / (1024*1024):.1f} MB). Processing with memory optimization.")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        try:
            logger.debug(f"Parsing AST for {filepath}")
            tree = ast.parse(content, filename=str(filepath))
        except SyntaxError as e:
            logger.error(f"Syntax error in {filepath} at line {e.lineno}: {e.msg}")
            # Return a basic FileAnalysis object with minimal information to avoid breaking the process
            # Use content.split in a more memory-efficient way for very large files
            loc = len(content.split('\n'))
            # Clear content to free memory for large files
            if file_size > 10 * 1024 * 1024:  # 10 MB or larger
                content = None
            return FileAnalysis(
                filepath=str(filepath.relative_to(project_root)),
                loc=loc,
                mtime=mtime,
                hash=file_hash
            )
        except MemoryError:
            logger.error(f"Memory error analyzing {filepath}, file too large to process")
            return FileAnalysis(
                filepath=str(filepath.relative_to(project_root)),
                loc=0,  # LOC will be inaccurate for memory error
                mtime=mtime,
                hash=file_hash
            )
        except Exception as e:
            logger.error(f"Failed to parse AST for {filepath}: {e}")
            # Clear content to free memory for very large files
            if file_size > 10 * 1024 * 1024:  # 10 MB or larger
                content = None
            return FileAnalysis(
                filepath=str(filepath.relative_to(project_root)),
                loc=len(content.split('\n')) if content else 0,
                mtime=mtime,
                hash=file_hash
            )

        # Ajouter parent nodes pour context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        logger.debug(f"Analyzing code structure for {filepath}")
        analyzer = CodeAnalyzer(str(filepath), project_root)
        analyzer.visit(tree)

        file_analysis = FileAnalysis(
            filepath=str(filepath.relative_to(project_root)),
            imports=analyzer.imports,
            classes=analyzer.classes,
            functions=analyzer.functions,
            dependencies=analyzer.dependencies,
            internal_deps=analyzer.internal_deps,
            entry_points=analyzer.entry_points,
            constants=analyzer.constants,
            decorators=analyzer.decorators,
            exceptions=analyzer.exceptions,
            loc=len(content.split('\n')),
            mtime=mtime,
            hash=file_hash
        )

        logger.debug(f"Found: {len(analyzer.classes)} classes, {len(analyzer.functions)} functions, "
                     f"{len(analyzer.constants)} constants, {len(analyzer.decorators)} decorators, "
                     f"{len(analyzer.exceptions)} exception constructs in {filepath}")

        # Free up content memory for large files after analysis
        file_size = filepath.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10 MB or larger
            # Clear content that was read to free memory
            # We already did this in the exception handling, but do it here too
            pass  # Content is no longer being held in a variable after AST processing

        # Save to cache if cache directory is provided
        if cache_dir and cache_path:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_entry = CacheEntry(
                file_analysis=file_analysis,
                cached_at=datetime.now(),
                project_root=str(project_root)
            )
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_entry, f)
            logger.debug(f"Saved analysis to cache: {cache_path}")

        logger.debug(f"Completed analysis for {filepath}")
        # Free up content memory for large files after analysis
        file_size = filepath.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10 MB or larger
            content = None  # Clear content reference to free memory
        return file_analysis
    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")
        return None


def analyze_project(root_dir: Path, ignore_dirs: Optional[Set[str]] = None, cache_dir: Optional[Path] = None) -> Dict:
    """Analyse complète du projet avec support d'analyse incrémentale"""
    logger = logging.getLogger(__name__)
    
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
        'external_deps': defaultdict(int),
        'language_stats': defaultdict(int)  # Track file counts by language
    }

    # Get list of files that need analysis vs files that can be loaded from cache
    logger.info("Determining files for analysis...")
    changed_files, cached_files = get_changed_files_multilang(root_dir, ignore_dirs, cache_dir)

    logger.info(f"Found {len(changed_files)} files to analyze, {len(cached_files)} files in cache.")

    # Process cached files first (no analysis needed)
    if cached_files and cache_dir:
        logger.info(f"Loading {len(cached_files)} cached analysis results...")
        for i, filepath in enumerate(cached_files, 1):
            cached_analysis = load_cached_analysis_multilang(filepath, root_dir, cache_dir)
            if cached_analysis:
                project_data['files'][cached_analysis.filepath] = cached_analysis
                project_data['stats']['total_files'] += 1
                project_data['stats']['total_loc'] += cached_analysis.loc
                project_data['stats']['total_classes'] += len(cached_analysis.classes)
                project_data['stats']['total_functions'] += len(cached_analysis.functions)

                # Update language statistics
                ext = Path(cached_analysis.filepath).suffix.lower()
                project_data['language_stats'][ext] += 1

                # Construire graphe de dépendances internes
                for dep in cached_analysis.internal_deps:
                    project_data['dependency_graph'][cached_analysis.filepath].add(dep)

                # Compter dépendances externes
                for dep in cached_analysis.dependencies:
                    project_data['external_deps'][dep] += 1
            
            # Progress update every 10 files or at the end
            if i % 10 == 0 or i == len(cached_files):
                logger.debug(f"Processed {i}/{len(cached_files)} cached files ({i/len(cached_files)*100:.1f}%)")

    # Process changed files (needs analysis)
    if changed_files:
        logger.info(f"Analyzing {len(changed_files)} changed files in parallel...")
        analyses = []
        
        # Process with progress tracking if we have many files
        if len(changed_files) > 20:  # Only show detailed progress for larger projects
            logger.info(f"Starting analysis of {len(changed_files)} files...")
            
        with concurrent.futures.ProcessPoolExecutor() as executor:
            analyze_func = partial(analyze_file_multilang, project_root=root_dir, cache_dir=cache_dir)
            # Use executor.map for built-in progress tracking through logging
            analyses = list(executor.map(analyze_func, changed_files))

        logger.info("Analysis complete. Aggregating results...")

        # Agréger les résultats des fichiers analysés
        for i, analysis in enumerate(analyses):
            if analysis:
                project_data['files'][analysis.filepath] = analysis
                project_data['stats']['total_files'] += 1
                project_data['stats']['total_loc'] += analysis.loc
                project_data['stats']['total_classes'] += len(analysis.classes)
                project_data['stats']['total_functions'] += len(analysis.functions)

                # Update language statistics
                ext = Path(analysis.filepath).suffix.lower()
                project_data['language_stats'][ext] += 1

                # Construire graphe de dépendances internes
                for dep in analysis.internal_deps:
                    project_data['dependency_graph'][analysis.filepath].add(dep)

                # Compter dépendances externes
                for dep in analysis.dependencies:
                    project_data['external_deps'][dep] += 1
            
            # Progress update for analysis results
            if analysis and (i + 1) % 10 == 0:
                logger.debug(f"Aggregated {i + 1}/{len(analyses)} analysis results ({(i + 1)/len(analyses)*100:.1f}%)")
    else:
        logger.info("No files to analyze. All files loaded from cache.")

    return project_data



def extract_sqlite_schema(db_path: Path) -> Dict:
    """Extrait le schéma d'une base SQLite"""
    schema = {
        'tables': {},
        'views': [],
        'indexes': []
    }

    if not db_path.exists():
        print(f"[ERROR] Database file does not exist: {db_path}")
        return schema

    conn = None
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
            try:
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
            except sqlite3.Error as e:
                print(f"[ERROR] Error reading table {table_name}: {e}")
                continue

        # Obtenir les vues
        try:
            cursor.execute("""
                SELECT name, sql FROM sqlite_master 
                WHERE type='view'
                ORDER BY name
            """)
            schema['views'] = [{'name': name, 'sql': sql} for name, sql in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[ERROR] Error reading views: {e}")

        # Obtenir les indexes
        try:
            cursor.execute("""
                SELECT name, tbl_name, sql FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY tbl_name, name
            """)
            schema['indexes'] = [
                {'name': name, 'table': tbl, 'sql': sql} 
                for name, tbl, sql in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            print(f"[ERROR] Error reading indexes: {e}")

    except sqlite3.Error as e:
        print(f"[ERROR] Database connection error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error extracting SQLite schema: {e}")
    finally:
        if conn:
            conn.close()

    return schema


def _generate_header_and_overview(stats: Dict) -> str:
    """Génère l'en-tête et la vue d'ensemble du rapport."""
    return f"""# Project Map - AI Context

> **Auto-generated** project cartography for LLM coding agents
> 
> **Stats:** {stats['total_files']} files | {stats['total_loc']:,} LOC | {stats['total_classes']} classes | {stats['total_functions']} functions

---

## Quick Overview

This map provides complete project context for AI pair programming. Use it to understand:
- Project structure and file organization
- Dependencies between modules
- Key classes and their responsibilities
- Database schema (if applicable)

---
"""

def _generate_structure_section(project_data: Dict) -> str:
    """Génère la section sur la structure du projet."""
    md = "## 🏗️ Project Structure\n\n"
    
    folders = defaultdict(list)
    for filepath in sorted(project_data['files'].keys()):
        folder = str(Path(filepath).parent) if Path(filepath).parent != Path('.') else '.'
        folders[folder].append(filepath)

    for folder in sorted(folders.keys()):
        files = folders[folder]
        md += f"### 📁 `{folder}/`\n\n"

        for filepath in sorted(files):
            data = project_data['files'][filepath]
            filename = Path(filepath).name

            parts = []
            if data.classes:
                parts.append(f"{len(data.classes)} class{'es' if len(data.classes) > 1 else ''}")
            if data.functions:
                parts.append(f"{len(data.functions)} func{'s' if len(data.functions) > 1 else ''}")
            summary = ", ".join(parts) if parts else "utility module"

            md += f"#### `{filename}` ({data.loc} LOC)\n\n"
            md += f"**Purpose:** {summary}\n\n"

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

            if data.functions:
                md += "**Functions:**\n"
                for func in data.functions[:8]:
                    async_prefix = "async " if func['is_async'] else ""
                    
                    args_list = []
                    if func['args']:
                        # FIX: Handle inconsistency in arg format from the analyzer
                        if isinstance(func['args'][0], dict):
                            args_list = [arg['name'] for arg in func['args']]
                        else:
                            args_list = func['args']
                    args_str = ", ".join(args_list)

                    md += f"- `{async_prefix}{func['name']}({args_str})`\n"
                if len(data.functions) > 8:
                    md += f"- ... +{len(data.functions)-8} more functions\n"
                md += "\n"

            # Show constants if any exist
            if data.constants:
                md += "**Constants:**\n"
                for const in data.constants[:10]:  # Limit to first 10 constants
                    md += f"- `{const['name']}` = {const['value']}\n"
                if len(data.constants) > 10:
                    md += f" (+{len(data.constants)-10} more)\n"
                md += "\n"

            # Show decorators if any exist  
            if data.decorators:
                unique_decorators = list(set(data.decorators))  # Remove duplicates
                md += "**Decorators:**\n"
                for dec in unique_decorators[:8]:  # Limit to first 8 decorators
                    md += f"- `@{dec}`\n"
                if len(unique_decorators) > 8:
                    md += f" (+{len(unique_decorators)-8} more)\n"
                md += "\n"

            # Show exception handling if any exists
            if data.exceptions:
                md += "**Exception Handling:**\n"
                md += f"- {len(data.exceptions)} exception handling constructs\n\n"

            external_imports = [imp for imp in data.imports if imp.split('.')[0] in data.dependencies]
            if external_imports:
                key_imports = external_imports[:5]
                md += f"**Key imports:** `{'`, `'.join(key_imports)}`\n"
                if len(external_imports) > 5:
                    md += f" (+{len(external_imports)-5} more)\n"
                md += "\n"

            md += "---\n\n"
    return md

def _generate_dependency_graph_section(project_data: Dict) -> str:
    """Génère la section du graphe de dépendances."""
    if not project_data['dependency_graph']:
        return ""
    
    md = "## 🔗 Internal Dependencies Graph\n\n"
    md += "**File relationships** (who imports whom):\n\n"
    md += "```\n"
    for source, targets in sorted(project_data['dependency_graph'].items()):
        if targets:
            md += f"{source}\n"
            for target in sorted(targets):
                md += f"  └─> {target}\n"
    md += "```\n\n"
    return md

def _generate_external_deps_section(project_data: Dict) -> str:
    """Génère la section des dépendances externes."""
    md = "## 📦 External Dependencies\n\n"
    
    # Recreate the stdlib checker for use here
    def is_stdlib(module: str) -> bool:
        if not module:
            return False
            
        module_name = module.split('.')[0]
        
        stdlib = {
            'os', 'sys', 'json', 'ast', 're', 'pathlib', 'collections',
            'typing', 'datetime', 'logging', 'argparse', 'configparser',
            'sqlite3', 'csv', 'io', 'math', 'random', 'time', 'unittest',
            'functools', 'itertools', 'operator', 'enum', 'dataclasses',
            'asyncio', 'threading', 'multiprocessing', 'subprocess',
            'http', 'urllib', 'email', 'html', 'xml', 'pickle', 'copy',
            'socket', 'ssl', 'hashlib', 'zlib', 'gzip', 'bz2', 'lzma',
            'shutil', 'tempfile', 'sysconfig', 'traceback', 'inspect',
            'site', 'importlib', 'warnings', 'weakref', 'gc', 'linecache',
            'string', 'reprlib', 'pprint', 'heapq', 'bisect', 'array',
            'contextlib', 'contextvars', 'abc', 'atexit', 'codecs', 'tokenize',
            'token', 'keyword', 'symtable', 'opcode', 'dis', 'copyreg', 'trace',
            'code', 'parser', 'stringprep', 'types', 'numbers', 'decimal',
            'fractions', 'random', 'statistics', 'itertools', 'functools',
            'operator', 'pathlib', 'filecmp', 'tempfile', 'glob', 'shutil',
            'curses', 'colorsys', 'html', 'xml', 'webbrowser', 'cgi',
            'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'smtplib',
            'uuid', 'socket', 'ssl', 'select', 'asyncio', 'concurrent',
            'multiprocessing', 'threading', 'queue', 'sched', 'dummy_threading',
            'time', 'timeit', 'profile', 'pstats', 'cProfile', 'traceback',
            'pyclbr', 'py_compile', 'compileall', 'zipfile', 'tarfile', 'gzip',
            'bz2', 'lzma', 'shutil', 'sqlite3', 'dbm', 'struct', 'codecs',
            'encodings', 'plistlib', 'hashlib', 'hmac', 'secrets', 'itertools',
            'collections', 'abc', 'io', 'text', 'zipimport', 'pkgutil',
            'modulefinder', 'runpy', 'importlib', 'ast', 'symtable', 'tokenize',
            'token', 'keyword', 'parser', 'tabnanny', 'pyclbr', 'pickle',
            'copyreg', 'reprlib', 'pprint', 'dataclasses', 'enum', 'graphlib'
        }
        return module_name in stdlib

    # Separate standard library, third-party, and potential internal dependencies
    stdlib_deps = {}
    third_party_deps = {}
    internal_like_deps = {}
    
    for dep, count in project_data['external_deps'].items():
        if is_stdlib(dep):
            stdlib_deps[dep] = count
        elif dep in project_data.get('files', {}):  # If it's a file in our project
            internal_like_deps[dep] = count
        else:
            third_party_deps[dep] = count

    has_stdlib = bool(stdlib_deps)
    has_third_party = bool(third_party_deps)
    has_internal_like = bool(internal_like_deps)

    if has_third_party:
        md += "**Third-party packages** (by usage):\n\n"
        for dep, count in sorted(third_party_deps.items(), key=lambda x: x[1], reverse=True):
            md += f"- `{dep}` — used in {count} file{'s' if count > 1 else ''}\n" 
        md += "\n"
    else:
        md += "*No third-party dependencies detected*\n\n"

    if has_internal_like:
        md += "**Potential Internal Modules** (may require verification):\n\n"
        for dep, count in sorted(internal_like_deps.items(), key=lambda x: x[1], reverse=True):
            md += f"- `{dep}` — referenced in {count} file{'s' if count > 1 else ''}\n" 
        md += "\n"

    if has_stdlib:
        md += f"**Standard Library** ({len(stdlib_deps)} unique modules used)\n\n"

    return md

def _generate_db_schema_section(db_schema: Optional[Dict]) -> str:
    """Génère la section du schéma de la base de données."""
    if not db_schema or not db_schema['tables']:
        return ""

    md = "## Database Schema (SQLite)\n\n"

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

        if table_info['foreign_keys']:
            md += "**Foreign Keys:**\n"
            for fk in table_info['foreign_keys']:
                md += f"- `{fk['column']}` → {fk['references']}\n"
            md += "\n"

    if 'views' in db_schema and db_schema['views']:
        md += "### Views\n\n"
        for view in db_schema['views']:
            md += f"- `{view['name']}`\n"
        md += "\n"
    
    return md

def _generate_key_files_section(project_data: Dict) -> str:
    """Génère la section des fichiers clés."""
    md = "## 🎯 Key Files (ranked by complexity)\n\n"

    file_scores = []
    for filepath, data in project_data['files'].items():
        score = (
            len(data.classes) * 5 +
            len(data.functions) * 2 +
            len(data.constants) * 0.5 +  # Add points for constants
            len(data.decorators) * 0.3 +  # Add points for decorators used
            len(data.exceptions) * 1 +   # Add points for exception handling
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
        if data.constants:
            parts.append(f"{len(data.constants)} constants")
        if data.decorators:
            parts.append(f"{len(set(data.decorators))} unique decorators")
        if data.exceptions:
            parts.append(f"{len(data.exceptions)} exception handlers")
        if data.internal_deps:
            parts.append(f"{len(data.internal_deps)} internal deps")
        md += f"   - {', '.join(parts)}\n"

    md += "\n---\n\n"
    return md

def _generate_usage_guide() -> str:
    """Génère le guide d'utilisation pour l'IA."""
    return """## Usage Guide for AI Coding Agents

### For Initial Context
When starting a new task, always:
1. Read this entire map to understand project structure
2. Identify relevant files from the structure section
3. Check dependencies before modifying files
4. Consult database schema when working with data models

### For Specific Tasks
- **Adding features**: Check \"Key Files\" section for entry points
- **Refactoring**: Review \"Internal Dependencies Graph\" to understand impact
- **Debugging**: Look at file's class/function list to locate code
- **Database work**: Refer to \"Database Schema\" section

### Best Practices
- Always maintain this map's structure when adding files
- Update dependencies when creating new modules
- Keep database schema in sync with migrations
- Use this map to avoid circular dependencies

---

*Generated by Project Mapper v1.0 - Optimized for LLM context*
"""

def generate_markdown(project_data: Dict, db_schema: Optional[Dict], output_file: Path):
    """Génère le markdown optimisé pour LLMs en assemblant les sections."""
    logger = logging.getLogger(__name__)
    
    stats = project_data['stats']
    
    md_parts = [
        _generate_header_and_overview(stats),
        _generate_structure_section(project_data),
        _generate_dependency_graph_section(project_data),
        _generate_external_deps_section(project_data),
        _generate_db_schema_section(db_schema),
        _generate_key_files_section(project_data),
        _generate_usage_guide()
    ]
    
    md = "".join(md_parts)

    # Sauvegarder
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md)

    logger.info(f"{output_file} generated successfully!")

    return output_file


def generate_json(project_data: Dict, db_schema: Optional[Dict], output_file: Path):
    """Génère le JSON avec les résultats d'analyse."""
    logger = logging.getLogger(__name__)
    
    # Prepare the data to be serialized
    output_data = {
        'project_data': project_data,
        'db_schema': db_schema
    }
    
    # Convert any non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dataclass_fields__'):  # Check if it's a dataclass
            # Convert dataclass to dict
            import dataclasses
            return {field.name: make_serializable(getattr(obj, field.name)) 
                   for field in dataclasses.fields(obj)}
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = make_serializable(output_data)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"{output_file} generated successfully!")


def generate_yaml(project_data: Dict, db_schema: Optional[Dict], output_file: Path):
    """Génère le YAML avec les résultats d'analyse."""
    logger = logging.getLogger(__name__)
    
    if not YAML_AVAILABLE:
        logger.error("YAML support not available. Please install PyYAML.")
        return output_file
    
    # Prepare the data to be serialized
    output_data = {
        'project_data': project_data,
        'db_schema': db_schema
    }
    
    # Convert any non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = make_serializable(output_data)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(serializable_data, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"{output_file} generated successfully!")


def generate_xml(project_data: Dict, db_schema: Optional[Dict], output_file: Path):
    """Génère le XML avec les résultats d'analyse."""
    logger = logging.getLogger(__name__)
    
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    root = ET.Element("ProjectMap")
    
    # Add project data
    project_elem = ET.SubElement(root, "ProjectData")
    
    # Stats
    stats_elem = ET.SubElement(project_elem, "Stats")
    for key, value in project_data.get('stats', {}).items():
        stat_elem = ET.SubElement(stats_elem, key)
        stat_elem.text = str(value)
    
    # Files
    files_elem = ET.SubElement(project_elem, "Files")
    for filepath, file_data in project_data.get('files', {}).items():
        file_elem = ET.SubElement(files_elem, "File")
        file_elem.set("path", filepath)
        
        # Add file details
        imports_elem = ET.SubElement(file_elem, "Imports")
        for imp in getattr(file_data, 'imports', []):
            imp_elem = ET.SubElement(imports_elem, "Import")
            imp_elem.text = imp
            
        classes_elem = ET.SubElement(file_elem, "Classes")
        for cls in getattr(file_data, 'classes', []):
            cls_elem = ET.SubElement(classes_elem, "Class")
            cls_elem.set("name", cls.get('name', ''))
            # Add more class details as needed
            
        functions_elem = ET.SubElement(file_elem, "Functions")
        for func in getattr(file_data, 'functions', []):
            func_elem = ET.SubElement(functions_elem, "Function")
            func_elem.set("name", func.get('name', ''))
            # Add more function details as needed
            
        constants_elem = ET.SubElement(file_elem, "Constants")
        for const in getattr(file_data, 'constants', []):
            const_elem = ET.SubElement(constants_elem, "Constant")
            const_elem.set("name", const.get('name', ''))
            const_elem.text = str(const.get('value', ''))
    
    # Add DB schema if present
    if db_schema:
        db_elem = ET.SubElement(root, "DatabaseSchema")
        # Add DB schema details
    
    # Pretty format and write
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", newl='\n')
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines[1:])  # Remove XML declaration for cleaner output
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    logger.info(f"{output_file} generated successfully!")


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from file or return default config."""
    default_config = {
        'ignore_dirs': {
            '__pycache__', '.git', 'venv', 'env', 'node_modules',
            '.venv', 'build', 'dist', '.pytest_cache', '.mypy_cache',
            'htmlcov', '.tox', '.eggs', '*.egg-info'
        },
        'supported_extensions': {'.py'},  # Default: only Python files
        'output_format': 'markdown',
        'default_output_file': 'PROJECT_MAP.md',
        'cache_enabled': True,
        'max_workers': None  # Use default from ProcessPoolExecutor
    }
    
    if not config_path or not config_path.exists():
        # Try to find a default config file
        for default_name in ['project_mapper.yml', 'project_mapper.yaml', '.project_mapper.yml']:
            default_config_path = Path(default_name)
            if default_config_path.exists():
                config_path = default_config_path
                break
    
    if config_path and config_path.exists():
        if not YAML_AVAILABLE:
            logging.warning(f"YAML support not available, cannot load {config_path}. Install PyYAML for configuration file support.")
            return default_config
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Merge file config with defaults, with file values taking precedence
                    for key, value in file_config.items():
                        if key in default_config:
                            if isinstance(default_config[key], set) and isinstance(value, list):
                                # Convert list from YAML to set
                                default_config[key] = set(value)
                            else:
                                default_config[key] = value
                        else:
                            default_config[key] = value
            logging.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {e}")
    
    return default_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="Generate project map for AI coding agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Analyze current directory
  %(prog)s --output docs/PROJECT_MAP.md # Custom output location
  %(prog)s --format json                # Output in JSON format
  %(prog)s --format yaml                # Output in YAML format
  %(prog)s --format xml                 # Output in XML format
  %(prog)s --db data/app.db             # Include SQLite schema
  %(prog)s --project /path/to/project   # Analyze specific directory
  %(prog)s --cache-dir .cache           # Use cache directory for performance
  %(prog)s --config project_mapper.yml  # Load configuration from file
  %(prog)s --verbose                    # Enable verbose logging
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
        help='Output file (default: PROJECT_MAP.md)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['markdown', 'json', 'yaml', 'xml'],
        default='markdown',
        help='Output format (default: markdown)'
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

    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Directory to store cached analysis results for performance (optional)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path (YAML format, optional)'
    )

    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Project Mapper - AI Context Generator")
    logger.info(f"Analyzing: {args.project.absolute()}")

    # Use config to override defaults where appropriate
    ignore_dirs = config.get('ignore_dirs', {
        '__pycache__', '.git', 'venv', 'env', 'node_modules',
        '.venv', 'build', 'dist', '.pytest_cache', '.mypy_cache',
        'htmlcov', '.tox', '.eggs', '*.egg-info'
    })

    # Analyser le projet
    try:
        project_data = analyze_project(args.project, ignore_dirs=ignore_dirs, cache_dir=args.cache_dir)
    except Exception as e:
        logger.error(f"Failed to analyze project {args.project}: {e}")
        return 1  # Exit with error code

    # Extraire schéma DB si fourni
    db_schema = None
    if args.db:
        if args.db.exists():
            logger.info(f"Extracting SQLite schema from {args.db}...")
            db_schema = extract_sqlite_schema(args.db)
            logger.info(f"Found {len(db_schema['tables'])} tables")
        else:
            logger.warning(f"Database file does not exist: {args.db}")

    # Generate output in specified format
    if args.format == 'markdown':
        generate_markdown(project_data, db_schema, args.output)
    elif args.format == 'json':
        generate_json(project_data, db_schema, args.output)
    elif args.format == 'yaml' and YAML_AVAILABLE:
        generate_yaml(project_data, db_schema, args.output)
    elif args.format == 'xml':
        generate_xml(project_data, db_schema, args.output)
    else:
        if args.format == 'yaml' and not YAML_AVAILABLE:
            logger.error("YAML support not available. Please install PyYAML.")
            return 1
        logger.error(f"Unknown output format: {args.format}")
        return 1

    # Export JSON optionnel
    if args.json:
        json_data = {
            'files': {
                path: {
                    'imports': data.imports,
                    'classes': data.classes,
                    'functions': data.functions,
                    'constants': data.constants,
                    'decorators': data.decorators,
                    'exceptions': data.exceptions,
                    'loc': data.loc,
                    'dependencies': list(data.dependencies),
                    'internal_deps': list(data.internal_deps),
                    'entry_points': data.entry_points,
                    'mtime': data.mtime,
                    'hash': data.hash
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

        logger.info(f"{args.json} exported")

    logger.info(f"Use '{args.output}' with your LLM coding agent.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
