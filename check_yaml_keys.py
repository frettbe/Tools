#!/usr/bin/env python3
"""
Script pour détecter les clés YAML inutilisées dans fr.yaml.

Ce script :
1. Parse le fichier fr.yaml
2. Scanne tous les fichiers Python du projet
3. Détecte tous les appels à translate()
4. Compare et liste les clés inutilisées
5. Optionnel : Génère un fr.yaml nettoyé

Usage:
    python check_yaml_keys.py
    python check_yaml_keys.py --clean  # Génère fr_clean.yaml
"""

import re
import yaml
from pathlib import Path


def load_yaml_keys(yaml_path: Path) -> set[str]:
    """
    Charge toutes les clés d'un fichier YAML (avec hiérarchie à plat).
    
    Exemple: {'window.title', 'menu.file', 'menu.file.new', ...}
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    keys = set()
    
    def flatten(d, prefix=''):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, dict):
                flatten(v, full_key)
    
    flatten(data)
    return keys


def find_translate_calls(code_dir: Path) -> set[str]:
    """
    Trouve tous les appels à translate() dans le code Python.
    
    Patterns détectés :
    - translate("key")
    - translate('key')
    - translate("key.subkey")
    - translate(f"key.{var}")  # Dynamique, on garde "key"
    """
    # Pattern pour capturer les appels translate()
    pattern = re.compile(
        r'translate\s*\(\s*["\']([a-zA-Z0-9_.]+)["\']',
        re.MULTILINE
    )
    
    used_keys = set()
    
    # Scanner tous les fichiers .py
    for py_file in code_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            matches = pattern.findall(content)
            used_keys.update(matches)
        except Exception as e:
            print(f"⚠️  Erreur lecture {py_file}: {e}")
    
    return used_keys


def analyze_keys(yaml_keys: set[str], used_keys: set[str]) -> dict:
    """Compare les clés YAML avec les clés utilisées dans le code."""
    unused = yaml_keys - used_keys
    missing = used_keys - yaml_keys
    
    # Filtrer les clés partielles (ex: "menu" est parent de "menu.file")
    # Une clé est inutilisée uniquement si elle n'a pas d'enfants utilisés
    actually_unused = set()
    for key in unused:
        # Vérifier si c'est une clé feuille (pas de sous-clés utilisées)
        has_children = any(k.startswith(f"{key}.") for k in used_keys)
        if not has_children:
            actually_unused.add(key)
    
    return {
        'unused': sorted(actually_unused),
        'missing': sorted(missing),
        'total_yaml': len(yaml_keys),
        'total_used': len(used_keys),
    }


def generate_clean_yaml(yaml_path: Path, unused_keys: set[str], output_path: Path):
    """Génère un fichier YAML sans les clés inutilisées."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    def clean_dict(d, prefix=''):
        to_delete = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if full_key in unused_keys:
                to_delete.append(k)
            elif isinstance(v, dict):
                clean_dict(v, full_key)
                # Supprimer les dicts vides après nettoyage
                if not v:
                    to_delete.append(k)
        
        for k in to_delete:
            del d[k]
    
    clean_dict(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
    
    print(f"✅ Fichier nettoyé généré : {output_path}")


def main():
    import sys
    
    # Chemins
    project_root = Path(__file__).parent
    yaml_path = project_root / "lang" / "fr.yaml"
    code_dir = project_root / "libapp"
    
    print("🔍 ANALYSE DES CLÉS YAML")
    print("=" * 60)
    
    # 1. Charger les clés YAML
    print(f"📄 Lecture de {yaml_path}...")
    yaml_keys = load_yaml_keys(yaml_path)
    print(f"   → {len(yaml_keys)} clés trouvées")
    
    # 2. Scanner le code
    print(f"🔎 Scan du code dans {code_dir}...")
    used_keys = find_translate_calls(code_dir)
    print(f"   → {len(used_keys)} clés utilisées")
    
    # 3. Analyser
    print("\n📊 RÉSULTATS")
    print("=" * 60)
    
    results = analyze_keys(yaml_keys, used_keys)
    
    print(f"✅ Clés YAML définies : {results['total_yaml']}")
    print(f"✅ Clés utilisées code : {results['total_used']}")
    print(f"🗑️  Clés inutilisées   : {len(results['unused'])}")
    print(f"⚠️  Clés manquantes    : {len(results['missing'])}")
    
    # Afficher les clés inutilisées
    if results['unused']:
        print("\n🗑️  CLÉS INUTILISÉES (peuvent être supprimées) :")
        for key in results['unused'][:20]:  # Top 20
            print(f"   • {key}")
        if len(results['unused']) > 20:
            print(f"   ... et {len(results['unused']) - 20} autres")
    
    # Afficher les clés manquantes
    if results['missing']:
        print("\n⚠️  CLÉS MANQUANTES (à ajouter dans fr.yaml) :")
        for key in results['missing'][:20]:  # Top 20
            print(f"   • {key}")
        if len(results['missing']) > 20:
            print(f"   ... et {len(results['missing']) - 20} autres")
    
    # Option --clean
    if '--clean' in sys.argv:
        output_path = project_root / "lang" / "fr_clean.yaml"
        print(f"\n🧹 Génération du fichier nettoyé...")
        generate_clean_yaml(yaml_path, set(results['unused']), output_path)
    else:
        print("\n💡 Astuce : Lance avec --clean pour générer fr_clean.yaml")
    
    print("\n✅ Analyse terminée !")


if __name__ == "__main__":
    main()
