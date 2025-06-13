# Auteur: Mohamed NAJID

import os
import sys
import subprocess
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

def check_dependencies():
    print("Verification des installations")
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "nltk"
    ]
    
    all_ok = True
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installation {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--user", pkg])
            except subprocess.CalledProcessError as e:
                 print(f"Erreur installation {pkg}: {e}")
                 all_ok = False

    if all_ok:
        print("Dépendances ready.")
    else:
        print("épendances non installées")
    return all_ok

def run_script(script_path, step_name):
    print(f"\n--- Exécution: {step_name}")
    full_script_path = os.path.join(script_dir, script_path)
    if not os.path.exists(full_script_path):
        print(f"Erreur: Script {full_script_path} non trouvé!")
        return False
        
    try:
        result = subprocess.run([sys.executable, full_script_path], check=False, capture_output=True, text=True, cwd=script_dir, encoding='utf-8')
        print(result.stdout)
        if result.returncode != 0:
             print(f"Erreur dans {step_name} (code: {result.returncode}):")
             print(result.stderr)
             return False
        print(f"\n--- {step_name} terminé --- ")
        return True
    except Exception as e:
        print(f"Erreur inconnue pendant {step_name}: {e}")
        return False

def main_pipeline():
    print("\n---- Pipeline Analyse CV ----\n")
    
    if not check_dependencies():
        print("arrêt du pipeline à cause d'erreurs de dépendances.")
        return

    script_preprocess = "1_pretraitement.py"
    script_analyze = "2_vectorisation_classification.py"
    script_validate = "3_validation.py"

    if not run_script(script_preprocess, "Prétraitement"): return
    if not run_script(script_analyze, "Analyse Compétences"): return
    if not run_script(script_validate, "Validation"): return
    
    print("\n-> Pipeline Terminé <-\n")
    print(f"Fichiers générés dans: {script_dir}") 
    print("- cv_pretraites.csv")
    print("- competences.csv")
    print("- competences_by_category.txt")
    print("- validation_report.txt")
    print("- competences_*.png")

if __name__ == "__main__":
    main_pipeline()
