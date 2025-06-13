# Auteur: Mohamed NAJID

# Importation des bibliotheques
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
from collections import Counter 
import nltk
from nltk.tokenize import word_tokenize
import re 
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Verification NLTK 
print("Vérification NLTK")
try:
    # Vérifier si la ressource pour la tokenisation est présente
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Fonctions 

def load_data_and_skills():
    # Définir les chemins des fichiers à charger 
    cv_path = os.path.join(script_dir, "cv_pretraites.csv")
    skills_path = os.path.join(script_dir, "competences.csv")
    
    # Verifier si les fichiers nécessaires existent
    if not os.path.exists(cv_path) or not os.path.exists(skills_path):
        print("Erreur: Fichiers cv_pretraites.csv ou competences.csv non trouvés.")
        print("Exécutez les scripts 1 et 2 d'abord.")
        return None, None # Retourne None si un fichier manque
    
    try:
        df_cv = pd.read_csv(cv_path)
        df_skills = pd.read_csv(skills_path)
        # Vérifie si la colonne contenant les compétences existe dans le fichier competences.csv
        if 'skill' not in df_skills.columns: 
            print(f"Erreur: Colonne 'skill' manquante dans {skills_path}")
            return df_cv, [] # retourne les CV mais une liste de competences vide
        # Récupère la liste des compétences 
        skills = df_skills['skill'].dropna().tolist()
        return df_cv, skills
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers: {e}")
        return None, None

def find_skills_in_cv_with_context(cv_text, skills_list):
    if not isinstance(cv_text, str):
        return [] 
        
    cv_lower = cv_text.lower() 
    tokens = word_tokenize(cv_lower) 
    results = [] # Liste pour stocker les resultats 
    # S'assurer que la liste de competences contient des chaines
    skills_list = [str(s) for s in skills_list if pd.notna(s)]

    # Parcourt chaque competence à chercher
    for skill in skills_list:
        skill_lower = skill.lower()
        context = "" #
        found = False # Indicateur si la competence est trouvée
        
        # Si la competence contient un espace 
        if " " in skill_lower:
            # Cherche la chaine exacte
            if skill_lower in cv_lower:
                try:
                    # Trouve la premiere occurrence de la competence
                    match = next(re.finditer(re.escape(skill_lower), cv_lower))
                    # Extrait le texte autour de la competence 
                    start = max(0, match.start() - 20)
                    end = min(len(cv_lower), match.end() + 20)
                    context = f"...{cv_lower[start:end]}..."
                    found = True
                except StopIteration:
                    pass 
        # Si la competence est un mot seul
        elif skill_lower in tokens:
            try:
                # Trouve l'index du mot dans la liste des tokens
                idx = tokens.index(skill_lower)
                # Extrait les mots autour 
                start = max(0, idx - 3)
                end = min(len(tokens), idx + 4)
                context = f"...{' '.join(tokens[start:end])}..."
                found = True
            except ValueError:
                 pass 
        # Si la competence est trouvee, l'ajoute aux resultats 
        if found:
            results.append((skill, context))
            
    # Supprimer les doublons 
    unique_results = []
    seen_skills = set()
    for skill, context in results:
        if skill not in seen_skills:
            unique_results.append((skill, context))
            seen_skills.add(skill)
            
    return unique_results

def analyze_examples(df, skills_list, num_examples=3):
    print(f"Analyse de {num_examples} exemples")
    # Verification si les donnees sont valides
    if df is None or df.empty:
        print("DataFrame vide")
        return {}
    if not skills_list:
        print("Liste de compétences vide")
        return {}
        
    # S'assurer qu'on ne demande pas plus d'exemples qu'il y a de CV
    actual_examples = min(num_examples, len(df))
    if actual_examples == 0:
        print("Pas assez de données pour sélectionner des exemples.")
        return {}
        
    # Choisit aleatoirement des indices de CV à analyser
    example_indices = np.random.choice(len(df), size=actual_examples, replace=False)
    examples_df = df.iloc[example_indices]
    
    analysis_results = {} # Dictionnaire pour stocker les resultats
    # Parcourt les CV exemples
    for _, row in examples_df.iterrows():
        cv_id = row["ID"]
        category = row["Category"]
        # Récuperation le texte nettoyé
        cv_text = row["cleaned_text"] if pd.notna(row["cleaned_text"]) else ""
        
        # Trouver les competences et leur contexte dans ce CV
        identified_skills = find_skills_in_cv_with_context(cv_text, skills_list)
        # Stockage les informations
        analysis_results[cv_id] = {
            "category": category,
            "skills_found": identified_skills
        }
    return analysis_results

def plot_skills_by_category(df, skills_list):
    print("Génération des graphiques par catégorie") 
    # Verification si les donnees sont valides 
    if df is None or df.empty or 'Category' not in df.columns:
        print("Données invalides pour la visualisation.")
        return
        
    categories = df["Category"].unique() # Liste des categories uniques
    category_counters = {} # Dictionnaire pour stocker les compteurs par categorie
    
    # Sélectionner les CV de la catégorie, creation un compteur pour cette catégorie, parcourir les CV de la catégorie, repitition de cet operation pour chaque catégorie
    for category in categories:
        category_df = df[df["Category"] == category] 
        counter = Counter() 
        for _, row in category_df.iterrows():
            cv_text = row["cleaned_text"] if pd.notna(row["cleaned_text"]) else ""
            # Trouver les competences 
            skills_in_cv = [skill for skill, _ in find_skills_in_cv_with_context(cv_text, skills_list)]
            counter.update(skills_in_cv) # Met à jour le compteur
        # Stocke les 10 competences les plus frequentes pour cette catégorie
        category_counters[category] = counter.most_common(10)
        
    # Definire le dossier de sortie pour les graphiques 
    output_dir = script_dir 
    try:
        os.makedirs(output_dir, exist_ok=True) 
    except Exception as e:
        print(f"Erreur lors de la création du dossier de sortie {output_dir}: {e}")
        return

    print(f"Début de la boucle de création des graphiques pour {len(category_counters)} catégories")
    # Pour chaque categorie et ses top competences
    for category, top_skills in category_counters.items():
        print(f"Traitement catégorie: {category}")
        if not top_skills:
            print(f"Aucune compétence à afficher pour {category}")
            continue # passer au catégorie suivante si pas de compétences trouvees
            
        # Séparer les competences et leurs comptes
        skills, counts = zip(*top_skills)
        
        safe_category_name = re.sub(r'[^\w\-.]+', '_', str(category)) 
        # Definir le chemin complet pour sauvegarder le graphique
        plot_filename = os.path.join(output_dir, f"competences_{safe_category_name}.png") 
        print(f"Préparation du graphique: {plot_filename}")

        # Try/Except pour la creation et sauvegarde du graphique
        fig = None 
        try:
            fig = plt.figure(figsize=(10, 6))
            # Creation d'un graphique à barres horizontales
            plt.barh(skills, counts, color='skyblue')
            plt.xlabel("Fréquence")
            plt.ylabel("Compétence")
            plt.title(f"Top 10 Compétences - Catégorie: {category}")
            plt.gca().invert_yaxis() # Affiche la competence la plus fréquente en haut
            plt.tight_layout() # Ajuste la mise en page
            
            # Sauvegarde le graphique
            print(f"Tentative sauvegarde: {plot_filename}")
            plt.savefig(plot_filename)
            print(f"Graphique sauvegardé avec succès: {plot_filename}")
            
        except Exception as e:
            # Affiche une erreur détaillée si la création ou sauvegarde échoue
            print(f" erreur lors de la creation/sauvegarde du graphique pour la catégorie '{category}' ")
            
        finally:
            # Assure que la figure est fermee
            if fig is not None:
                try:
                    plt.close(fig)
                    print(f"Figure pour '{category}' fermée.")
                except Exception as close_err:
                    print(f"Erreur lors de la fermeture de la figure pour '{category}': {close_err}")
            else:
                 print(f"Aucune figure à fermer pour '{category}'.")
        
    print("Fin de la boucle de création des graphiques.")

def save_validation_report(results, output_path):
    print(f"Sauvegarde rapport validation: {output_path}")
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("Rapport de Validation - Extraction Compétences\n")
            f.write("="*40 + "\n\n")
            
            # pour chaque CV exemple analysé
            for cv_id, data in results.items():
                f.write(f"CV ID: {cv_id} (Catégorie: {data['category']})\n")
                f.write("Compétences trouvées:\n")
                # Si aucune competence trouvée
                if not data["skills_found"]:
                    f.write("  (aucune)\n")
                # Si des competences sont trouvees
                else:
                    # Affichage de chaque competence et son contexte 
                    for skill, context in data["skills_found"]:
                        context_display = context[:100] + ('...' if len(context) > 100 else '')
                        f.write(f"  - {skill}: {context_display}\n")
                f.write("-"*40 + "\n")
        print("Rapport OK.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du rapport: {e}")

# Fonction principale du script 

def run_validation():
    df_cv, skills = load_data_and_skills()
    if df_cv is None or skills is None:
        print("arret de la validation car les données n'ont pas pu être chargées.")
        return
        
    # Verification la presence de la colonne de texte et gestion les NaN
    if 'cleaned_text' not in df_cv.columns:
        print("Erreur: Colonne 'cleaned_text' manquante dans le fichier prétraité.")
        return
    df_cv['cleaned_text'] = df_cv['cleaned_text'].fillna('')

    # Analyser quelques exemples
    example_results = analyze_examples(df_cv, skills, num_examples=3)
    
    # Bloc Try/Except autour de la generation des graphiques
    try:
        # Crée les graphiques par catégorie
        plot_skills_by_category(df_cv, skills)
    except Exception as plot_err:
        print(f"### ERREUR de plot ###")
        print(f"Erreur: {plot_err}")
        import traceback
        traceback.print_exc()
        print("#################################################################")

    
    # Definir le chemin pour le rapport de validation 
    report_path = os.path.join(script_dir, "validation_report.txt") 
    # Sauvegarde le rapport
    save_validation_report(example_results, report_path)
    
    print("\nValidation terminée.")
    print(f"Rapport: {report_path}")
    print(f"Graphiques sauvegardés dans: {script_dir}") # Message modifié

if __name__ == "__main__":
    run_validation()

