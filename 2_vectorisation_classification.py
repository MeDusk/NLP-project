# Auteur: Mohamed NAJID

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import os

# récupère le chemin du dossier où se trouve ce script
script = os.path.dirname(os.path.abspath(__file__))

# Vérification des ressources NLTK
print("Verification NLTK")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Fonctions
def load_processed_data(csv_path):
    print(f"Chargement: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Erreur: Fichier {csv_path} non trouve")
        return None
    try:
        df = pd.read_csv(csv_path)
        # Reconvertir les chaines de caractères en listes
        df["tokens"] = df["tokens"].apply(lambda x: x.split() if isinstance(x, str) else [])
        df["entities"] = df["entities"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
        print(f"CV chargés: {len(df)}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement de {csv_path}: {e}")
        return None

def vectorize_tfidf(texts):
    print("Vectorisation TF-IDF...")
    # Creation d'un objet TfidfVectorizer avec des paramètres
    vectorizer = TfidfVectorizer(
        max_features=3000, 
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 1)
    )
    # Appliquer la transformation TF-IDF aux textes
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"Matrice TF-IDF: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer # Retourne la matrice et l'objet vectorizer

def extract_skills_tfidf(vectorizer, num_skills=150):
    print(f"Extraction {num_skills} competences")
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = np.array(vectorizer.idf_)
    sorted_indices = np.argsort(idf_scores)[::-1]
    potential_skills = [feature_names[i] for i in sorted_indices[:num_skills]]
    return potential_skills

def filter_skills(potential_skills, exclude_words=None):
    # Liste de mots à exclure par defaut 
    if exclude_words is None:
        exclude_words = ["summary", "experience", "skills", "education", "work", "project", "responsibilities", "company", "university"]
    
    # Creation d'une nouvelle liste en gardant seulement les competences qui ne sont pas dans la liste d'exclusion, ont plus de 2 lettres, ne sont pas uniquement des chiffres
    filtered = [skill for skill in potential_skills 
                if skill not in exclude_words and len(skill) > 2 and not skill.isdigit()]
    return filtered

def group_skills_kmeans(skills, num_clusters=15):
    print(f"Regroupement en {num_clusters} clusters...")
    if not skills:
        print("Aucune compétence à regrouper.")
        return {}
        
    # vectoriser les competences elles-memes avec TF-IDF pour pouvoir calculer des distances
    vectorizer = TfidfVectorizer()
    skill_matrix = vectorizer.fit_transform(skills)
    
    # ajustement de nombre de clusters si on a moins de competences que demandé
    actual_clusters = min(num_clusters, skill_matrix.shape[0])
    if actual_clusters < 2:
        print("Pas assez de compétences pour le clustering.")
        return {0: skills}
        
    # Applique K-Means
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(skill_matrix)
    
    # Creation d'un dictionnaire pour stocker les résultats
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(skills[i])
    return cluster_dict

def find_skills_in_cv(cv_text, skills_list):
    # Vérifie si le texte est valide
    if not isinstance(cv_text, str):
        return []
    cv_lower = cv_text.lower()
    tokens = word_tokenize(cv_lower)
    found_skills = []
    # S'assurer que la liste de competences contient bien des chaines de caractères
    skills_list = [str(s) for s in skills_list if pd.notna(s)]
    # Parcourir chaque compétence de la liste à chercher
    for skill in skills_list:
        skill_lower = skill.lower()
        if " " in skill_lower:
            if skill_lower in cv_lower:
                found_skills.append(skill)
        elif skill_lower in tokens:
            found_skills.append(skill)
    return list(set(found_skills)) # Retourne la liste des competences trouvées


def analyze_skills_by_category(df, skills_list, text_col="cleaned_text"):
    print("Analyse par catégorie...")
    skills_by_category = {}
    if 'Category' not in df.columns:
        print("Erreur: Colonne 'Category' non trouvée.")
        return {}
    # Analyse des competences pour chaque catégorie unique de CV dans le DataFrame
    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]
        category_skills_counter = Counter()
        # Parcourt chaque CV de la catégorie
        for _, row in category_df.iterrows():
            cv_text = row[text_col] if pd.notna(row[text_col]) else ""
            skills_found = find_skills_in_cv(cv_text, skills_list)
            category_skills_counter.update(skills_found)

        # Recuperer les 10 competences les plus frequentes pour cette categorie
        top_skills = category_skills_counter.most_common(10)
        skills_by_category[category] = top_skills # Stocke le resultat
    return skills_by_category

def save_analysis_results(skills_list, clusters, skills_by_cat, output_path):
    print(f"Sauvegarde résultats: {output_path}")
    try:
        # Creation d'un DataFrame avec la liste des competences
        skills_df = pd.DataFrame({"skill": skills_list})
        skills_df["cluster"] = -1
        # Remplit la colonne cluster en utilisant le dictionnaire des clusters
        for cluster_id, sk_list in clusters.items():
            for sk in sk_list:
                skills_df.loc[skills_df["skill"] == sk, "cluster"] = cluster_id
        
        # Sauvegarder le DataFrame des competences/clusters en CSV
        skills_df.to_csv(output_path, index=False)
        
        # Sauvegarder l'analyse par categorie dans un fichier texte separé
        cat_output_path = output_path.replace(".csv", "_by_category.txt")
        with open(cat_output_path, "w", encoding='utf-8') as f: 
            for category, top_skills in skills_by_cat.items():
                f.write(f"Category: {category}\n")
                for skill, count in top_skills:
                    f.write(f"  - {skill}: {count}\n")
                f.write("\n")
        print("Sauvegarde OK.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats: {e}")

# Fonction principale du script
def run_analysis():
    processed_csv = os.path.join(script, "cv_pretraites.csv")
    output_csv = os.path.join(script, "competences.csv")
    
    df_processed = load_processed_data(processed_csv)
    if df_processed is None:
        return
        
    # Verification de la presence de la colonne de texte et gestion des valeurs manquantes
    if 'cleaned_text' not in df_processed.columns:
        print("Erreur: Colonne 'cleaned_text' manquante dans le fichier pretraité.")
        return
    df_processed['cleaned_text'] = df_processed['cleaned_text'].fillna('')

    # Vectoriser les textes, extraire et filtrer les competences
    tfidf_matrix, vectorizer = vectorize_tfidf(df_processed["cleaned_text"])
    potential_skills = extract_skills_tfidf(vectorizer, num_skills=150)
    skills = filter_skills(potential_skills)
    print(f"Competences filtrées: {len(skills)}")

    # Regrouper et analyser les competences par categorie, et suavegarde des resultas
    skill_clusters = group_skills_kmeans(skills, num_clusters=10)
    skills_per_category = analyze_skills_by_category(df_processed, skills)
    save_analysis_results(skills, skill_clusters, skills_per_category, output_csv)
    print(f"\nAnalyse terminée. Résultats: {output_csv}")

if __name__ == "__main__":
    run_analysis()
