# Auteur: Mohamed NAJID

# Importation des bibliotheques
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # pour la similarité cosinus


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text) 
    text = re.sub(r"\d+", " ", text) 
    text = re.sub(r"\s+", " ", text).strip() 
    return text

def load_cv_data(csv_path):
    print(f"Chargement des CV prétraités: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Erreur: Fichier {csv_path} non trouvé.")
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        # S'assurer que la colonne de texte nettoyé existe et remplacer les NaN
        if 'cleaned_text' not in df.columns:
            print("Erreur: Colonne 'cleaned_text' manquante dans cv_pretraites.csv")
            return None
        # Verification du presence de la colonne ID
        if 'ID' not in df.columns:
            print("Erreur: Colonne 'ID' manquante dans cv_pretraites.csv")
            return None
            
        df['cleaned_text'] = df['cleaned_text'].fillna('')
        print(f"CV chargés: {len(df)}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement de {csv_path}: {e}")
        return None

def load_job_offer(file_path):
    print(f"Chargement de l'offre d'emploi: {file_path}")
    if not os.path.exists(file_path):
        print(f"Erreur: Fichier {file_path} non trouvé.")
        return None
    try:
        # Essaye plusieurs encodages 
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        content = None
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                print(f"Offre chargée avec encodage: {enc}")
                break 
            except UnicodeDecodeError:
                print(f"Échec lecture avec {enc}, essai suivant...")
            except Exception as read_err:
                print(f"Erreur de lecture inattendue avec {enc}: {read_err}")
        
        if content is None:
            print("Erreur: Impossible de lire le fichier offre_emploi.txt")
            return None
            
        return content
    except Exception as e:
        print(f"Erreur lors de l'ouverture de {file_path}: {e}")
        return None

# Fonction principale du matching 

def run_matching():
    # Determiner le dossier du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cv_file = os.path.join(script_dir, "cv_pretraites.csv")
    offer_file = os.path.join(script_dir, "offre_emploi.txt")
    output_file = os.path.join(script_dir, "ranking_cv.csv")

    # 1. Charger les données
    df_cv = load_cv_data(cv_file)
    job_offer_text = load_job_offer(offer_file)

    if df_cv is None or job_offer_text is None:
        print("Arrêt du matching car les données n'ont pas pu être chargées.")
        return

    # 2. Nettoyer le texte de l'offre d'emploi
    cleaned_offer_text = clean_text(job_offer_text)
    if not cleaned_offer_text:
        print("Erreur: Le texte de l'offre d'emploi est vide après nettoyage.")
        return

    # 3. Préparer les textes pour la vectorisation
    all_texts = df_cv['cleaned_text'].tolist() + [cleaned_offer_text]
    print(f"Nombre total de documents à vectoriser: {len(all_texts)}")

    # 4. Vectorisation TF-IDF
    print("Vectorisation TF-IDF de l'ensemble des textes (CV + offre)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        print(f"Matrice TF-IDF globale crée: {tfidf_matrix.shape}")
    except Exception as e:
        print(f"Erreur lors de la vectorisation TF-IDF: {e}")
        return

    # Separer la matrice TF-IDF : vecteurs des CV et vecteur de l'offre
    tfidf_cvs = tfidf_matrix[:-1] 
    tfidf_offer = tfidf_matrix[-1] 

    print(f"Vecteurs TF-IDF des CV: {tfidf_cvs.shape}")
    print(f"Vecteur TF-IDF de l'offre: {tfidf_offer.shape}")

    # 5. Calcul de la similarité cosinus
    print("Calcul de la similarité cosinus entre l'offre et chaque CV")
    try:
        cosine_similarities = cosine_similarity(tfidf_offer, tfidf_cvs).flatten()
        # .flatten() transforme la matrice resultat (1 x N) en un simple tableau (N)
        print(f"Nombre de scores de similarité calculés: {len(cosine_similarities)}")
        df_cv['similarity_score'] = cosine_similarities
        
    except Exception as e:
        print(f"Erreur lors du calcul de la similarité cosinus:")
        return

    # 6. Classement des CV
    print("Classement des CV par score de similarité décroissant")
    # Trie 
    df_ranked = df_cv.sort_values(by='similarity_score', ascending=False)
    
    # Sélectionner les colonnes pertinentes pour le fichier de sortie
    output_columns = ['ID', 'Category', 'similarity_score']
    # Vérifier si la colonne Category existe
    if 'Category' not in df_ranked.columns:
        print("Erreur: Colonne 'Category' non trouvée")
        output_columns = ['ID', 'similarity_score']
        
    df_output = df_ranked[output_columns]

    # Afficher les 5 CV les plus pertinents 
    print("\nTop 5 des CV les plus pertinents:")
    print(df_output.head())

    # 7. Sauvegarde des resultats
    print(f"Sauvegarde du classement dans: {output_file}")
    try:
        # Sauvegarde dans un fichier CSV
        df_output.to_csv(output_file, index=False, encoding='utf-8')
        print("Classement sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier {output_file}: {e}")
        return
        
    print("\nMATCHING ACHIEVED ")

if __name__ == "__main__":
    run_matching()
