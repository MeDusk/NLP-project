# Auteur: Mohamed NAJID

# Importation des bibliothèques nécessaires
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os

script = os.path.dirname(os.path.abspath(__file__))

# Téléchargement des ressources NLTK
print("Téléchargement NLTK")
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)
    nltk.download("punkt_tab", quiet=True) 
    print("Téléchargement NLTK OK.")
except Exception as e:
    # Affiche une erreur si le téléchargement echoué
    print(f"Erreur lors du téléchargement des ressources NLTK: {e}")
  
# Fonctions de traitement
def load_data(csv_path):
    print(f"Chargement: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Erreur: Le fichier {csv_path} n'a pas été trouvé.")
        print("Assurez-vous que Resume.csv est dans le même dossier que les scripts.")
        return None
    try:
        df = pd.read_csv(csv_path, encoding='latin-1') 
        print(f"CV chargés: {len(df)}")
        return df
    except UnicodeDecodeError:
        print("Erreur d'encodage avec latin-1. Essai avec utf-8...")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"CV chargés: {len(df)}")
            return df
        except Exception as e:
            print(f"Erreur lors du chargement de {csv_path} avec utf-8 également: {e}")
            return None
    except Exception as e:
        print(f"Erreur lors du chargement de {csv_path}: {e}")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text # Retourner le texte nettoyé

def tokenize_text(text, lang="english"):
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words(lang))
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return filtered_tokens
    except Exception as e:
        # Gestion des erreurs de NLTK
        print(f"Erreur de tokenisation NLTK: {e}")
        return [] # Retourne une liste vide en cas d'erreur

def extract_entities(text):
    try:
        # Decoupage en mots et effectue l'etiquetage grammatical
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
    except Exception as e:
        print(f"Erreur NLTK (POS tagging): {e}")
        return []
        
    entities = []
    # Parcourir les mots et leurs étiquettes grammaticales
    for word, tag in pos_tags:
        if tag.startswith("NN") and len(word) > 2:
            entities.append(word)
            

    # definition des expressions regulières pour trouver des competences techniques courantes
    skill_patterns = [
        r"\b[a-z]+\+\+\b", 
        r"\b[a-z]+#\b",
        r"\b[a-z]+\.[a-z]+\b",
        r"\b[a-z]+-[a-z]+\b",
        r"\bpython\b", r"\bjava\b", r"\bsql\b", r"\bexcel\b"
    ]
    
    text_lower = text.lower()
    # Appliquer chaque motif pour trouver les compétences
    for pattern in skill_patterns:
        try:
            found = re.findall(pattern, text_lower)
            entities.extend(found)
        except Exception as e:
            print(f"Erreur Regex sur motif '{pattern}': {e}")

    # Supprimer les doublons et trie la liste
    entities = list(set(entities))
    entities.sort()
    return entities

def preprocess_resumes(df, text_col="Resume_str"):
    print("Prétraitement")
    df_processed = df.copy()# Creation d'une copie pour ne pas modifier l'original
    # Verification si la colonne contenant le texte des CV existe
    if text_col not in df.columns:
        print(f"Erreur: Colonne '{text_col}' non trouvée dans le CSV.")
        return None
        
    df_processed["cleaned_text"] = df_processed[text_col].apply(clean_text)
    df_processed["tokens"] = df_processed["cleaned_text"].apply(tokenize_text)
    df_processed["entities"] = df_processed["cleaned_text"].apply(extract_entities)
    print("Prétraitement OK.")
    return df_processed # Return le DataFrame prétraité

def save_processed_data(df, output_path):
    print(f"Sauvegarde: {output_path}")
    df_save = df.copy()
    # Les listes doivent etre converties en chaines pour être sauvegardées en CSV
    if 'tokens' in df_save.columns:
        df_save["tokens"] = df_save["tokens"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    if 'entities' in df_save.columns:
        df_save["entities"] = df_save["entities"].apply(lambda x: "|".join(x) if isinstance(x, list) else "")
    try:
        df_save.to_csv(output_path, index=False, encoding='utf-8') 
        print("Sauvegarde OK.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de {output_path}: {e}")

# Fonction principale du script
def run_preprocessing():
    csv_file = os.path.join(script, "Resume.csv") 
    output_file = os.path.join(script, "cv_pretraites.csv")
    
    data_df = load_data(csv_file)
    if data_df is None:
        print("arret car le chargement des donnees a echoue.")
        return
        
    print("\nDonnees brutes (aperçu):")
    print(data_df.head(2))
    
    processed_df = preprocess_resumes(data_df)
    if processed_df is None:
        print("arret car le pretraitement a echoue.")
        return
        
    # Affichage d'un aperçu des données pretraitées
    print("\nDonnees pretraitées (apercu):")
    cols_to_show = [col for col in ["ID", "Category", "cleaned_text"] if col in processed_df.columns]
    if cols_to_show:
        print(processed_df[cols_to_show].head(2))
    
    save_processed_data(processed_df, output_file)
    print(f"\nResultat sauvegardé: {output_file}")

if __name__ == "__main__":
    run_preprocessing()