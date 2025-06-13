*   `1_pretraitement.py` : Il prend le gros fichier `Resume.csv`, nettoie le texte (enlève la ponctuation, les chiffres, met tout en minuscules), coupe en mots et enlève les mots inutiles (comme "le", "la", "de"...). Il essaie aussi de repérer des mots qui pourraient être des compétences. Il sauvegarde tout ça dans `cv_pretraites.csv`.
*   `2_vectorisation_classification.py` : Il lit le fichier `cv_pretraites.csv`. Il utilise TF-IDF pour transformer les textes en chiffres (vecteurs). Il essaie ensuite de regrouper les compétences qui se ressemblent (avec K-Means). Il sauvegarde la liste des compétences trouvées dans `competences.csv` et un résumé par catégorie dans `competences_by_category.txt`.
*   `3_validation.py` : Il prend les CV et les compétences trouvées pour regarder quelques exemples et voir ce que ça donne. Il fait aussi des graphiques (`competences_*.png`) pour montrer les compétences les plus fréquentes par catégorie. Il écrit un petit rapport dans `validation_report.txt`.
*   `main.py` : Il lance les 3 scripts du dessus dans le bon ordre. Il vérifie aussi si t'as les bonnes bibliothèques Python installées et essaie de les mettre si elles manquent.

## Comment lancer le code

1.  **Les fichiers au bon endroit** :
    *   Mets tous les fichiers `.py` (main.py, 1_pretraitement.py...) dans un dossier avec le fichier resume.csv, ce qui est déjà fait.


2.  **Lance le script principal** :
    *   Dans la fenêtre de commande (`cmd`), tapez :
        ```bash
        python main.py
        ```

3.  **Résultats** :
    *   Tous les fichiers créés (`cv_pretraites.csv`, `competences.csv`, `validation_report.txt`, les `.png`...) sont dans votre dossier, comme ça l'exécution sera terminée.


Je serais à votre disposition pour toutes clarifications.
Mohamed NAJID.