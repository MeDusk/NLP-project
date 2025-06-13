Ce module compare automatiquement des CV avec une offre d'emploi et les classe par ordre de pertinence.

Fichiers Requis
Dossier projet/
─ cv_pretraites.csv    
─ offre_emploi.txt     
─ matching.py          

Fichier Généré
Dossier projet/
─ ranking_cv.csv       # Classement des CV (sortie)


Utilisation
(N.B: vous devez d'abord executer la premiere partie avec la commande "python main.py")
-> bash : python matching.py


Comment ça marche

Le système de matching CV fonctionne en quatre étapes principales : il charge d'abord les CV prétraités et l'offre d'emploi, puis nettoie le texte de l'offre pour le standardiser. Ensuite, il utilise la technique TF-IDF pour convertir tous les textes (CV + offre) en vecteurs numériques, où chaque mot du vocabulaire devient une dimension et sa valeur reflète son importance dans le document. Le système calcule alors la similarité cosinus entre le vecteur de l'offre et chaque vecteur de CV, générant un score entre 0 et 1 qui mesure leur proximité thématique. Enfin, il classe tous les CV par ordre décroissant de leur score de similarité et sauvegarde ce ranking dans un fichier CSV, permettant aux recruteurs d'identifier rapidement les candidats les plus pertinents pour le poste.

Résultat
Le fichier ranking_cv.csv contient les CV classés du plus pertinent au moins pertinent avec leur score de similarité.

Je serais à votre disposition pour toutes clarifications.
Mohamed NAJID.