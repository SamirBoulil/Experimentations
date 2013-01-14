pred_experimentations
=====================

#Visualiserl es résultats
Dans le dossier "resultats" on trouvera l'ensemble des matrices telles que :
### Un calcul de mesures
|           | clust Predit               | ...
|ref Clust  |(precision, rappel, fmesure)| ...
| ...       |                            | ...

### Un détail des correspondances
REFERENCE CLUSTER  I <-> CLUSTER N°J <-> FMESURE
detail termes cluster reference I
detail termes cluster J

### Détails de tous les clusters
Liste des termes des clusters de référence
liste des termes des clusters calculés de la CAH


#Restitution des résultats
Les résultats de toutes les comparaisons sont copiées dans le dossier x\_distancefile\_experimentations
Les optimums pour chaque experimentations sont copiés dans le dossier résultats.


#Dépendances

Il y a des dépendances à installer avec ce programme python par exemple : Orange pour les groupes de CAH et matplotlib pour les graphes de mesures.

