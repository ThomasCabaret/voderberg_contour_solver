# Solveur symbolique de contours de type Voderberg

Ce projet étudie des contours orientés de la forme `(P0) A (P1) B` recouverts
par deux copies congruentes. Les coupures sont introduites uniquement par les
points projetés ou par la résolution des équations de mots.

## Installation Windows

Python 3.11 ou plus récent est recommandé.

```bat
py -3 -m pip install -r requirements.txt
```

Z3, NumPy et SciPy sont installés comme paquets Python ordinaires. Docker, WSL
et exécutable externe ne sont pas nécessaires.

## Démarrage rapide

```bat
run_tests.cmd
run_audit.cmd
run_web.cmd
run_geometry.cmd
```

Les scripts Windows attendent une action utilisateur avant de se fermer.

## Audit séquentiel

L'audit affiche une boucle distincte pour chaque opération indépendante :

1. génération des placements et parité des contacts;
2. caractérisation structurelle de chaque système d'équations de mots;
3. résolution bornée des équations de mots, avec diagnostic de troncature;
4. résolution des classes d'angles ponctuels;
5. tour total du contour prototype;
6. contraintes aux deux pôles;
7. obstruction translationnelle du prototype;
8. diagnostic optionnel des quatre parités;
9. construction des contours intérieur et extérieur;
10. résolution conjointe exacte des rotations;
11. obstructions translationnelles conjointes élémentaires;
12. coïncidences forcées de points;
13. Z3/NLSAT, uniquement sur les survivants de toutes les étapes précédentes;
14. écriture des rapports.

L'audit écrit :

```text
geometric_filter_audit.json       résumé et compteurs
formal_equation_audit.json        structure et recherche bornée des 2816 systèmes
geometric_filter_profiles.json    tous les profils terminaux
geometric_filter_survivors.json   uniquement les candidats finaux
```

Les deux gros exports de profils sont écrits en JSON compact et en flux : le
programme ne construit plus une chaîne JSON géante avec `json.dumps` avant de
l'écrire. Le fichier des survivants est enregistré avant le fichier détaillé,
de sorte qu'une panne pendant l'export complet ne prive pas le solveur
géométrique de son entrée.

Pour les audits très larges, on peut ne pas construire ni écrire l'export de
tous les profils, tout en conservant le résumé, l'audit formel et les
survivants :

```bat
py -3 project_cli.py audit --no-detailed-profiles-output
```

`--no-profiles-output` conserve son sens plus radical : il désactive les deux
exports de profils, y compris celui des survivants.

`formal_equation_audit.json` contient, pour chaque placement :

- les équations initiales;
- le nombre de variables et d'occurrences;
- la multiplicité maximale d'une variable;
- la classification linéaire, quadratique, cubique ou supérieure;
- la présence de l'involution;
- les composantes du graphe d'interaction des variables;
- la méthode complète recommandée;
- le nombre d'états visités par la recherche bornée;
- les limites de profondeur ou d'états atteintes;
- les profils terminaux trouvés dans les bornes.

Le résumé global donne les histogrammes correspondants. La classification
`quadratic` signifie que chaque variable apparaît au plus deux fois dans tout le
système initial; c'est la première classe à traiter avec un solveur de graphe de
Nielsen complet.

L'audit distingue maintenant cinq résultats de recherche formelle :

- `initially_inconsistent`;
- `exhausted_with_terminal_profiles`;
- `truncated_with_terminal_profiles`;
- `exhausted_without_terminal_profiles`;
- `truncated_without_terminal_profiles`.

Cette distinction est essentielle : une recherche tronquée peut déjà avoir
trouvé des profils tout en en oubliant d'autres. Le compteur
`terminal_profile_sets_potentially_incomplete_due_to_bounds_case_count` mesure
ce cas. Le compteur
`solution_existence_unresolved_due_to_bounds_case_count` mesure les systèmes
tronqués pour lesquels aucun terminal n'a encore été trouvé.

Les bornes par défaut sont dans `settings.py` :

```python
DEFAULT_AUDIT_MAX_DEPTH = 5
DEFAULT_AUDIT_MAX_STATES = 100
```

Elles peuvent être remplacées sans modifier le fichier :

```bat
py -3 project_cli.py audit --max-depth 10 --max-states 1000 --skip-z3
py -3 project_cli.py audit --max-depth 20 --max-states 10000 --skip-z3
```

La valeur `0` retire la borne correspondante, mais une recherche sans borne peut
ne jamais terminer sur un système cyclique :

```bat
py -3 project_cli.py audit --max-depth 0 --max-states 10000 --skip-z3
```

Z3 est exécuté par défaut. Pour l'omettre :

```bat
py -3 project_cli.py audit --skip-z3
```

Pour allonger ou raccourcir sa recherche :

```bat
py -3 project_cli.py audit --z3-timeout-ms 120000
```

Un résultat `unsat` est un rejet exact du système polynomial encodé. Un
`sat_candidate`, un `timeout` ou `unknown` reste dans le fichier des survivants.

## Recherche géométrique séparée

Le programme lit `geometric_filter_survivors.json` et représente chaque
variable formelle par une polyligne comportant un nombre configurable de points
intermédiaires.

```bat
run_geometry.cmd
```

Le paramètre principal est :

```bat
py -3 project_cli.py geometry --intermediate-points 0
py -3 project_cli.py geometry --intermediate-points 1
py -3 project_cli.py geometry --intermediate-points 3
```

- `0` : chaque variable est un segment droit;
- `1` : chaque variable possède deux arêtes;
- `n` : chaque variable possède `n + 1` arêtes.

Les occurrences directes et inverses réutilisent la même polyligne. Pour au
moins deux arêtes, les virages internes libres sont paramétrés de façon que leur
somme soit exactement la rotation totale `Kappa` imposée à la variable. Avec
zéro point intermédiaire, toutes les rotations de courbe sont nécessairement
nulles, ce qui est cohérent avec un segment droit.

La recherche utilise une optimisation globale heuristique. Les solutions
trouvées sont écrites dans `geometric_candidates.json`, puis affichées dans une
fenêtre avec les boutons précédent/suivant.

Dans le dessin :

- une couleur correspond à une variable formelle `V0`, `V1`, etc.;
- un trait continu est une occurrence directe;
- un trait pointillé est une occurrence inverse;
- chaque occurrence est étiquetée par son token formel.

Réglages pratiques :

```bat
py -3 project_cli.py geometry --attempts 5 --intermediate-points 2
py -3 project_cli.py geometry --max-profiles 10
py -3 project_cli.py geometry --view-only
py -3 project_cli.py geometry --no-gui
```

Par défaut, `--max-profiles` vaut `0`, ce qui signifie que **tous** les
survivants sont essayés. Une valeur positive sert uniquement à limiter la
recherche. Le lecteur parcourt le tableau JSON progressivement et ne conserve
que la représentation légère nécessaire à l'optimisation, plutôt que le gros
enregistrement détaillé de chaque profil.

Une solution trouvée est un certificat concret **dans le modèle polygonal
configuré**. Un échec de recherche ne prouve rien. Ce programme ne vérifie pas
encore la disjonction complète des trois copies.

## Visualiseur des résultats formels

```bat
run_web.cmd
```

Puis ouvrir `http://127.0.0.1:8765/`.

Le visualiseur ouvre désormais `geometric_filter_survivors.json` par défaut,
et non l'export de tous les profils rejetés. Le serveur envoie le fichier par
blocs sans en fabriquer une seconde copie en mémoire. Le navigateur ne crée
que la page courante du tableau (200 lignes par défaut), avec navigation et
choix du nombre de lignes par page.

## Profil formel

Une solution contient les courbes et les classes d'angles, par exemple :

```text
(P0 = a0) V0 (-a1) V1 (a2 = 0) V1^-1 (a1) V0^-1 (P1 = a3) V0 (-a1) V1 (a2 = 0) V1^-1
```

## Fichiers principaux

```text
settings.py                        réglages et noms canoniques
project_cli.py                     point d'entrée unique
audit_geometric_filters.py         audit séquentiel et rapports
formal_equation_audit.py           caractérisation des équations de mots
external_boundary_constraints.py   système intérieur/extérieur
forced_point_coincidence.py        coïncidences forcées
joint_translation_z3.py            filtre polynomial Z3/NLSAT
geometry_search_viewer.py          recherche polygonale et fenêtre interactive
results_web.py                     visualiseur web des profils formels
```
