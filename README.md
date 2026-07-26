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
3. résolution bornée des équations de mots, avec diagnostics séparés de profondeur, d'états et de plafond cyclique;
4. résolution des classes d'angles ponctuels;
5. canonicalisation optionnelle des solutions décorées, mappings des deux copies inclus;
6. tour total du contour prototype;
7. contraintes aux deux pôles;
8. obstruction translationnelle du prototype;
9. diagnostic optionnel des quatre parités;
10. construction des contours intérieur et extérieur;
11. deux blocs linéaires rationnels configurables pour les angles et les périmètres des contours intérieur/extérieur;
12. obstructions translationnelles conjointes élémentaires;
13. coïncidences forcées sur les contours intérieur et extérieur;
14. placement symbolique des trois copies dans un repère commun et rejet des auto-coïncidences/recouvrements forcés;
15. couches Z3/NLSAT configurables : fermeture par cordes, cordes-longueurs, puis aires signées avec `A_exterieur = 3 A_interieur`;
16. écriture des rapports.

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

### Plafond temporaire des cycles formels

Une troisième limite, indépendante de la profondeur et du nombre d'états,
contrôle les échos dus au déroulement répété du même système résiduel :

```python
DEFAULT_FORMAL_MAX_CYCLE_UNROLLS = 3
```

Avec la valeur `3`, une branche peut revenir trois fois sur le même système
résiduel. Le retour suivant est coupé. Cette politique ne reconnaît pas encore
une famille paramétrique `U^n`; elle limite seulement son déroulement fini.
Chaque coupure est donc enregistrée comme une **troncature**, même si la branche
a déjà produit des profils.

Pour changer le plafond :

```bat
py -3 project_cli.py audit --max-depth 20 --max-states 10000 --max-cycle-unrolls 3 --skip-z3
```

Pour désactiver uniquement cette feature et retrouver la recherche bornée par
profondeur/états seule :

```bat
py -3 project_cli.py audit --max-cycle-unrolls 0 --skip-z3
```

Les compteurs spécifiques sont notamment :

```text
bounded_search_cycle_capped_case_count
bounded_search_cycle_capped_with_terminal_profiles_case_count
bounded_search_cycle_capped_without_terminal_profiles_case_count
bounded_search_cycle_pruned_state_count
```

Le plafond cyclique est volontairement isolé dans `formal_cycle_cap.py`. Il ne
modifie ni les règles de Nielsen/Levi, ni la canonicalisation, ni les filtres
géométriques.

### Clé canonique de solution décorée

Après la résolution des angles, l'audit calcule par défaut une clé pour la
**solution complète** : contour terminal et appariements des deux copies. La clé
inclut les positions terminales des segments appariés, leur sens de parcours et
la parité directe/réfléchie de chaque copie.

Elle identifie comme équivalents :

- l'échange de `P0` et `P1`;
- le renommage et la réorientation des variables de courbe;
- le renommage signé des classes d'angles;
- la permutation des deux copies identiques;
- le miroir global de toute la configuration.

Elle ne fusionne pas un changement de parité d'une seule copie ni un autre
appariement de segments. Elle ne reconnaît pas encore les familles
paramétriques issues de cycles. Aucun profil n'est supprimé : chaque
registrement reçoit seulement `solution_equivalence.key`, la taille de sa
classe observée et l'identifiant d'un représentant.

Le résumé contient :

```text
decorated_solution_canonicalization_summary
```

Pour désactiver cette étape sans toucher au solveur formel :

```bat
py -3 project_cli.py audit --skip-solution-canonicalization
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
- `1` : au plus deux arêtes par variable;
- `n` : au plus `n + 1` arêtes par variable.

Le nombre demandé est maintenant un **maximum**. Le moteur lit le
`terminal_mapping` produit par l'audit et propage exactement, variable par
variable, les transformations imposées par les deux contacts : inversion du
sens, isométrie directe ou réflexion. Les longueurs et tous les virages internes
des polylignes appariées sont donc couplés, et non plus seulement leur rotation
totale `Kappa`.

Les symétries réduisent automatiquement les degrés de liberté :

- une symétrie centrale conserve un template non trivial mais impose une forme
  palindromique/anti-palindromique;
- une réflexion échangeant les extrémités conserve un template miroir non
  trivial;
- une réflexion qui apparie une variable avec elle-même dans le même sens force
  une courbe droite : les points intermédiaires artificiels sont alors supprimés
  et la variable est représentée par une seule arête.

Le JSON des candidats contient le plan de réduction, les transformations entre
variables et le nombre effectif de points intermédiaires de chaque variable.
Un ancien fichier de survivants dépourvu de `terminal_mapping` doit être
régénéré avec la commande `audit` avant d'utiliser le solveur corrigé.

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

REM Recherche plus approfondie sans changer le nombre de sommets du modèle
py -3 project_cli.py geometry --intermediate-points 1 --attempts 16 --max-iterations 1000 --population-size 24 --seed 271828

REM Ancien modèle non contraint, uniquement pour comparaison/debug
py -3 project_cli.py geometry --skip-contact-template-constraints
```

Pour retrouver une quatrième occurrence symétrique déjà réalisable avec le
modèle courant, il vaut mieux augmenter d'abord `--attempts`,
`--max-iterations` et `--population-size` sans augmenter
`--intermediate-points`. Ajouter des points change le modèle et accroît fortement
la dimension de l'optimisation. Un changement de ces paramètres ou de `--seed`
crée automatiquement une nouvelle identité de run. `--fresh` n'est nécessaire
que pour effacer et rejouer exactement la même configuration.

Par défaut, `--max-profiles` vaut `0`, ce qui signifie que **tous** les
survivants sont essayés. Une valeur positive sert uniquement à limiter la
recherche. Le lecteur parcourt le tableau JSON progressivement et ne conserve
que la représentation légère nécessaire à l'optimisation, plutôt que le gros
enregistrement détaillé de chaque profil.

### Reprise transactionnelle

La recherche géométrique utilise par défaut une base SQLite :

```text
geometry_search_checkpoint.sqlite3
```

Après chaque profil terminé, elle enregistre dans une transaction durable :

- `found` avec le candidat complet;
- `no_candidate`;
- `error` avec le message d'erreur.

La même commande reprend automatiquement exactement le même run :

```bat
py -3 project_cli.py geometry --intermediate-points 3
```

L'identité du run comprend le contenu SHA-256 du fichier de survivants, la
liste ordonnée des profils sélectionnés et tous les réglages numériques. Un
changement de `--intermediate-points`, `--attempts`, `--max-iterations`,
`--population-size`, `--seed`, `--max-profiles`, du mode de contraintes de
templates ou du fichier d'entrée crée un
run séparé dans la même base; les résultats incompatibles ne sont jamais
mélangés.

En cas de `Ctrl+C`, tous les profils précédemment terminés restent acquis et le
profil interrompu sera repris. `geometric_candidates.json` est aussi réécrit
avec les candidats déjà enregistrés.

Commandes utiles :

```bat
py -3 project_cli.py geometry --intermediate-points 3 --fresh
py -3 project_cli.py geometry --intermediate-points 3 --retry-errors
py -3 project_cli.py geometry --checkpoint autre_checkpoint.sqlite3
py -3 project_cli.py geometry --no-resume
```

- `--fresh` efface uniquement le run correspondant exactement à la configuration courante;
- `--retry-errors` rejoue les profils enregistrés en erreur;
- `--no-resume` désactive totalement SQLite et rétablit le comportement one-shot.

Le mécanisme de reprise est isolé dans `geometry_checkpoint.py`; il ne connaît
pas les mathématiques de l'optimiseur.

Une solution trouvée est un certificat concret **dans le modèle polygonal
configuré**, avec congruence locale exacte des courbes appariées sous les
isométries directes/réfléchies indiquées. Un échec de recherche ne prouve rien.
Le pipeline formel place maintenant les trois copies dans un repère commun et
rejette certaines coïncidences ou superpositions forcées. Il ne certifie
toujours pas toutes les intersections possibles entre arcs courbes, la
disjonction complète des trois intérieurs ni l'absence de tout contact parasite.

## Filtres géométriques conjoints ajoutés

`global_linear_contour_filter.py` est le filtre linéaire global actif. Il
travaille uniquement sur les deux contours formels fermés déjà construits : le
contour de la pièce et le contour extérieur des trois copies. Les interfaces
internes ne sont pas résolues une seconde fois, car leur compatibilité appartient
au solveur formel des contacts.

Le filtre résout deux blocs rationnels exacts et indépendants :

- un bloc angulaire réunissant les deux équations de tour, les deux contraintes
  de pôles, les bornes des classes d'angles et les bornes principales de chaque
  angle réellement rencontré sur les deux contours, y compris les angles
  composites aux pôles extérieurs;
- un bloc de longueurs géométriques positives, avec normalisation du périmètre
  de la pièce et égalité du périmètre extérieur.

Chaque bloc maximise une marge stricte rationnelle. Le profil est rejeté dès
qu'un bloc n'admet aucune marge positive. Les anciens filtres d'angle rapides
restent en amont comme pré-filtres bon marché.

Les variables scalaires d'aire ne sont volontairement pas ajoutées à ce LP :
sans cordes ni déterminants, `A_exterieur = 3*A_interieur` et les bornes
isopérimétriques restent toujours satisfaisables en choisissant une aire
positive arbitrairement petite.

Le module autonome `global_metric_contour_model.py` construit maintenant la
couche suivante sans dépendre d'un solveur. Pour chaque variable de courbe, il
répertorie :

- une longueur d'arc positive `L[X]`;
- une corde locale `D[X]`;
- une aire signée d'arc `S[X]`;
- la phase, la conjugaison et le signe exact de chaque occurrence dans les
  contours intérieur et extérieur.

`joint_translation_z3.py` compile ce modèle en couches polynomiales imbriquées.
La couche cordes-longueurs impose :

```text
L[X] > 0
perimetre_interieur = perimetre_exterieur = 1
norme(D[X]) <= L[X]
```

La couche d'aire ajoute la loi exacte de concaténation de degré deux, la
positivité de l'aire intérieure et :

```text
A_exterieur = 3 A_interieur
```

Les bornes rationnelles utilisent seulement `pi > 3` :
`|S[X]| <= L[X]^2/3` et `A_interieur <= 1/36`. Elles sont volontairement plus
faibles que les bornes avec `pi`, mais restent exactes pour le rejet. Les sommes
d'aire sont encodées par des accumulateurs auxiliaires afin que la taille SMT
croisse linéairement avec le nombre d'occurrences.

Configuration des couches :

```bat
rem Désactiver seulement le bloc angulaire linéaire
py -3 project_cli.py audit --skip-global-angle-filter

rem Désactiver seulement le bloc de périmètres linéaire
py -3 project_cli.py audit --skip-global-length-filter

rem Garder cordes-longueurs mais retirer l'aire
py -3 project_cli.py audit --skip-signed-area-layer

rem Revenir au modèle polynomial historique de fermeture seule
py -3 project_cli.py audit --skip-chord-length-layer
```

La couche d'aire dépend de la couche cordes-longueurs; désactiver cette dernière
désactive automatiquement l'aire. `--skip-z3` conserve les rapports et la
construction des problèmes, mais n'exécute aucun rejet polynomial.

`placed_copy_geometry.py` applique une seule isométrie globale à chaque copie et
exprime tous ses points distingués dans le même repère que la pièce de
référence. Le filtre symbolique ne rejette que les identités sûres :
auto-coïncidence d'une copie et recouvrement exact d'arcs avec les intérieurs du
même côté. Les équations ponctuelles complètes sont ensuite ajoutées au modèle
Z3. Une réponse `SAT` reste seulement un candidat; les intersections génériques
d'arcs courbes ne sont pas toutes encodées.

Le suivi des pistes mathématiques et des limites se trouve dans
`FILTER_RESEARCH_MEMO.md`.

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
parametric_graph.py                 graphe résiduel cyclique Nielsen/Levi
parametric_expressions.py           AST des familles finies et à puissances
exact_partial_word_solver.py        compilation exacte du fragment pris en charge
family_representative_expansion.py  spécialisation aval des exposants
formal_cycle_cap.py                 plafond de l'ancien mode borné
solution_canonicalization.py        clé du contour et des deux mappings
voderberg_type_classifier.py        classification formelle type 1 / type 2
geometry_checkpoint.py              reprise transactionnelle SQLite
positive_length_filter.py           rejet exact des contradictions de longueurs de mots
curve_relation_algebra.py           relations formelles inverse/miroir
curve_term_solver.py                valeurs Straight/Mirror/Inverse
curve_template_constraints.py       compilateur polygonal des termes de courbes
external_boundary_constraints.py   système intérieur/extérieur
rational_linear_program.py         simplexe exact sur les rationnels
global_linear_contour_filter.py     filtre LP exact des deux contours décorés
forced_point_coincidence.py        coïncidences forcées sur une frontière
placed_copy_geometry.py            trois copies dans un repère symbolique commun
joint_translation_z3.py            filtre polynomial et isométries globales
geometry_search_viewer.py          recherche polygonale et fenêtre interactive
FILTER_RESEARCH_MEMO.md             suivi des filtres faits et restant à étudier
results_web.py                     visualiseur web des profils formels
```

## Classification formelle Voderberg type 1 / type 2

Chaque placement reçoit maintenant une classification combinatoire indépendante,
stockée dans `voderberg_type`. Elle ne prouve pas la réalisabilité géométrique :
elle reconnaît uniquement les deux schémas grossiers d'arcs et de parités.

- `type1` : un contact principal direct s'apparie au même arc parcouru en sens
  inverse; l'autre contact est l'image directe d'un sous-arc propre non vide du
  contact principal.
- `type2` : un contact principal réfléchi s'apparie à un arc dont l'intérieur
  traverse exactement un des deux pôles; l'autre contact est l'image directe
  d'un sous-arc propre non vide du contact principal.

Le sélecteur agit **après la construction des profils formels terminaux**, mais
avant les angles, holonomies, frontières, coïncidences et Z3 :

```bat
REM Aucun filtrage; tous les profils continuent dans le pipeline
py -3 project_cli.py audit --voderberg-types all --skip-z3

REM Seulement les profils compatibles type 1
py -3 project_cli.py audit --voderberg-types type1 --skip-z3

REM Seulement les profils compatibles type 2
py -3 project_cli.py audit --voderberg-types type2 --skip-z3

REM Seulement l'union des types 1 et 2, en excluant les autres profils
py -3 project_cli.py audit --voderberg-types type1+type2 --skip-z3
```

La différence entre `all` et `type1+type2` est importante : `all` conserve aussi
les profils qui ne correspondent à aucun des deux types.

Le résumé JSON contient notamment :

```text
type1_compatible_profile_count
type2_compatible_profile_count
compatible_with_both_profile_count
compatible_with_neither_profile_count
terminal_profile_count_before_type_selection
terminal_profile_count_selected_for_downstream_pipeline
type1_compatible_placement_case_count_generated
type2_compatible_placement_case_count_generated
```

Avec les bornes historiques `max-depth=5`, `max-states=100` et le plafond
cyclique à 3, l'audit retrouve 26 profils terminaux compatibles type 1 et 58
compatibles type 2. Ces nombres dépendent des bornes formelles.

Le solveur géométrique peut également filtrer un fichier de survivants déjà
généré :

```bat
py -3 project_cli.py geometry --voderberg-types type2 --intermediate-points 3
```

Un fichier ancien dépourvu du champ `voderberg_type` reste lisible avec
`--voderberg-types all`, mais doit être régénéré pour une sélection `type1` ou
`type2`. Le sélecteur fait partie de l'identité du checkpoint SQLite, donc deux
recherches géométriques de types différents ne partagent pas leurs résultats.

## Filtre exact de longueurs de mots

Avant la recherche de Nielsen/Levi, chaque variable formelle reçoit une
longueur entière strictement positive. Les deux équations de mots imposent deux
égalités linéaires sur ces longueurs. Si elles sont incompatibles, le système
est rejeté immédiatement et n'entre pas dans le solveur borné.

Le filtre est activé par défaut. Pour reproduire l'ancien comportement :

```bat
py -3 project_cli.py audit --skip-positive-length-filter --skip-z3
```

L'analyse reste écrite dans `formal_equation_audit.json`, même lorsqu'on
désactive son pouvoir de rejet. Sur les 2 816 placements actuels, elle détecte
2 184 contradictions de longueurs : 544 étaient déjà contradictoires après la
simplification initiale et 1 640 sont des rejets supplémentaires. Il reste donc
632 systèmes envoyés au solveur formel à branchements.

## Solveur formel exact partiel et familles à puissances

Le mode formel par défaut est désormais `exact-partial`. Il ne déroule plus un
cycle jusqu'à une profondeur arbitraire pour présenter ses premiers passages
comme des solutions indépendantes.

Pour chaque système qui passe le filtre de longueurs :

1. le programme construit le graphe résiduel de transformations de
   Nielsen/Levi en identifiant les systèmes résiduels équivalents;
2. aucun verdict exact n'est émis si la construction atteint sa limite de
   nœuds ou d'arêtes;
3. lorsque le graphe est complet, les chemins finis et les cycles qui ajoutent
   seulement des contextes fixes sont compilés en expressions formelles;
4. plusieurs composantes successives peuvent produire des puissances
   imbriquées;
5. une composante morphique ou branchée hors du langage pris en charge est
   arrêtée à son entrée et conservée comme frontière dynamique non développée;
6. les autres branches finies ou paramétrées du même graphe restent conservées.

Les statuts sont :

```text
EXACT_UNSAT
EXACT_FINITE
EXACT_POWER
EXACT_NESTED_POWER
EXACT_SUPPORTED_FAMILIES_WITH_UNSUPPORTED_FRONTIER
EXACT_GRAPH_UNSUPPORTED_FAMILY_LANGUAGE
UNRESOLVED_GRAPH_LIMIT
UNRESOLVED_FAMILY_LIMIT
```

`EXACT_GRAPH_UNSUPPORTED_FAMILY_LANGUAGE` signifie que toutes les branches
utiles du cas passent par une dynamique plus générale que les concaténations,
inverses et puissances imbriquées actuellement prises en charge.
`EXACT_SUPPORTED_FAMILIES_WITH_UNSUPPORTED_FRONTIER` signifie que le même cas
contient à la fois des familles compilées et au moins une telle frontière
dynamique. `UNRESOLVED_GRAPH_LIMIT` signifie au contraire que le graphe lui-même
n'a pas été terminé : aucune conclusion d'exhaustivité n'est alors permise.

Commande habituelle :

```bat
py -3 project_cli.py audit --skip-z3
```

Budgets du graphe exact partiel :

```bat
py -3 project_cli.py audit ^
  --exact-graph-max-nodes 2000 ^
  --exact-graph-max-edges 12000 ^
  --exact-max-families 20000 ^
  --skip-z3
```

La valeur `0` retire une limite de graphe, mais un système hors fragment fini
peut alors ne jamais terminer. L'ancien énumérateur borné reste disponible pour
comparaison :

```bat
py -3 project_cli.py audit ^
  --formal-solver-mode legacy-bounded ^
  --max-depth 20 ^
  --max-states 10000 ^
  --skip-z3
```

Avec les limites par défaut sur les 2 816 systèmes actuels :

```text
2 184  EXACT_UNSAT, principalement par le filtre de longueurs positives
  214  EXACT_FINITE
   72  EXACT_POWER
   10  EXACT_NESTED_POWER
   22  EXACT_SUPPORTED_FAMILIES_WITH_UNSUPPORTED_FRONTIER
   20  EXACT_GRAPH_UNSUPPORTED_FAMILY_LANGUAGE
  294  UNRESOLVED_GRAPH_LIMIT
```

Les branches compilées représentent actuellement 2 692 familles formelles :
874 finies, 1 200 à puissance simple et 618 à puissances imbriquées. Les 48
frontières dynamiques non compilées sont enregistrées sans aucun déroulement.

### Politique explicite d'expansion des familles

Par défaut, seules les familles réellement finies sont converties en profils
pour les filtres géométriques. Les familles paramétrées restent uniquement dans
l'audit formel :

```bat
py -3 project_cli.py audit --family-expansion-policy none --skip-z3
```

Une spécialisation peut être demandée explicitement. Pour prendre la valeur
minimale de chaque exposant :

```bat
py -3 project_cli.py audit --family-expansion-policy minimum --skip-z3
```

Pour fixer tous les exposants à `2`, sous réserve de leurs minima :

```bat
py -3 project_cli.py audit ^
  --family-expansion-policy fixed ^
  --family-representative-exponent 2 ^
  --skip-z3
```

Pour développer toutes les combinaisons d'exposants entre leur minimum et `3`
inclus :

```bat
py -3 project_cli.py audit ^
  --family-expansion-policy range ^
  --family-expansion-max-exponent 3 ^
  --skip-z3
```

`--family-expansion-max-specializations` impose un garde-fou par famille. Les
frontières dynamiques hors langage puissance ne sont jamais développées par ces
options; elles restent classées pour une future politique dédiée.

## Valeurs formelles de courbes : Straight, Mirror et Inverse

Après le solveur générique de mots et après la construction du mapping
terminal, une couche spécialisée interprète les relations de courbes sans
modifier le solveur générique.

L'algèbre formelle indépendante est dans `curve_relation_algebra.py`. Elle ne
connaît ni coordonnées, ni points de polyligne, ni optimiseur. Elle manipule
uniquement :

```text
X
Inverse(X)
Mirror(X)
Mirror(Inverse(X))
Straight(lambda)
```

Les auto-relations sont normalisées ainsi :

```text
identité                       : X reste libre
inverse par isométrie directe  : déjà résolu par le solveur de mots
même sens sous réflexion       : X = Straight(lambda), lambda > 0
inverse sous réflexion         : X = Y (a) Mirror(Inverse(Y))
```

Les segments droits d'une même composante partagent la même classe de longueur
positive `lambdaN`. Une valeur `Straight(lambda)` n'est pas représentée par des
points intermédiaires alignés : le compilateur polygonal utilise directement
une arête.

`curve_term_solver.py` produit les termes formels. L'ancien
`curve_template_constraints.py` est conservé comme compilateur et vérificateur
numérique séparé; le programme vérifie que les deux couches donnent les mêmes
transformations avant de lancer l'optimiseur.

La spécialisation peut être désactivée indépendamment :

```bat
py -3 project_cli.py audit --skip-curve-term-specialization --skip-z3
```

## Canonical profile reduction and curve-substitution subsumption

The audit now performs two independent formal reductions before selecting the
survivor file used by the geometric search.

1. Decorated-solution canonicalization identifies exact duplicates under pole
   exchange, contour reversal/global mirror, permutation of the two copies,
   curve renaming/reorientation, and signed angle-class renaming. Only one
   representative of each class remains eligible for the survivor pipeline.
2. Contour-shape subsumption removes a canonical profile `Q` when a more general
   profile `P` can reproduce its decorated contour by a nonerasing substitution
   of every curve variable of `P` by a decorated path of `Q`. Existing angle
   classes of `P` must specialize consistently; newly introduced internal
   points and angles in `Q` are allowed. For soundness, the current pruning pass
   only uses general profiles whose curve-relation components are all formally
   free.

All removed profiles remain present in the detailed audit with a certificate:
`formal_profile_reduction` records the canonical representative or the explicit
curve/angle substitution. They are omitted from
`geometric_filter_survivors.json`.

Diagnostic switches:

```bash
python project_cli.py audit --keep-equivalent-profiles
python project_cli.py audit --skip-profile-subsumption
```

The canonical machine-readable contour and mappings are exported in
`solution_equivalence.canonical_json`.
