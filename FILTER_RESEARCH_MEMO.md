# Mémo des pistes de filtrage géométrique

Ce fichier suit les pistes discutées, leur statut d'implémentation et les limites
qui restent ouvertes. Il doit être mis à jour avec les futurs correctifs.

## Implémenté

### A. Filtre linéaire global des deux contours

Statut : **implémenté et actif**.

Le module `global_linear_contour_filter.py` travaille sur le contour décoré de
la pièce et sur le contour décoré extérieur déjà construits. Il ne réencode pas
les interfaces internes, qui sont le contenu du solveur formel des contacts.

Il résout exactement deux blocs rationnels indépendants.

Bloc angulaire :

- tour total du contour intérieur;
- tour total du contour extérieur;
- bornes strictes de toutes les classes d'angles;
- deux contraintes de secteurs aux pôles `P0` et `P1`;
- bornes strictes de chaque angle physique réellement rencontré sur les deux
  contours, notamment les formes composites aux pôles extérieurs.

Bloc de longueurs géométriques :

- une longueur strictement positive par variable de courbe;
- invariance sous inversion et réflexion, obtenue par l'emploi de la même
  variable de longueur;
- normalisation du périmètre intérieur;
- égalité du périmètre extérieur au périmètre intérieur.

Chaque bloc maximise une marge rationnelle. Le profil n'est conservé que si les
deux marges sont strictement positives. Les blocs restent séparés parce qu'ils
ne partagent encore aucune variable; leur conjonction est exactement un LP
bloc-diagonal.

Les aires scalaires abstraites ne sont pas ajoutées : avant l'introduction des
cordes et du terme déterminant, leur positivité, `A_ext = 3 A_int` et les bornes
isopérimétriques sont une relaxation vide qui ne peut rejeter aucun profil.

### B. Isométrie globale explicite de chaque copie

Statut : **implémenté dans le modèle commun et dans l'encodage Z3**.

Le module `placed_copy_geometry.py` construit dans un même repère :

- la pièce de référence;
- la copie attachée au contact `A`;
- la copie attachée au contact `B`.

Chaque copie est régie par une seule isométrie directe ou réfléchie, déterminée
par l'ancre et l'orientation du contact. La même transformation est appliquée à
tous les points distingués et à tous les arcs de la copie. Les équations
ponctuelles correspondantes sont transmises à `joint_translation_z3.py`.

### C. Coïncidences et recouvrements forcés dans le repère commun

Statut : **implémenté de manière exacte mais volontairement incomplète**.

Le filtre rejette actuellement :

- une identité symbolique forçant deux points distincts d'une même copie à
  coïncider;
- un recouvrement exact de deux occurrences d'arc dont les intérieurs sont
  forcés du même côté.

Un contact ponctuel supplémentaire entre deux copies différentes n'est pas
rejeté à lui seul : il peut s'agir d'une tangence physiquement admissible.
Une intersection générique entre deux arcs courbes n'est pas détectable à ce
niveau sans représentation géométrique plus riche.

## Restant à étudier


### D. Niveaux linéaires et convexes suivants

Statut : **non implémenté**.

- relaxations corde-longueur par polygones rationnels extérieurs ou SOCP;
- fermeture vectorielle des contours intérieur et extérieur;
- relèvement SDP des produits corde/rotation et des déterminants d'aire;
- système exact corde/rotation/aire en QF_NRA, seulement après les relaxations
  précédentes.

Ces niveaux doivent rester dans des modules séparés. Une relaxation infaisable
est un certificat de rejet; une relaxation faisable n'est pas une preuve de
réalisabilité.

### E. Points fixes et classification des isométries

- copie directe fixant deux points distincts, donc identité;
- réflexion pure imposant un contact porté par son axe;
- détection précoce de deux copies forcées à avoir la même isométrie.

### F. Symétries de blocs complets

Étendre les relations `Inverse` / `Mirror` des variables individuelles aux mots
décorés complets, angles intermédiaires compris.

### G. Solveur exact des profils entièrement droits

Encoder les longueurs, directions, fermetures, intersections de segments et
disjonctions des intérieurs comme problème semialgébrique fini.

### H. Inégalités corde-longueur

Sous une hypothèse explicite de rectifiabilité :

- `norme(corde) <= longueur`;
- inégalités polygonales strictes pour une boucle non dégénérée;
- contradictions entre un sous-arc propre et un arc congruent qui le contient.

### I. Aire orientée et moments

Ajouter progressivement aire, barycentre et moments quadratiques comme
invariants de concaténation. Leur gain doit être mesuré après l'imposition des
isométries globales, car certaines égalités deviennent alors automatiques.

### J. Réduction topologique avant les équations de mots

Étudier une classification par le réseau à quatre arcs entre les deux pôles,
ou une reformulation par systèmes d'identification d'intervalles / complexes de
bandes. Cette piste est une refonte de recherche plutôt qu'un filtre local.

## Formal profile reduction

Implemented:

- exact canonical classes under pole exchange, contour reversal/global mirror,
  copy permutation, curve renaming/reorientation, and signed angle renaming;
- actual removal of non-representative class members from the survivor pipeline;
- exact contour-shape subsumption by nonerasing decorated curve substitution;
- explicit substitution certificates and retained detailed audit records;
- conservative restriction: only profiles with entirely free curve-relation
  components are currently allowed to absorb another profile.

Important distinction:

- symmetry equivalence means two decorated mapping configurations are identical
  modulo the admitted internal symmetries;
- contour-shape subsumption means every contour represented by the refined
  profile is already an instance of a more general solved profile. The refined
  profile's own contact mapping need not be a subdivision of the general one,
  because the general solved profile supplies a valid mapping for the same
  contour;
- exact refinement of both mappings is available as a stronger diagnostic API,
  but is not required for shape-family pruning.

Remaining research:

- compare profiles whose general curve components have reflection/straightness
  constraints instead of being fully free;
- detect overlaps of geometric families that are not related by a uniform word
  substitution;
- separate mathematical profile reduction from numerical polygonal mesh
  refinement.
