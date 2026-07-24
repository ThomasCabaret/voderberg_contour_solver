# Patch : exports JSON en flux, lecture progressive et vue des survivants

## Cause du `MemoryError`

L'ancien export `geometric_filter_profiles.json` appelait :

```python
json.dumps(payload, indent=2, ensure_ascii=True)
```

Cette instruction construisait la totalité du fichier sous forme d'une seule
chaîne Python avant l'écriture. Sur l'ancien export de référence, un profil
occupait en moyenne environ 11,7 ko en JSON compact et 22,8 ko avec indentation.
Pour 135 418 profils, l'ordre de grandeur est donc d'environ 1,5 Gio en compact
et 2,9 Gio avec indentation, sans compter les dictionnaires Python déjà en
mémoire et les copies temporaires d'encodage.

## Changements

### Audit et fichiers JSON

- écriture atomique avec `JSONEncoder.iterencode`, sans chaîne JSON géante;
- JSON compact pour les deux gros tableaux de profils;
- sauvegarde de `geometric_filter_survivors.json` avant l'export détaillé;
- messages `Writing:` et `Wrote:` pour chaque fichier;
- nouvelle option :

```bat
py -3 project_cli.py audit --no-detailed-profiles-output
```

Elle écrit le résumé, l'audit formel et les survivants, mais ne construit pas
les enregistrements des profils rejetés. C'est l'option recommandée lorsqu'on
veut uniquement poursuivre avec le solveur géométrique et la page web.

`--no-profiles-output` continue de désactiver les deux fichiers de profils.

### Solveur géométrique

- `--max-profiles 0` est désormais la valeur par défaut : tous les survivants
  sont essayés;
- une valeur positive limite explicitement le nombre de profils;
- le tableau `profiles` du JSON est décodé progressivement;
- les gros enregistrements source ne sont plus conservés dans chaque objet de
  recherche géométrique.

### Visualiseur web

- le fichier par défaut est maintenant `geometric_filter_survivors.json`;
- le serveur transmet le fichier par blocs et ne le resérialise plus en mémoire;
- le tableau HTML est paginé : 200 lignes par défaut, choix entre 50, 100, 200
  et 500;
- seuls les profils retenus sont donc chargés par défaut. Un autre JSON reste
  sélectionnable avec `--json`.

## Vérifications effectuées

- 16 tests ciblés réussis;
- audit réel court avec export détaillé;
- audit réel court avec `--no-detailed-profiles-output`;
- lecture progressive testée avec des blocs de 7 caractères;
- démarrage du serveur et téléchargement HTTP du JSON par blocs.
