# Z3 replacement patch

This patch removes the active dReal integration and replaces it with a
pip-installable Z3/NLSAT polynomial filter.

## Apply

1. Copy the files from this patch over the project root.
2. Delete the files listed in `REMOVE_OBSOLETE_FILES.txt`.
3. Install the dependency:

```bat
py -3 -m pip install -r requirements.txt
```

4. Run:

```bat
run_tests.cmd
run_audit.cmd
```

Z3 runs by default during the audit. Use `--skip-z3` to prepare the problems
without executing the solver.

## Semantics

- Z3 `unsat` is a sound exact rejection for the polynomial relaxation.
- Z3 `sat_candidate` is not a proof of a realizable tile.
- The core angle and pole filters remain unchanged.
