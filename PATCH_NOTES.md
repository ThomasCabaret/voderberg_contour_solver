# Patch: formal-equation audit and configurable polyline complexity

Copy the files in this patch to the project root and replace files with the
same names.

## Formal word-equation audit

The normal audit now writes `formal_equation_audit.json` and includes its
summary in `geometric_filter_audit.json` and `geometric_filter_profiles.json`.

Each of the 2816 placement systems records:

- variables, equations and occurrence multiplicities;
- linear/quadratic/cubic/higher classification;
- involution and orientation occurrence data;
- variable interaction components;
- recommended complete-solver family;
- bounded search state counts, branch counts and truncation reasons;
- terminal profile ids found within the configured bounds.

The bounded formal search is executed once; the same traversal supplies both
terminal profiles and diagnostic statistics.

## Geometry search

Use:

```bat
py -3 project_cli.py geometry --intermediate-points N
```

`N=0` uses one straight edge per formal variable. `N=1` preserves the previous
two-edge model. In general, a variable has `N+1` edges. Internal turns are
parameterized so their sum equals the variable's total Kappa rotation.

## Validation

- Full audit without Z3 reproduced 1078 terminal profiles and 50 final
  pre-Z3 survivors.
- Formal structure totals: 608 quadratic systems and 2208 cubic systems.
- Targeted unit tests pass.
- Geometry smoke tests ran with 0 and 2 intermediate points.


## Search-completeness cross-table update

The formal audit now separates productive and empty searches from exhausted and
truncated searches. It explicitly counts:

- exhausted searches with terminal profiles;
- truncated searches with terminal profiles, whose profile set may be incomplete;
- exhausted searches without terminal profiles;
- truncated searches without terminal profiles, whose satisfiability is unresolved;
- immediately inconsistent systems.

`--max-depth` and `--max-states` are documented CLI controls. Zero removes the
corresponding bound and may make cyclic searches non-terminating.
