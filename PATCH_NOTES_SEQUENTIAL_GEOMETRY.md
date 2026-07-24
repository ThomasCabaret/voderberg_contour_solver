# Sequential audit and heuristic geometry patch

Copy every file in this patch to the project root and replace files with the
same name.

## Audit changes

The audit now gives each independent filter its own loop and log stage. Fast
filters retain diagnostics for every terminal profile. Shared-boundary stages
receive only the 307 core survivors, and Z3 receives only the survivors of all
preceding exact checks.

A new file is written after the audit:

```text
geometric_filter_survivors.json
```

Z3 `unsat` profiles are removed from this file. SAT candidates, timeouts,
unknown results, and profiles for which Z3 was skipped remain candidates.

## Geometry program

Install the updated requirements, run the audit, then launch:

```bat
run_geometry.cmd
```

The program reads the survivor file, searches a restricted two-edge-polyline
model, writes `geometric_candidates.json`, and opens a Tk window for browsing
found candidates.

This is a heuristic discovery program. A displayed candidate is concrete for
the restricted prototype-contour model, but it is not yet a certificate that
the three physical tile copies are pairwise interior-disjoint. Failure to find
a candidate is inconclusive.
