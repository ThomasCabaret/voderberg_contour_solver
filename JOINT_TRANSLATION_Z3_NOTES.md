# Polynomial joint-translation filter with Z3/NLSAT

`joint_translation_z3.py` replaces the former dReal backend. It depends only on
the pip-installable package `z3-solver`.

## Installation

From the project directory on Windows:

```bat
py -3 -m pip install -r requirements.txt
```

## Encoded model

Every angular variable is represented by a unit complex number `(c, s)` with

```text
c^2 + s^2 = 1.
```

Integral sums of angles are represented by complex multiplication. The two
translation closures use the same chord variables `D[X]`, and reflected copies
use conjugated chords. The resulting formula is polynomial QF_NRA and is passed
to Z3's `qfnra-nlsat` tactic.

The encoding intentionally forgets winding-number and principal-angle interval
information. Those constraints are handled by the existing core filters. This
makes the Z3 formula a relaxation:

- `unsat`: exact, sound rejection for the encoded necessary conditions;
- `sat_candidate`: the relaxation has a model, but this is not a proof of a
  simple or realizable tile;
- `unknown` or `timeout`: no conclusion.

The full pointwise rigid-isometry realization problem is not yet encoded.

## Commands

Run the audit with Z3 enabled, which is the default:

```bat
py -3 project_cli.py audit
```

Prepare problems without running Z3:

```bat
py -3 project_cli.py audit --skip-z3
```

Limit checks or change the per-profile timeout:

```bat
py -3 project_cli.py audit --z3-max-profiles 20 --z3-timeout-ms 10000
```

Generate and solve the Voderberg regression problem:

```bat
run_z3_voderberg.cmd
```
