# Experimental total joint-translation solver

`joint_translation_dreal.py` is standalone and is not imported by the current
pipeline.

## What is solved

The module consumes the shared inner/outer system built by
`external_boundary_constraints.py`.

It first eliminates the independent total-turn equations exactly by rational
Gaussian elimination. The remaining free point-turn parameters are restricted
to `(-pi, pi)`. Free curve-turn parameters are reduced to representatives in
`[-pi, pi]`; the code checks that this reduction is sound for the generated
integer-affine substitutions.

It then emits one bounded existential dReal formula containing:

- two complex translation closures, hence four real equations;
- shared `Theta` and `Kappa` parameters;
- shared chord variables `D[X] = (dx[X], dy[X])`;
- reflected occurrences through the correct conjugated-chord formula;
- global chord normalization;
- optionally, a strict nonzero condition for every chord.

The normalization is exact for this homogeneous system. Any nonzero solution
can be rescaled so that the sum of squared chord norms is one.

## Why dReal

The coefficient of a chord is a sum of terms such as

```text
cos(phi) + i sin(phi)
```

where `phi` is a shared affine angle expression. This is a nonlinear
transcendental existential problem. dReal accepts SMT-LIB formulas extended with
`sin` and `cos` and implements a Delta-complete procedure.

Interpretation:

- `unsat`: exact proof that the encoded shared closure system is impossible;
- `delta-sat`: candidate satisfying a Delta-perturbed formula; it must be
  validated numerically and eventually geometrically;
- timeout: no conclusion.

This is therefore complete in the Delta-decision sense, not an exact source of
positive existence proofs.

## Important model limitation

The formula solves the current closure system. It does not yet encode every
pointwise equation saying that one *single rigid isometry* maps the whole
contact arc. Consequently `delta-sat` does not prove that the tile copies can be
placed consistently. `unsat` remains a valid discard for the weaker encoded
system.

There is also a strong structural expectation that, after full rigid-isometry
constraints are introduced, outer translation closure will be algebraically
implied by inner closure and the two contact endpoint mappings. The current
module intentionally keeps both equations so this can be checked rather than
assumed.

## Usage

Emit the Voderberg problem:

```bat
python joint_translation_dreal.py --voderberg ^
  --output voderberg_joint_translation.smt2 ^
  --metadata voderberg_joint_translation.json
```

Run it when dReal is available on `PATH`:

```bat
python joint_translation_dreal.py --voderberg ^
  --output voderberg_joint_translation.smt2 ^
  --metadata voderberg_joint_translation.json ^
  --run --precision 1e-8
```

A specific terminal profile can be selected with:

```bat
python joint_translation_dreal.py ^
  --case-id 1447 ^
  --derivation equal_length,left_strictly_shorter,involutive_palindrome ^
  --output profile.smt2
```

The dReal executable can be supplied explicitly:

```bat
python joint_translation_dreal.py --voderberg --output profile.smt2 ^
  --run --dreal C:\path\to\dreal.exe
```

On Windows, dReal may be easier to run through WSL or a Linux container. The
Python module itself only generates SMT2 and invokes an executable; it does not
require dReal's Python bindings.
