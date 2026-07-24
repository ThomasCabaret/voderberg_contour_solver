# Prototype: external boundary and shared closure system

This note accompanies `external_boundary_constraints.py`. The module is not
integrated into the production audit pipeline yet.

## 1. External contour construction

For each covering copy, the contact factor and the free factor are complementary
arcs of the same prototype cycle.

The placement already stores the unique isometry parity required by opposite-side
contact:

```text
mirror_sign = -contact_direction
```

The positively oriented physical boundary of a direct copy follows the prototype
counterclockwise direction. The positively oriented boundary of a reflected copy
follows the prototype clockwise direction. Therefore the free outer factors are:

```text
copy A: A_start -> A_end in direction a_mirror_sign, physical P0 -> P1
copy B: B_start -> B_end in direction b_mirror_sign, physical P1 -> P0
```

Their concatenation is the candidate outer boundary of the union.

## 2. Outer turns at the poles

At a pole, the three physical tile sectors have prototype turning angles
`tau_0`, `tau_1`, `tau_2`. Their interior angles are `pi - tau_i`.

The union interior angle is the sum of the three sectors, so the outer-boundary
turn is:

```text
tau_outer = tau_0 + tau_1 + tau_2 - 2*pi
```

The formula uses the same `Theta` classes as the reference contour and counts
multiplicity when two physical copies present the same prototype point.

## 3. Shared rotation equations

The module constructs two equations over the same variables:

```text
turn(reference boundary) = 2*pi
turn(union boundary)     = 2*pi
```

Curve variables `Kappa[X]` are unbounded. Point variables `Theta_i` satisfy
`Theta_i/pi in (-1, 1)`. The two linear equations are solved jointly and exactly.
The solver handles all ranks of the two-equation system; when only bounded point
variables remain, it tests strict membership in the corresponding 1D interval
or 2D zonotope.

## 4. Shared translation equations

Each free curve variable has one prototype chord `D[X]`. A reflected copy uses
`conj(D[X])` in the local start-tangent frame. Every occurrence contributes a
phasor-rotated chord:

```text
exp(i * phase) * D[X]
exp(i * phase) * conj(D[X])
```

The module constructs both closure equations with shared `Kappa`, `Theta`, and
chord variables. This is the correct object for a future simultaneous solver.

The current prototype only applies elementary exact obstructions separately to
each boundary. It does not yet solve the complete joint trigonometric/vector
system.

## 5. Assumptions still to validate before integration

- The three sectors at each pole are contiguous in the local cyclic order. The
  current opposite-side placement and pole inequalities are expected to encode
  this, but a dedicated local ribbon-order check would make it explicit.
- Additional contacts between the two exterior copies are not included. They
  would alter the actual external boundary.
- The generated outer cycle is a candidate boundary. Simplicity and absence of
  self-intersection remain future geometric conditions.
- Forced coincidences between distinct outer-boundary point occurrences should
  later be checked with `forced_point_coincidence.py` using the same shared pose
  system.

## 6. Projection constraints on curve-turn variables

A reflected contact does not only relate point turns. It also acts on the total
turn of every aligned curve occurrence. For aligned literals `L` and `R`:

```text
sign(L) * Kappa[var(L)]
    = mirror_sign * sign(R) * Kappa[var(R)]
```

The prototype derives and solves these signed scalar equations before testing
inner/outer rotation. In the current terminal word representation aligned
literals are usually identical. A reflected projection can therefore force
`Kappa[X] = -Kappa[X]`, hence `Kappa[X] = 0`.

This constraint was not previously represented by the point-angle layer. It is
kept local to this experimental module until the model is reviewed and an
integration decision is made.

## 7. Rotation identity observed in the bounded audit

After applying the projection-induced Kappa constraints, every one of the 1,078
bounded terminal profiles tested satisfied the formal identity:

```text
Turn(outer union boundary) = 3 * Turn(reference tile boundary) - 4*pi
```

Consequently, requiring the reference tile to turn by `2*pi` automatically
requires the external boundary to turn by `2*pi`. This is the expected gluing
identity for three disk boundaries with two shared contact arcs.

Therefore the external contour does not appear to add an independent total-turn
equation in this model. Its main new information should come from:

- the shared translation equations;
- forced coincidences of external-boundary points;
- the external boundary's combinatorial cycle structure;
- later, simplicity/nonintersection constraints.
