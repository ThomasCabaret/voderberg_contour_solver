"""Machine-readable status of every analysis stage."""
from __future__ import annotations
from typing import Dict
import settings

def method_registry() -> Dict[str, Dict[str, object]]:
    return {
        "placement_enumerator": {"status": settings.STATUS_EXACT, "automatic_rejection_allowed": True, "description": "Finite exhaustive enumeration of pole-image placements and contact-imposed parities."},
        "formal_word_solver": {"status": settings.STATUS_BOUNDED, "automatic_rejection_allowed": False, "description": "Every emitted terminal profile is checked, but the audit explores only configured depth/state bounds and may also apply an explicitly reported temporary cycle-unroll cap."},
        "formal_cycle_unroll_cap": {"status": settings.STATUS_BOUNDED, "automatic_rejection_allowed": False, "description": "Independent anti-echo policy. It prunes a branch after repeated returns to the same residual system, reports every hit as truncation, and does not recognize parametric families."},
        "decorated_solution_canonicalization": {"status": settings.STATUS_EXACT, "automatic_rejection_allowed": False, "description": "Canonical key for the terminal contour together with both copy mappings, modulo pole exchange, curve/angle renaming, copy permutation, and global mirror. It annotates but does not delete profiles."},
        "point_angle_solver": {"status": settings.STATUS_EXACT, "automatic_rejection_allowed": True, "description": "Exact signed angle classes and forced-zero point turns."},
        "total_turn_filter": {"status": settings.STATUS_EXACT, "automatic_rejection_allowed": True, "description": "Exact necessary total-turn feasibility test in the symbolic model."},
        "pole_angle_filter": {"status": settings.STATUS_EXACT, "automatic_rejection_allowed": True, "description": "Exact necessary simultaneous inequalities at P0 and P1."},
        "translation_holonomy_filter": {"status": settings.STATUS_SOUND_INCOMPLETE, "automatic_rejection_allowed": True, "description": "Rejects only formally proved chord-closure contradictions; undecided systems are retained."},
        "forced_point_coincidence": {"status": settings.STATUS_SOUND_INCOMPLETE, "automatic_rejection_allowed": False, "description": "Standalone prototype, not integrated into core status."},
        "external_boundary_builder": {"status": settings.STATUS_EXPERIMENTAL, "automatic_rejection_allowed": False, "description": "Constructs the shared inner/outer boundary system and exact linear turn constraints as a separate report layer."},
        "joint_translation_z3": {"status": settings.STATUS_SOUND_INCOMPLETE, "automatic_rejection_allowed": True, "description": "Z3/NLSAT solves a polynomial relaxation. UNSAT is an exact discard for the encoded necessary conditions; SAT is only a candidate and the complete rigid-tile realization problem is not yet encoded."},
    }
