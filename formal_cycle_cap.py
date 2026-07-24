#!/usr/bin/env python3
"""Independent temporary policy for bounding residual-cycle unrolling.

This module does not attempt parametric cycle recognition.  It merely tracks
how often the same residual equation system has been revisited along one search
branch and can prune a fourth (or configured) return.  Any pruning performed by
this policy is explicitly reported as a truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Optional, Tuple


ResidualKey = Hashable


@dataclass(frozen=True)
class CycleVisitHistory:
    visits: Mapping[ResidualKey, int]

    @classmethod
    def start(cls, initial_key: ResidualKey) -> "CycleVisitHistory":
        return cls({initial_key: 1})

    def signature(self) -> Tuple[Tuple[str, int], ...]:
        # String forms are used only to make the path-state key deterministic
        # and hashable independently of the concrete Equation implementation.
        return tuple(sorted((repr(key), count) for key, count in self.visits.items()))

    def advance(
        self,
        residual_key: ResidualKey,
        max_cycle_unrolls: Optional[int],
    ) -> Tuple[Optional["CycleVisitHistory"], bool]:
        """Return the advanced history, or ``(None, True)`` when capped.

        The first visit is not an unrolling.  With a cap of three, a residual
        state may therefore occur four times on a branch: the initial visit and
        three returns through one or more cycles.
        """
        previous = int(self.visits.get(residual_key, 0))
        new_count = previous + 1
        repeated_visits = max(0, new_count - 1)
        if max_cycle_unrolls is not None and repeated_visits > max_cycle_unrolls:
            return None, True
        updated: Dict[ResidualKey, int] = dict(self.visits)
        updated[residual_key] = new_count
        return CycleVisitHistory(updated), False
