"""Small state objects shared by plot renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LayoutReservation:
    """Explicit figure-space reserved by renderer decorations.

    Render branches add legends and colorbars before the final layout pass. Keeping
    these reservations in a typed object prevents branch-local variables from
    becoming hidden cross-branch state.
    """

    legend_bottom: float = 0.0
    legend_right: float = 0.0

    def reserve_bottom(self, value: float) -> None:
        self.legend_bottom = max(self.legend_bottom, float(value))

    def reserve_right(self, value: float) -> None:
        self.legend_right = max(self.legend_right, float(value))

    @property
    def has_reservation(self) -> bool:
        return self.legend_bottom > 0.0 or self.legend_right > 0.0
