from __future__ import annotations

from enum import Enum
from typing import Callable, Literal, TypeAlias

from vstools import NotFoundEnumValue, vs

__all__ = [
    "Match", "OrphanMatch",
    "SectionPreset",
    "CustomPostFiltering",
]


Match: TypeAlias = Literal['b'] | Literal['c'] | Literal['n'] | Literal['p'] | Literal['u']
"""A type representing all possible fieldmatches."""

OrphanMatch: TypeAlias = Literal['b'] | Literal['n'] | Literal['p'] | Literal['u']
"""Valid matches to be considered orphans.."""

SectionPreset = Callable[[vs.VideoNode], vs.VideoNode]
"""A callable preset applied to a section."""


class CustomPostFiltering(Enum):
    """When to perform custom filtering."""

    SOURCE = -1
    """Apply the custom filter after the source clip is loaded."""

    FIELD_MATCH = 0
    """Apply the custom filter after the field match is applied."""

    DECIMATE = 1
    """Apply the custom filter after the decimation is applied."""

    @classmethod
    def from_str(cls, value: str) -> CustomPostFiltering:
        """Convert a string to a CustomPosition."""

        norm_val = value.upper().replace('POST', '').strip().replace(' ', '_')

        try:
            return cls[norm_val]
        except KeyError:
            raise NotFoundEnumValue(
                "Could not find a matching CustomPostFiltering value!", cls.from_str, value
            )

    def __str__(self) -> str:
        return self.name.replace('_', ' ').title()
