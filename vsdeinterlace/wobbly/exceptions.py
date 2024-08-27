from vstools import CustomError

__all__: list[str] = [
    "WobblyError",
    "InvalidCycleError",
    "InvalidMatchError",
    "MatchMismatchError",
    "SectionError",
]


class WobblyError(CustomError):
    """Thrown when an error related to Wobbly is thrown."""


class InvalidCycleError(WobblyError):
    """Raised when a wrong cycle is given."""


class InvalidMatchError(WobblyError, TypeError):
    """Thrown when an invalid fieldmatch value is given."""


class MatchMismatchError(WobblyError, ValueError):
    """Thrown when a fieldmatch value is given that is not allowed."""


class SectionError(WobblyError):
    """Raised when there's an issue with a section."""
