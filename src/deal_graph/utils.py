"""
Utility helpers for the Deal Graph pipeline.

uuid7() wraps fastuuid.uuid7() to return a stdlib uuid.UUID instance.
fastuuid.UUID is a Rust-backed type that is NOT isinstance-compatible with
uuid.UUID, so we roundtrip through the string representation.
"""

import fastuuid
from uuid import UUID


def uuid7() -> UUID:
    """Generate a UUIDv7 (time-sortable) as a stdlib uuid.UUID."""
    return UUID(str(fastuuid.uuid7()))
