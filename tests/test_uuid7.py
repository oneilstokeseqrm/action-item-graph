"""
Tests for the uuid7() helper.

Validates that uuid7() returns a stdlib uuid.UUID with correct UUIDv7 properties:
version 7, RFC 4122 variant bits, time-sortable ordering, and timestamp accuracy.
"""

import time
from uuid import UUID

from deal_graph.utils import uuid7


class TestUuid7Basic:
    """Core properties that must always hold."""

    def test_returns_stdlib_uuid(self):
        """uuid7() must return a stdlib uuid.UUID, not fastuuid.UUID."""
        result = uuid7()
        assert isinstance(result, UUID)
        # Verify it's truly stdlib, not a subclass from fastuuid
        assert type(result) is UUID

    def test_version_is_7(self):
        """UUIDv7 has version == 7."""
        result = uuid7()
        assert result.version == 7

    def test_rfc4122_variant_bits(self):
        """RFC 4122 variant: bits 62-63 of the 128-bit value must be 0b10."""
        result = uuid7()
        variant_bits = (result.int >> 62) & 0b11
        assert variant_bits == 0b10

    def test_string_roundtrip(self):
        """UUID(str(x)) == x — lossless string serialization."""
        result = uuid7()
        assert UUID(str(result)) == result

    def test_no_deal_prefix(self):
        """String representation must NOT have a deal_ prefix."""
        result = uuid7()
        assert not str(result).startswith('deal_')

    def test_standard_uuid_format(self):
        """String representation follows 8-4-4-4-12 hex format."""
        result = uuid7()
        s = str(result)
        parts = s.split('-')
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


class TestUuid7Timestamp:
    """Timestamp embedded in the top 48 bits."""

    def test_timestamp_within_tolerance(self):
        """Top 48 bits encode Unix ms timestamp within 100ms of wall clock."""
        now_ms = time.time_ns() // 1_000_000
        result = uuid7()
        ts_ms = result.int >> 80
        assert abs(ts_ms - now_ms) <= 100, (
            f'Timestamp delta {abs(ts_ms - now_ms)}ms exceeds 100ms tolerance'
        )


class TestUuid7Ordering:
    """Time-sortable ordering guarantee."""

    def test_ordering_after_sleep(self):
        """UUIDs generated 3ms apart must be strictly ordered."""
        a = uuid7()
        time.sleep(0.003)  # 3ms — guarantees different ms timestamp
        b = uuid7()

        # Full integer comparison
        assert b.int > a.int, 'Later UUID must have larger int value'

        # Timestamp portion itself must be strictly greater
        ts_a = a.int >> 80
        ts_b = b.int >> 80
        assert ts_b > ts_a, 'Later UUID must have strictly greater timestamp bits'
