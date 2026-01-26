"""
Tests for the logging module.
"""

import pytest

from action_item_graph.logging import (
    PipelineTimer,
    logging_context,
    get_trace_id,
    get_tenant_id,
    get_account_id,
)


class TestLoggingContext:
    """Test logging context management."""

    def test_logging_context_sets_values(self):
        """Test that logging context sets values correctly."""
        with logging_context(
            trace_id="trace_123",
            tenant_id="tenant_abc",
            account_id="acct_xyz",
        ):
            assert get_trace_id() == "trace_123"
            assert get_tenant_id() == "tenant_abc"
            assert get_account_id() == "acct_xyz"

    def test_logging_context_restores_values(self):
        """Test that context is restored after exiting."""
        # Set initial context
        with logging_context(trace_id="outer"):
            assert get_trace_id() == "outer"

            # Nested context
            with logging_context(trace_id="inner"):
                assert get_trace_id() == "inner"

            # Should be restored
            assert get_trace_id() == "outer"

        # Should be None outside
        assert get_trace_id() is None

    def test_logging_context_partial_values(self):
        """Test that partial context values work."""
        with logging_context(tenant_id="tenant_only"):
            assert get_tenant_id() == "tenant_only"
            assert get_trace_id() is None
            assert get_account_id() is None


class TestPipelineTimer:
    """Test pipeline timing functionality."""

    def test_timer_records_stages(self):
        """Test that timer records stage durations."""
        timer = PipelineTimer()

        with timer.stage("stage1"):
            pass  # Minimal work

        with timer.stage("stage2"):
            pass

        assert "stage1" in timer.stages
        assert "stage2" in timer.stages
        assert timer.stages["stage1"] >= 0
        assert timer.stages["stage2"] >= 0

    def test_timer_manual_record(self):
        """Test manual recording of stage times."""
        timer = PipelineTimer()
        timer.record("custom_stage", 150.5)

        assert "custom_stage" in timer.stages
        assert timer.stages["custom_stage"] == 150.5

    def test_timer_total_ms(self):
        """Test total elapsed time calculation."""
        timer = PipelineTimer()
        # Timer starts automatically
        assert timer.total_ms >= 0

    def test_timer_summary(self):
        """Test summary dictionary format."""
        timer = PipelineTimer()
        timer.record("extraction", 100.0)
        timer.record("matching", 50.0)

        summary = timer.summary()

        assert "total_ms" in summary
        assert "stages" in summary
        assert summary["stages"]["extraction"] == 100.0
        assert summary["stages"]["matching"] == 50.0
