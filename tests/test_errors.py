"""
Tests for the errors module.
"""

import pytest

from action_item_graph.errors import (
    ActionItemGraphError,
    PipelineError,
    ValidationError,
    ExtractionError,
    OpenAIError,
    OpenAIRateLimitError,
    Neo4jError,
    Neo4jConnectionError,
    Neo4jConstraintError,
    PartialSuccessResult,
    wrap_openai_error,
    wrap_neo4j_error,
)


class TestErrorHierarchy:
    """Test error class hierarchy."""

    def test_base_error_with_context(self):
        """Test that base error captures context."""
        error = ActionItemGraphError(
            "Something went wrong",
            context={"key": "value", "count": 42},
        )

        assert error.message == "Something went wrong"
        assert error.context == {"key": "value", "count": 42}
        assert "key=value" in str(error) or "key" in str(error)

    def test_base_error_without_context(self):
        """Test error without context."""
        error = ActionItemGraphError("Simple error")

        assert error.message == "Simple error"
        assert error.context == {}
        assert str(error) == "Simple error"

    def test_error_inheritance(self):
        """Test that errors inherit correctly."""
        pipeline_error = PipelineError("Pipeline failed")
        validation_error = ValidationError("Invalid input")
        extraction_error = ExtractionError("Extraction failed")

        assert isinstance(pipeline_error, ActionItemGraphError)
        assert isinstance(validation_error, PipelineError)
        assert isinstance(extraction_error, PipelineError)

    def test_client_error_inheritance(self):
        """Test client error hierarchy."""
        openai_error = OpenAIError("API error")
        rate_limit = OpenAIRateLimitError("Rate limited")
        neo4j_error = Neo4jError("DB error")
        connection_error = Neo4jConnectionError("Connection failed")

        assert isinstance(openai_error, ActionItemGraphError)
        assert isinstance(rate_limit, OpenAIError)
        assert isinstance(neo4j_error, ActionItemGraphError)
        assert isinstance(connection_error, Neo4jError)


class TestErrorWrapping:
    """Test error wrapping utilities."""

    def test_wrap_openai_rate_limit(self):
        """Test wrapping rate limit errors."""
        original = Exception("Rate limit exceeded")
        wrapped = wrap_openai_error(original)

        assert isinstance(wrapped, OpenAIRateLimitError)
        assert "Rate limit" in wrapped.message or "rate limit" in wrapped.message.lower()

    def test_wrap_openai_content_policy(self):
        """Test wrapping content policy errors."""
        original = Exception("Content policy violation: refused to process")
        wrapped = wrap_openai_error(original)

        assert isinstance(wrapped, OpenAIError)
        assert wrapped.context.get("error_type") == "Exception"

    def test_wrap_openai_generic(self):
        """Test wrapping generic OpenAI errors."""
        original = Exception("Unknown API error")
        wrapped = wrap_openai_error(original, context={"attempt": 3})

        assert isinstance(wrapped, OpenAIError)
        assert "attempt" in wrapped.context
        assert wrapped.context["attempt"] == 3

    def test_wrap_neo4j_connection(self):
        """Test wrapping connection errors."""
        original = Exception("Unable to connect to database")
        wrapped = wrap_neo4j_error(original)

        assert isinstance(wrapped, Neo4jConnectionError)

    def test_wrap_neo4j_constraint(self):
        """Test wrapping constraint errors."""
        original = Exception("Unique constraint violation")
        wrapped = wrap_neo4j_error(original)

        assert isinstance(wrapped, Neo4jConstraintError)

    def test_wrap_neo4j_generic(self):
        """Test wrapping generic Neo4j errors."""
        original = Exception("Query syntax error")
        wrapped = wrap_neo4j_error(original)

        assert isinstance(wrapped, Neo4jError)


class TestPartialSuccessResult:
    """Test partial success handling."""

    def test_empty_result(self):
        """Test empty partial success result."""
        result = PartialSuccessResult()

        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.total_count == 0
        assert result.all_succeeded is True  # Vacuously true
        assert result.all_failed is True  # Vacuously true
        assert result.partial_success is False

    def test_all_success(self):
        """Test all items succeeding."""
        result = PartialSuccessResult()
        result.add_success(item_id="id1", data={"status": "created"})
        result.add_success(item_id="id2", data={"status": "created"})

        assert result.success_count == 2
        assert result.failure_count == 0
        assert result.all_succeeded is True
        assert result.all_failed is False
        assert result.partial_success is False

    def test_all_failure(self):
        """Test all items failing."""
        result = PartialSuccessResult()
        error1 = ExtractionError("Failed to extract")
        error2 = ValidationError("Invalid input")
        result.add_failure(error1, item_id="id1")
        result.add_failure(error2, item_id="id2")

        assert result.success_count == 0
        assert result.failure_count == 2
        assert result.all_succeeded is False
        assert result.all_failed is True
        assert result.partial_success is False

    def test_partial_success(self):
        """Test partial success scenario."""
        result = PartialSuccessResult()
        result.add_success(item_id="id1")
        result.add_failure(ValidationError("Failed"), item_id="id2")
        result.add_success(item_id="id3")

        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.total_count == 3
        assert result.all_succeeded is False
        assert result.all_failed is False
        assert result.partial_success is True

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = PartialSuccessResult()
        result.add_success(item_id="id1")
        result.add_failure(ValidationError("Bad input"), item_id="id2")

        data = result.to_dict()

        assert data["success_count"] == 1
        assert data["failure_count"] == 1
        assert data["total_count"] == 2
        assert data["all_succeeded"] is False
        assert "id1" in data["succeeded_ids"]
        assert "id2" in data["failed_ids"]
        assert len(data["errors"]) == 1
