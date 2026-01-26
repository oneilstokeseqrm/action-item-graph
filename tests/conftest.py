"""
Pytest configuration and shared fixtures for live integration tests.

Key fixtures:
- openai_api_key: OpenAI API key from environment
- neo4j_credentials: Neo4j credentials dict
- sample_tenant_id: Test tenant ID
- sample_account_id: Test account ID

NOTE: Tests create their own Neo4j and OpenAI client instances.
When running the full test suite, connection exhaustion may occur
on Neo4j Aura due to many concurrent connections. Run tests in
smaller batches if this occurs.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv

env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)


@pytest.fixture
def openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        pytest.skip('OPENAI_API_KEY not set')
    return key


@pytest.fixture
def neo4j_credentials() -> dict[str, str]:
    """Get Neo4j credentials from environment."""
    uri = os.getenv('NEO4J_URI')
    password = os.getenv('NEO4J_PASSWORD')
    if not uri or not password:
        pytest.skip('NEO4J_URI or NEO4J_PASSWORD not set')
    return {
        'uri': uri,
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': password,
        'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
    }


@pytest.fixture
def sample_transcript() -> str:
    """Sample transcript for testing extraction."""
    return """
John: Thanks for joining the call today. We need to discuss the proposal timeline.
Sarah: Agreed. I'll send over the updated deck by Friday.
John: Perfect. And I'll schedule the follow-up demo with the technical team for next week.
Sarah: Sounds good. Also, we should loop in legal to review the contract terms.
John: I'll reach out to them today.
""".strip()


@pytest.fixture
def sample_tenant_id() -> str:
    """Sample tenant ID for testing."""
    return '550e8400-e29b-41d4-a716-446655440000'


@pytest.fixture
def sample_account_id() -> str:
    """Sample account ID for testing."""
    return 'acct_test_001'
