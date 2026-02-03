"""
Neo4j client for the Deal Graph pipeline.

Provides DealNeo4jClient which connects to the existing neo4j_structured
database and manages enrichment schema (vector indexes, DealVersion constraints).
"""

from .neo4j_client import DealNeo4jClient

__all__ = ['DealNeo4jClient']
