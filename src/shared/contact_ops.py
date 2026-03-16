"""
Shared contact operations used by both action item and deal pipelines.

Contains ENGAGED_ON relationship management and contact name matching.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


async def merge_contacts_to_deal(
    neo4j_client: Any,
    tenant_id: str,
    contact_ids: list[str],
    opportunity_id: str,
    source: str = 'envelope',
) -> int:
    """MERGE (Contact)-[:ENGAGED_ON]->(Deal) for each contact.

    Uses MATCH for both nodes (this service doesn't create Contact nodes).
    If either node is missing, the MATCH returns nothing and the MERGE
    is skipped — safe for race conditions with the skeleton pipeline.

    Returns count of relationships created/matched.
    """
    query = """
        MATCH (c:Contact {tenant_id: $tenant_id, contact_id: $contact_id})
        MATCH (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
        MERGE (c)-[r:ENGAGED_ON]->(d)
        ON CREATE SET r.created_at = datetime(), r.source = $source
        RETURN r IS NOT NULL as created
    """
    count = 0
    for cid in contact_ids:
        result = await neo4j_client.execute_write(
            query,
            {
                'tenant_id': tenant_id,
                'contact_id': cid,
                'opportunity_id': opportunity_id,
                'source': source,
            },
        )
        if result:
            count += 1
    return count


async def enrich_engaged_on_role(
    neo4j_client: Any,
    tenant_id: str,
    contact_id: str,
    opportunity_id: str,
    role: str,
    confidence: float,
) -> bool:
    """Enrich an existing ENGAGED_ON relationship with LLM-extracted role.

    Uses unconditional SET (not ON CREATE SET) so re-processing updates
    the role. Safe to call multiple times.
    """
    query = """
        MATCH (c:Contact {tenant_id: $tenant_id, contact_id: $contact_id})
              -[r:ENGAGED_ON]->
              (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
        SET r.role = $role,
            r.confidence = $confidence,
            r.enriched_at = datetime()
        RETURN r IS NOT NULL as enriched
    """
    result = await neo4j_client.execute_write(
        query,
        {
            'tenant_id': tenant_id,
            'contact_id': contact_id,
            'opportunity_id': opportunity_id,
            'role': role,
            'confidence': confidence,
        },
    )
    return result[0]['enriched'] if result else False


def match_name_to_contact(
    name: str,
    contacts: list[dict],
    min_ratio: float = 0.75,
) -> dict | None:
    """Fuzzy-match a freetext name to a contact from the envelope.

    Uses SequenceMatcher for sequence-based similarity (same approach
    as the owner resolver). Returns the best-matching contact dict
    or None if no match meets threshold.
    """
    if not name or not contacts:
        return None
    name_lower = name.lower().strip()
    best_match: dict | None = None
    best_ratio = 0.0
    for c in contacts:
        c_name = (c.get('name') or '').lower().strip()
        if not c_name:
            continue
        ratio = SequenceMatcher(None, name_lower, c_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = c
    return best_match if best_ratio >= min_ratio else None
