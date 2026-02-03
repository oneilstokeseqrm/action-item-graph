"""
Integration tests for DealRepository.

Tests CRUD operations for Deal, DealVersion, Interaction enrichment,
and Account verification against the neo4j_structured database.

Run with: pytest tests/test_deal_repository.py -v

Requires DEAL_NEO4J_* environment variables to be set.
"""

import uuid
from datetime import datetime

import pytest

from deal_graph.clients.neo4j_client import DealNeo4jClient
from deal_graph.models.deal import Deal, DealStage, MEDDICProfile
from deal_graph.repository import DealRepository


# =========================================================================
# Helpers
# =========================================================================


def _unique_opp_id() -> str:
    """Generate a unique opportunity_id for test isolation."""
    return str(uuid.uuid4())


def _unique_interaction_id() -> str:
    """Generate a unique interaction_id for test isolation."""
    return f'int_test_{uuid.uuid4().hex[:8]}'


TENANT_ID = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
ACCOUNT_ID = 'acct_deal_test_001'


# =========================================================================
# Account Tests
# =========================================================================


class TestAccountOperations:
    """Test Account verification/creation."""

    @pytest.mark.asyncio
    async def test_verify_account_creates_if_missing(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """verify_account() creates the Account if skeleton hasn't yet."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            repo = DealRepository(client)

            test_account = f'acct_test_{uuid.uuid4().hex[:8]}'
            result = await repo.verify_account(TENANT_ID, test_account)

            assert result is not None
            assert result['account_id'] == test_account
            assert result['tenant_id'] == str(TENANT_ID)

            print(f'\nverify_account created: {test_account}')

            # Cleanup
            await client.execute_write(
                'MATCH (a:Account {account_id: $aid, tenant_id: $tid}) DETACH DELETE a',
                {'aid': test_account, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_verify_account_idempotent(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """Calling verify_account() twice on the same account is safe."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            repo = DealRepository(client)

            test_account = f'acct_test_{uuid.uuid4().hex[:8]}'
            r1 = await repo.verify_account(TENANT_ID, test_account)
            r2 = await repo.verify_account(TENANT_ID, test_account)

            assert r1['account_id'] == r2['account_id']
            print(f'\nverify_account idempotent ✓')

            # Cleanup
            await client.execute_write(
                'MATCH (a:Account {account_id: $aid, tenant_id: $tid}) DETACH DELETE a',
                {'aid': test_account, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()


# =========================================================================
# Interaction Tests
# =========================================================================


class TestInteractionOperations:
    """Test Interaction read and enrichment."""

    @pytest.mark.asyncio
    async def test_ensure_interaction_creates_if_missing(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """ensure_interaction() creates node when skeleton hasn't run."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            repo = DealRepository(client)

            iid = _unique_interaction_id()
            result = await repo.ensure_interaction(
                tenant_id=TENANT_ID,
                interaction_id=iid,
                content_text='Test transcript content for deal extraction.',
                interaction_type='transcript',
                timestamp=datetime(2026, 1, 31, 10, 30),
                source='web-mic',
            )

            assert result['interaction_id'] == iid
            assert result['content_text'] == 'Test transcript content for deal extraction.'
            assert result['interaction_type'] == 'transcript'
            print(f'\nensure_interaction created: {iid}')

            # Cleanup
            await client.execute_write(
                'MATCH (i:Interaction {interaction_id: $iid, tenant_id: $tid}) DETACH DELETE i',
                {'iid': iid, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_read_interaction(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """read_interaction() retrieves content_text from an existing node."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            repo = DealRepository(client)

            # Create a test interaction first
            iid = _unique_interaction_id()
            await repo.ensure_interaction(
                tenant_id=TENANT_ID,
                interaction_id=iid,
                content_text='Sarah discussed the $250K platform deal with James.',
            )

            # Read it back
            result = await repo.read_interaction(TENANT_ID, iid)

            assert result is not None
            assert result['content_text'] == 'Sarah discussed the $250K platform deal with James.'
            print(f'\nread_interaction content_text: {result["content_text"][:60]}...')

            # Cleanup
            await client.execute_write(
                'MATCH (i:Interaction {interaction_id: $iid, tenant_id: $tid}) DETACH DELETE i',
                {'iid': iid, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_read_nonexistent_interaction(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """read_interaction() returns None for nonexistent node."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            repo = DealRepository(client)

            result = await repo.read_interaction(TENANT_ID, 'nonexistent_interaction_id')
            assert result is None
            print('\nread_interaction returns None for missing node ✓')
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_enrich_interaction(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """enrich_interaction() adds processed_at and deal_count."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            repo = DealRepository(client)

            iid = _unique_interaction_id()
            await repo.ensure_interaction(
                tenant_id=TENANT_ID,
                interaction_id=iid,
                content_text='Test content for enrichment.',
            )

            # Enrich
            result = await repo.enrich_interaction(
                tenant_id=TENANT_ID,
                interaction_id=iid,
                deal_count=2,
            )

            assert result is not None
            assert result['deal_count'] == 2
            assert 'processed_at' in result
            # Original content_text should be preserved
            assert result['content_text'] == 'Test content for enrichment.'
            print(f'\nenrich_interaction: deal_count={result["deal_count"]}, '
                  f'processed_at={result["processed_at"]}')

            # Cleanup
            await client.execute_write(
                'MATCH (i:Interaction {interaction_id: $iid, tenant_id: $tid}) DETACH DELETE i',
                {'iid': iid, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()


# =========================================================================
# Deal Tests
# =========================================================================


class TestDealOperations:
    """Test Deal CRUD with MERGE-on-skeleton pattern."""

    @pytest.mark.asyncio
    async def test_create_deal_from_scratch(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """create_deal() creates a new Deal with skeleton + enrichment properties."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            opp_id = _unique_opp_id()
            deal = Deal(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                name='Acme Platform Deal',
                stage=DealStage.QUALIFICATION,
                amount=250000.0,
                account_id=ACCOUNT_ID,
                meddic=MEDDICProfile(
                    metrics='$200K annual savings from 80% reduction in reconciliation time',
                    metrics_confidence=0.8,
                    economic_buyer='Maria Chen, VP Engineering',
                    economic_buyer_confidence=0.9,
                ),
                opportunity_summary='Enterprise data platform deal with Acme Corp.',
                confidence=0.85,
                source_interaction_id=uuid.uuid4(),
            )

            result = await repo.create_deal(deal)

            # Verify skeleton properties
            assert result['opportunity_id'] == opp_id
            assert result['tenant_id'] == str(TENANT_ID)
            assert result['name'] == 'Acme Platform Deal'
            assert result['stage'] == 'qualification'
            assert result['amount'] == 250000.0

            # Verify enrichment properties
            assert result['meddic_economic_buyer'] == 'Maria Chen, VP Engineering'
            assert result['meddic_metrics_confidence'] == 0.8
            assert result['opportunity_summary'] == 'Enterprise data platform deal with Acme Corp.'
            assert result['confidence'] == 0.85

            print(f'\ncreate_deal: {opp_id}')
            print(f'  name: {result["name"]}')
            print(f'  amount: {result["amount"]}')
            print(f'  meddic_completeness: {result.get("meddic_completeness", "N/A")}')

            # Cleanup
            await client.execute_write(
                'MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid}) DETACH DELETE d',
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_create_deal_merge_existing_skeleton(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """create_deal() enriches a pre-existing skeleton Deal stub."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            opp_id = _unique_opp_id()

            # Simulate skeleton creating a stub Deal
            await client.execute_write(
                """
                CREATE (d:Deal {
                    tenant_id: $tid,
                    opportunity_id: $oid,
                    name: 'Skeleton Stub',
                    stage: 'prospecting'
                })
                """,
                {'tid': str(TENANT_ID), 'oid': opp_id},
            )

            # Now our pipeline enriches it
            deal = Deal(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                name='Skeleton Stub',
                stage=DealStage.QUALIFICATION,
                amount=150000.0,
                account_id=ACCOUNT_ID,
                meddic=MEDDICProfile(
                    identified_pain='Data silos costing $200K annually',
                    identified_pain_confidence=0.9,
                ),
                opportunity_summary='Data platform consolidation opportunity.',
                confidence=0.85,
            )

            result = await repo.create_deal(deal)

            # Should have enriched the existing stub, not created a duplicate
            assert result['opportunity_id'] == opp_id
            assert result['stage'] == 'qualification'  # Updated from prospecting
            assert result['amount'] == 150000.0
            assert result['meddic_identified_pain'] == 'Data silos costing $200K annually'
            assert result['opportunity_summary'] == 'Data platform consolidation opportunity.'

            # Verify only one Deal node exists with this key
            count_result = await client.execute_query(
                'MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid}) RETURN count(d) as cnt',
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
            assert count_result[0]['cnt'] == 1
            print(f'\nMERGE on existing skeleton: enrichment applied, no duplicate ✓')

            # Cleanup
            await client.execute_write(
                'MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid}) DETACH DELETE d',
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_deal(self, deal_neo4j_credentials: dict[str, str]):
        """get_deal() retrieves by composite key."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            opp_id = _unique_opp_id()
            deal = Deal(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                name='Get Test Deal',
                amount=100000.0,
            )
            await repo.create_deal(deal)

            result = await repo.get_deal(TENANT_ID, opp_id)
            assert result is not None
            assert result['opportunity_id'] == opp_id
            assert result['name'] == 'Get Test Deal'
            print(f'\nget_deal: {result["name"]} ✓')

            # Nonexistent deal
            missing = await repo.get_deal(TENANT_ID, 'nonexistent_opp_id')
            assert missing is None
            print('get_deal returns None for missing ✓')

            # Cleanup
            await client.execute_write(
                'MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid}) DETACH DELETE d',
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_update_deal(self, deal_neo4j_credentials: dict[str, str]):
        """update_deal() increments version and updates properties."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            opp_id = _unique_opp_id()
            deal = Deal(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                name='Update Test Deal',
                stage=DealStage.QUALIFICATION,
                amount=100000.0,
                version=1,
            )
            await repo.create_deal(deal)

            # Update enrichment properties
            result = await repo.update_deal(
                TENANT_ID,
                opp_id,
                {
                    'stage': 'proposal',
                    'amount': 200000.0,
                    'meddic_champion': 'James Park, Director of Sales Ops',
                    'meddic_champion_confidence': 0.85,
                },
            )

            assert result['stage'] == 'proposal'
            assert result['amount'] == 200000.0
            assert result['meddic_champion'] == 'James Park, Director of Sales Ops'
            assert result['version'] == 2  # Incremented
            print(f'\nupdate_deal: stage=proposal, amount=200K, version=2 ✓')

            # Cleanup
            await client.execute_write(
                'MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid}) DETACH DELETE d',
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()


# =========================================================================
# DealVersion Tests
# =========================================================================


class TestDealVersionOperations:
    """Test DealVersion snapshot creation and history."""

    @pytest.mark.asyncio
    async def test_create_version_snapshot(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """create_version_snapshot() captures Deal state before update."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            opp_id = _unique_opp_id()
            deal = Deal(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                name='Version Test Deal',
                stage=DealStage.QUALIFICATION,
                amount=150000.0,
                meddic=MEDDICProfile(
                    champion='Sarah Jones, VP Engineering',
                    champion_confidence=0.9,
                ),
                opportunity_summary='Data platform deal.',
                version=1,
            )
            await repo.create_deal(deal)

            # Create snapshot
            interaction_id = uuid.uuid4()
            version = await repo.create_version_snapshot(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                change_summary='Budget expanded from $150K to $300K after CEO demo.',
                changed_fields=['amount', 'stage', 'meddic_economic_buyer'],
                change_source_interaction_id=interaction_id,
            )

            assert version['deal_opportunity_id'] == opp_id
            assert version['version'] == 1  # Snapshot of version 1
            assert version['name'] == 'Version Test Deal'
            assert version['amount'] == 150000.0
            assert version['stage'] == 'qualification'
            assert version['meddic_champion'] == 'Sarah Jones, VP Engineering'
            assert version['change_summary'] == 'Budget expanded from $150K to $300K after CEO demo.'
            assert version['changed_fields'] == ['amount', 'stage', 'meddic_economic_buyer']
            assert version['change_source_interaction_id'] == str(interaction_id)

            print(f'\ncreate_version_snapshot:')
            print(f'  version_id: {version["version_id"]}')
            print(f'  snapshot version: {version["version"]}')
            print(f'  change_summary: {version["change_summary"][:50]}...')
            print(f'  changed_fields: {version["changed_fields"]}')

            # Cleanup
            await client.execute_write(
                """
                MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid})
                OPTIONAL MATCH (d)-[:HAS_VERSION]->(v:DealVersion)
                DETACH DELETE d, v
                """,
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_deal_history(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """get_deal_history() returns version chain ordered by version desc."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            opp_id = _unique_opp_id()
            deal = Deal(
                tenant_id=TENANT_ID,
                opportunity_id=opp_id,
                name='History Test Deal',
                stage=DealStage.PROSPECTING,
                amount=100000.0,
                version=1,
            )
            await repo.create_deal(deal)

            # Create version 1 snapshot, then update
            await repo.create_version_snapshot(
                TENANT_ID, opp_id,
                change_summary='Pain identified: data silos',
                changed_fields=['meddic_identified_pain'],
            )
            await repo.update_deal(TENANT_ID, opp_id, {
                'stage': 'qualification',
                'meddic_identified_pain': 'Data silos costing $200K/year',
            })

            # Create version 2 snapshot, then update
            await repo.create_version_snapshot(
                TENANT_ID, opp_id,
                change_summary='Champion identified: Sarah',
                changed_fields=['meddic_champion', 'amount'],
            )
            await repo.update_deal(TENANT_ID, opp_id, {
                'stage': 'proposal',
                'amount': 250000.0,
                'meddic_champion': 'Sarah Jones',
            })

            # Get history
            history = await repo.get_deal_history(TENANT_ID, opp_id)
            assert len(history) == 2
            # Ordered by version DESC
            assert history[0]['version'] == 2
            assert history[1]['version'] == 1

            print(f'\nget_deal_history: {len(history)} versions')
            for v in history:
                print(f'  v{v["version"]}: {v["change_summary"]}')

            # Cleanup
            await client.execute_write(
                """
                MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid})
                OPTIONAL MATCH (d)-[:HAS_VERSION]->(v:DealVersion)
                DETACH DELETE d, v
                """,
                {'oid': opp_id, 'tid': str(TENANT_ID)},
            )
        finally:
            await client.close()


# =========================================================================
# Query Tests
# =========================================================================


class TestDealQueries:
    """Test account-scoped deal queries."""

    @pytest.mark.asyncio
    async def test_get_deals_for_account(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """get_deals_for_account() returns deals scoped to account."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()
            repo = DealRepository(client)

            test_account = f'acct_test_{uuid.uuid4().hex[:8]}'
            opp_ids = []

            # Create two deals for the same account
            for i, stage in enumerate([DealStage.QUALIFICATION, DealStage.PROPOSAL]):
                opp_id = _unique_opp_id()
                opp_ids.append(opp_id)
                deal = Deal(
                    tenant_id=TENANT_ID,
                    opportunity_id=opp_id,
                    name=f'Account Deal {i + 1}',
                    stage=stage,
                    amount=(i + 1) * 100000.0,
                    account_id=test_account,
                )
                await repo.create_deal(deal)

            # Query
            deals = await repo.get_deals_for_account(TENANT_ID, test_account)
            assert len(deals) >= 2

            # Filter by stage
            qual_deals = await repo.get_deals_for_account(
                TENANT_ID, test_account, stage=DealStage.QUALIFICATION
            )
            assert len(qual_deals) >= 1
            assert all(d['stage'] == 'qualification' for d in qual_deals)

            print(f'\nget_deals_for_account: {len(deals)} deals, '
                  f'{len(qual_deals)} in qualification')

            # Cleanup
            for opp_id in opp_ids:
                await client.execute_write(
                    'MATCH (d:Deal {opportunity_id: $oid, tenant_id: $tid}) DETACH DELETE d',
                    {'oid': opp_id, 'tid': str(TENANT_ID)},
                )
        finally:
            await client.close()
