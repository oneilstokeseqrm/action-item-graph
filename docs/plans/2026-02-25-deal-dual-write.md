# Deal Dual-Write (Phase B) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the Postgres dual-write pattern (established in Phase A for Action Items) to the Deal pipeline, so every Deal extraction also writes to the `opportunities` table and a new `deal_versions` table in Neon Postgres.

**Architecture:** The DealPipeline already writes to Neo4j. We add an optional `postgres_client` parameter (same `PostgresClient` class from Phase A) and a failure-isolated dual-write stage after Neo4j merging completes. New methods on `PostgresClient` handle Deal-specific UPSERT/INSERT. A Prisma migration in `eq-frontend` creates the required columns and table.

**Tech Stack:** Python 3.11, SQLAlchemy 2.0 async (asyncpg), PostgreSQL (Neon), pgvector, Prisma raw SQL migration

**Design doc:** This document serves as both design and implementation plan.

**Key constraint:** The `opportunities` table has an AFTER UPDATE trigger on 8 columns (`stage`, `amount`, `close_date`, `deal_status`, `forecast_category`, `next_step`, `description`, `lost_reason`) that fires `notify_forecast_job()`. Our dual-write MUST NOT write to these columns. AI extraction data goes into dedicated new columns.

---

## Pre-implementation Notes

### Column Strategy

| Category | Approach |
|----------|----------|
| **Trigger-protected (8 cols)** | DO NOT WRITE. CRM-owned. |
| **Forecast-written (7 cols)** | DO NOT WRITE. `latest_forecast_*`, `forecast_count`, `attribute_maturity_stage`, `latest_user_context` |
| **Existing AI cols** | WRITE: `latest_ai_summary`, `ai_evolution_summary`, `ai_workflow_metadata`, `ontology_completeness` |
| **New extraction cols** | ADD + WRITE: MEDDIC, ontology dimensions, embeddings, cross-reference, metadata |

### Ontology Dimension IDs (15 total, from `TRANSCRIPT_EXTRACTED_DIMENSIONS`)

**Qualification:** `champion_strength`, `economic_buyer_access`, `identified_pain`, `metrics_business_case`, `decision_criteria_alignment`, `decision_process_clarity`
**Competitive:** `competitive_position`, `incumbent_displacement_risk`
**Commercial:** `pricing_alignment`, `procurement_legal_progress`
**Engagement:** `responsiveness`
**Timeline:** `close_date_credibility`
**Technical:** `technical_fit`, `integration_security_risk`
**Organizational:** `change_readiness`

### Design Decision: Option C+ (approved by user)

Individual `dim_*` columns (forecast-pipeline-compatible) + `ontology_scores_json` JSONB (evidence-rich) + scalar `ontology_completeness` (queryable). All three written simultaneously.

---

## Task 1: Prisma Migration — Add Columns to `opportunities` + Create `deal_versions`

> **Cross-repo:** This migration goes in `/Users/peteroneil/eq-frontend/prisma/migrations/`

**Files:**
- Create: `/Users/peteroneil/eq-frontend/prisma/migrations/20260225140000_deal_graph_sync/migration.sql`

**Step 1: Write the migration SQL**

Create the migration directory and file:

```bash
mkdir -p /Users/peteroneil/eq-frontend/prisma/migrations/20260225140000_deal_graph_sync
```

Write `migration.sql`:

```sql
-- Phase B: Deal Pipeline Dual-Write
-- Adds extraction columns to opportunities + creates deal_versions table.
-- Follows Phase A pattern (action_item_graph_sync): raw SQL, IF NOT EXISTS guards.

-- =============================================================================
-- 1. Cross-reference to Neo4j Deal node + extraction metadata
-- =============================================================================
ALTER TABLE "opportunities"
  ADD COLUMN IF NOT EXISTS "graph_opportunity_id" UUID,
  ADD COLUMN IF NOT EXISTS "deal_ref" VARCHAR(50),
  ADD COLUMN IF NOT EXISTS "qualification_status" VARCHAR(20),
  ADD COLUMN IF NOT EXISTS "source_user_id" TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS "opportunities_graph_opportunity_id_key"
  ON "opportunities"("graph_opportunity_id");

-- =============================================================================
-- 2. MEDDIC qualification profile (15 columns)
-- =============================================================================
ALTER TABLE "opportunities"
  ADD COLUMN IF NOT EXISTS "meddic_metrics" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_metrics_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_economic_buyer" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_economic_buyer_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_decision_criteria" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_decision_criteria_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_decision_process" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_decision_process_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_identified_pain" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_identified_pain_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_champion" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_champion_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_completeness" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "meddic_paper_process" TEXT,
  ADD COLUMN IF NOT EXISTS "meddic_competition" TEXT;

-- =============================================================================
-- 3. Ontology dimension scores (30 individual columns)
--    Named dim_{id} to match Neo4j property names and enable forecast pipeline
--    to read via SELECT * ... WHERE key.startswith('dim_') pattern.
-- =============================================================================
ALTER TABLE "opportunities"
  -- Qualification
  ADD COLUMN IF NOT EXISTS "dim_champion_strength" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_champion_strength_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_economic_buyer_access" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_economic_buyer_access_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_identified_pain" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_identified_pain_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_metrics_business_case" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_metrics_business_case_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_decision_criteria_alignment" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_decision_criteria_alignment_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_decision_process_clarity" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_decision_process_clarity_confidence" DECIMAL(3,2),
  -- Competitive
  ADD COLUMN IF NOT EXISTS "dim_competitive_position" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_competitive_position_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_incumbent_displacement_risk" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_incumbent_displacement_risk_confidence" DECIMAL(3,2),
  -- Commercial
  ADD COLUMN IF NOT EXISTS "dim_pricing_alignment" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_pricing_alignment_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_procurement_legal_progress" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_procurement_legal_progress_confidence" DECIMAL(3,2),
  -- Engagement
  ADD COLUMN IF NOT EXISTS "dim_responsiveness" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_responsiveness_confidence" DECIMAL(3,2),
  -- Timeline
  ADD COLUMN IF NOT EXISTS "dim_close_date_credibility" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_close_date_credibility_confidence" DECIMAL(3,2),
  -- Technical
  ADD COLUMN IF NOT EXISTS "dim_technical_fit" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_technical_fit_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "dim_integration_security_risk" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_integration_security_risk_confidence" DECIMAL(3,2),
  -- Organizational
  ADD COLUMN IF NOT EXISTS "dim_change_readiness" SMALLINT,
  ADD COLUMN IF NOT EXISTS "dim_change_readiness_confidence" DECIMAL(3,2);

-- Rich JSONB store (includes evidence text — richer than individual columns)
ALTER TABLE "opportunities"
  ADD COLUMN IF NOT EXISTS "ontology_scores_json" JSONB,
  ADD COLUMN IF NOT EXISTS "ontology_version" VARCHAR(64);

-- NOTE: ontology_completeness already exists (added by Wave 5.5 forecast migration)

-- =============================================================================
-- 4. Embeddings (pgvector)
--    Prefixed with extraction_ to distinguish from any future CRM embeddings.
-- =============================================================================
ALTER TABLE "opportunities"
  ADD COLUMN IF NOT EXISTS "extraction_embedding" vector(1536),
  ADD COLUMN IF NOT EXISTS "extraction_embedding_current" vector(1536);

-- =============================================================================
-- 5. Extraction metadata
-- =============================================================================
ALTER TABLE "opportunities"
  ADD COLUMN IF NOT EXISTS "extraction_confidence" DECIMAL(3,2),
  ADD COLUMN IF NOT EXISTS "extraction_version" INT DEFAULT 1,
  ADD COLUMN IF NOT EXISTS "source_interaction_id" UUID;

-- Index on source_interaction_id for provenance queries
CREATE INDEX IF NOT EXISTS "opportunities_source_interaction_id_idx"
  ON "opportunities"("source_interaction_id");

-- =============================================================================
-- 6. deal_versions table (audit trail)
-- =============================================================================
CREATE TABLE IF NOT EXISTS "deal_versions" (
  "id" UUID NOT NULL DEFAULT gen_random_uuid(),
  "tenant_id" UUID NOT NULL,
  "opportunity_id" UUID NOT NULL,
  "version_number" INT NOT NULL,

  -- Snapshot of Deal state at version time
  "name" TEXT NOT NULL,
  "stage" VARCHAR(20) NOT NULL,
  "amount" DECIMAL(15,2),
  "opportunity_summary" TEXT,
  "evolution_summary" TEXT,

  -- MEDDIC snapshots
  "meddic_metrics" TEXT,
  "meddic_economic_buyer" TEXT,
  "meddic_decision_criteria" TEXT,
  "meddic_decision_process" TEXT,
  "meddic_identified_pain" TEXT,
  "meddic_champion" TEXT,
  "meddic_completeness" DECIMAL(3,2),

  -- Ontology snapshots
  "ontology_scores_json" JSONB,
  "ontology_completeness" DECIMAL(5,2),
  "ontology_version" VARCHAR(64),

  -- Change tracking
  "change_summary" TEXT DEFAULT '',
  "changed_fields" JSONB DEFAULT '[]',
  "change_source_interaction_id" UUID,

  -- Timestamps (bi-temporal)
  "created_at" TIMESTAMPTZ NOT NULL DEFAULT now(),
  "valid_from" TIMESTAMPTZ NOT NULL DEFAULT now(),
  "valid_until" TIMESTAMPTZ,

  CONSTRAINT "deal_versions_pkey" PRIMARY KEY ("id")
);

-- Compound uniqueness: one version per deal per tenant
CREATE UNIQUE INDEX IF NOT EXISTS "deal_versions_tenant_id_opportunity_id_version_number_key"
  ON "deal_versions"("tenant_id", "opportunity_id", "version_number");

-- Foreign keys
ALTER TABLE "deal_versions"
  ADD CONSTRAINT "deal_versions_tenant_id_fkey"
  FOREIGN KEY ("tenant_id") REFERENCES "tenants"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "deal_versions"
  ADD CONSTRAINT "deal_versions_opportunity_id_fkey"
  FOREIGN KEY ("opportunity_id") REFERENCES "opportunities"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;

-- Index for querying versions by opportunity
CREATE INDEX IF NOT EXISTS "deal_versions_opportunity_id_idx"
  ON "deal_versions"("opportunity_id");

CREATE INDEX IF NOT EXISTS "deal_versions_tenant_id_idx"
  ON "deal_versions"("tenant_id");

-- =============================================================================
-- 7. RLS policies for deal_versions
-- =============================================================================
ALTER TABLE "deal_versions" ENABLE ROW LEVEL SECURITY;

CREATE POLICY "deal_versions_tenant_isolation" ON "deal_versions"
  USING ("tenant_id" = current_setting('app.tenant_id')::UUID);
```

**Step 2: Apply migration to Neon**

```bash
cd /Users/peteroneil/eq-frontend && npx prisma migrate deploy
```

Or apply directly via Neon MCP `run_sql` if preferred.

**Step 3: Verify via Neon MCP**

```sql
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'opportunities' AND column_name LIKE 'meddic_%'
ORDER BY ordinal_position;

SELECT column_name FROM information_schema.columns
WHERE table_name = 'deal_versions' ORDER BY ordinal_position;
```

---

## Task 2: Add Deal Methods to PostgresClient — Failing Tests

**Files:**
- Create: `tests/test_deal_postgres_client.py`

**Step 1: Write failing tests**

```python
"""Tests for Deal dual-write methods on PostgresClient.

Mirrors the test structure of test_postgres_client.py (Phase A) but covers
Deal-specific UPSERT/INSERT operations on opportunities + deal_versions.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from deal_graph.models.deal import Deal, DealStage, DealVersion, MEDDICProfile, OntologyScores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_deal():
    """A Deal with MEDDIC, ontology scores, and embeddings."""
    return Deal(
        tenant_id=UUID('11111111-1111-1111-1111-111111111111'),
        opportunity_id=UUID('22222222-2222-2222-2222-222222222222'),
        deal_ref='deal_2222222222222222',
        name='Acme Cloud Migration',
        stage=DealStage.QUALIFICATION,
        amount=150000.0,
        account_id='33333333-3333-3333-3333-333333333333',
        currency='USD',
        meddic=MEDDICProfile(
            metrics='$150K annual savings',
            metrics_confidence=0.85,
            economic_buyer='Sarah Jones, VP Engineering',
            economic_buyer_confidence=0.90,
            decision_criteria='SOC2 compliance, 99.9% uptime',
            decision_criteria_confidence=0.75,
            champion='James Park, Director of Sales Ops',
            champion_confidence=0.80,
        ),
        ontology_scores=OntologyScores(
            scores={'champion_strength': 2, 'competitive_position': 3},
            confidences={'champion_strength': 0.85, 'competitive_position': 0.90},
            evidence={'champion_strength': 'James actively promoted our solution', 'competitive_position': 'No competing vendors mentioned'},
        ),
        ontology_version='abc123',
        opportunity_summary='Cloud migration for Acme Corp',
        evolution_summary='Initial qualification call confirmed budget authority',
        embedding=[0.1] * 1536,
        embedding_current=[0.2] * 1536,
        version=2,
        confidence=0.88,
        source_interaction_id=UUID('44444444-4444-4444-4444-444444444444'),
    )


@pytest.fixture
def sample_deal_version():
    """A DealVersion snapshot."""
    return DealVersion(
        version_id=UUID('55555555-5555-5555-5555-555555555555'),
        deal_opportunity_id=UUID('22222222-2222-2222-2222-222222222222'),
        tenant_id=UUID('11111111-1111-1111-1111-111111111111'),
        version=1,
        name='Acme Cloud Migration',
        stage=DealStage.PROSPECTING,
        amount=100000.0,
        opportunity_summary='Initial contact with Acme',
        evolution_summary='',
        meddic_metrics=None,
        meddic_completeness=0.0,
        ontology_scores_json=json.dumps({'champion_strength': 1}),
        ontology_completeness=0.1,
        change_summary='Deal progressed from prospecting to qualification',
        changed_fields=['stage', 'amount', 'meddic_metrics'],
        change_source_interaction_id=UUID('44444444-4444-4444-4444-444444444444'),
    )


# ---------------------------------------------------------------------------
# upsert_deal tests
# ---------------------------------------------------------------------------


class TestUpsertDeal:
    """Tests for PostgresClient.upsert_deal()."""

    @pytest.mark.asyncio
    async def test_upsert_deal_executes_sql(self, sample_deal):
        """upsert_deal should execute an INSERT ... ON CONFLICT UPDATE."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        # Must target graph_opportunity_id for conflict resolution
        assert 'graph_opportunity_id' in sql_text
        assert 'ON CONFLICT' in sql_text

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_meddic_fields(self, sample_deal):
        """MEDDIC fields should be flattened to individual columns."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params['meddic_metrics'] == '$150K annual savings'
        assert params['meddic_metrics_confidence'] == 0.85
        assert params['meddic_economic_buyer'] == 'Sarah Jones, VP Engineering'
        assert params['meddic_champion'] == 'James Park, Director of Sales Ops'

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_ontology_dimensions(self, sample_deal):
        """Individual dim_* columns and ontology_scores_json should both be set."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        # Individual columns
        assert params['dim_champion_strength'] == 2
        assert params['dim_champion_strength_confidence'] == 0.85
        assert params['dim_competitive_position'] == 3
        # JSONB (includes evidence)
        scores_json = json.loads(params['ontology_scores_json'])
        assert scores_json['champion_strength']['score'] == 2
        assert scores_json['champion_strength']['evidence'] == 'James actively promoted our solution'

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_embeddings(self, sample_deal):
        """Embeddings should be converted to pgvector format."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params['extraction_embedding'].startswith('[0.1,')
        assert params['extraction_embedding_current'].startswith('[0.2,')

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_summaries_to_ai_columns(self, sample_deal):
        """opportunity_summary → latest_ai_summary, evolution_summary → ai_evolution_summary."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params['latest_ai_summary'] == 'Cloud migration for Acme Corp'
        assert params['ai_evolution_summary'] == 'Initial qualification call confirmed budget authority'

    @pytest.mark.asyncio
    async def test_upsert_deal_does_not_write_trigger_columns(self, sample_deal):
        """Must NOT write to the 8 forecast-trigger-protected columns."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        trigger_cols = ['stage', 'amount', 'close_date', 'deal_status',
                        'forecast_category', 'next_step', 'description', 'lost_reason']
        for col in trigger_cols:
            # Verify no bare column name writes (meddic_* and dim_* contain
            # substrings like 'stage' but are distinct columns)
            # Check the SET clause specifically
            assert f'"{col}"' not in sql_text or f'meddic_{col}' in sql_text or f'dim_{col}' in sql_text


# ---------------------------------------------------------------------------
# insert_deal_version tests
# ---------------------------------------------------------------------------


class TestInsertDealVersion:
    """Tests for PostgresClient.insert_deal_version()."""

    @pytest.mark.asyncio
    async def test_insert_deal_version_executes_sql(self, sample_deal_version):
        """insert_deal_version should INSERT with ON CONFLICT DO NOTHING."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.insert_deal_version(sample_deal_version)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        assert 'deal_versions' in sql_text
        assert 'ON CONFLICT' in sql_text
        assert 'DO NOTHING' in sql_text

    @pytest.mark.asyncio
    async def test_insert_deal_version_maps_changed_fields_to_jsonb(self, sample_deal_version):
        """changed_fields list should be serialized to JSONB."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
        client._engine = mock_engine

        await client.insert_deal_version(sample_deal_version)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        changed = json.loads(params['changed_fields'])
        assert 'stage' in changed
        assert 'amount' in changed


# ---------------------------------------------------------------------------
# persist_deal_full tests
# ---------------------------------------------------------------------------


class TestPersistDealFull:
    """Tests for PostgresClient.persist_deal_full() convenience method."""

    @pytest.mark.asyncio
    async def test_persist_deal_full_writes_deal_and_version(self, sample_deal, sample_deal_version):
        """Should call upsert_deal, insert_deal_version, and link_deal_to_interaction."""
        from action_item_graph.clients.postgres_client import PostgresClient

        interaction_id = uuid4()
        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock()
        client.insert_deal_version = AsyncMock()
        client.link_deal_to_interaction = AsyncMock()

        await client.persist_deal_full(
            deal=sample_deal, version=sample_deal_version, interaction_id=interaction_id,
        )

        client.upsert_deal.assert_called_once_with(sample_deal)
        client.insert_deal_version.assert_called_once_with(sample_deal_version)
        client.link_deal_to_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_deal_full_version_failure_does_not_block_deal(self, sample_deal, sample_deal_version):
        """Version insert failure should not prevent deal upsert (failure isolation)."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock()
        client.insert_deal_version = AsyncMock(side_effect=Exception('version write failed'))
        client.link_deal_to_interaction = AsyncMock()

        # Should NOT raise
        await client.persist_deal_full(deal=sample_deal, version=sample_deal_version)

        client.upsert_deal.assert_called_once()
        client.insert_deal_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_deal_full_link_failure_does_not_block(self, sample_deal):
        """Interaction link failure should not block deal upsert (failure isolation)."""
        from action_item_graph.clients.postgres_client import PostgresClient

        interaction_id = uuid4()
        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock()
        client.insert_deal_version = AsyncMock()
        client.link_deal_to_interaction = AsyncMock(side_effect=Exception('link failed'))

        # Should NOT raise
        await client.persist_deal_full(
            deal=sample_deal, version=None, interaction_id=interaction_id,
        )

        client.upsert_deal.assert_called_once()
        client.link_deal_to_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_deal_full_no_version_no_interaction(self, sample_deal):
        """When version and interaction_id are None, only upsert_deal should be called."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock()
        client.insert_deal_version = AsyncMock()
        client.link_deal_to_interaction = AsyncMock()

        await client.persist_deal_full(deal=sample_deal, version=None)

        client.upsert_deal.assert_called_once()
        client.insert_deal_version.assert_not_called()
        client.link_deal_to_interaction.assert_not_called()
```

**Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_deal_postgres_client.py -v
```

Expected: FAIL — `PostgresClient` has no `upsert_deal`, `insert_deal_version`, or `persist_deal_full` methods.

---

## Task 3: Implement Deal PostgresClient Methods

**Files:**
- Modify: `src/action_item_graph/clients/postgres_client.py`

**Step 1: Add imports**

At the top of the file, add the Deal model imports alongside the existing action item imports:

```python
from deal_graph.models.deal import Deal, DealVersion, OntologyScores
```

**Step 2: Add the `_DEAL_TRIGGER_PROTECTED_COLUMNS` constant**

After `_STATUS_MAP`, add:

```python
# Columns on opportunities table that fire notify_forecast_job() on UPDATE.
# The Deal dual-write MUST NOT write to these columns.
_DEAL_TRIGGER_PROTECTED_COLUMNS = frozenset({
    'stage', 'amount', 'close_date', 'deal_status',
    'forecast_category', 'next_step', 'description', 'lost_reason',
})
```

**Step 3: Add `_ontology_to_jsonb()` helper**

After `_embedding_to_pgvector()`:

```python
def _ontology_to_jsonb(scores: OntologyScores) -> str | None:
    """Serialize OntologyScores to JSONB string with evidence text.

    Output format per dimension:
    { "dimension_id": { "score": int|null, "confidence": float, "evidence": str|null } }
    """
    if not scores.scores:
        return None
    result = {}
    for dim_id, score in scores.scores.items():
        result[dim_id] = {
            'score': score,
            'confidence': scores.confidences.get(dim_id, 0.0),
            'evidence': scores.evidence.get(dim_id),
        }
    return json.dumps(result)
```

**Step 4: Add `_ontology_dim_params()` helper**

```python
def _ontology_dim_params(scores: OntologyScores) -> dict[str, int | float | None]:
    """Flatten OntologyScores to individual dim_* column parameters.

    Returns dict like {'dim_champion_strength': 2, 'dim_champion_strength_confidence': 0.85, ...}
    Dimensions not present in scores get None values.
    """
    from deal_graph.pipeline.merger import TRANSCRIPT_EXTRACTED_DIMENSIONS

    params: dict[str, int | float | None] = {}
    for dim_id in TRANSCRIPT_EXTRACTED_DIMENSIONS:
        params[f'dim_{dim_id}'] = scores.scores.get(dim_id)
        params[f'dim_{dim_id}_confidence'] = scores.confidences.get(dim_id)
    return params
```

**Step 5: Add `upsert_deal()` method**

Inside the `PostgresClient` class, after `persist_action_item_full()`:

```python
async def upsert_deal(
    self,
    deal: Deal,
    source_user_id: str | None = None,
) -> None:
    """Upsert a Deal into the opportunities table.

    Conflict resolution on graph_opportunity_id (unique index).
    Maps Deal model fields to Postgres columns:
    - MEDDIC fields → individual meddic_* columns
    - Ontology scores → individual dim_* columns + ontology_scores_json JSONB
    - Summaries → latest_ai_summary, ai_evolution_summary (existing cols)
    - Embeddings → pgvector literals
    Does NOT write to trigger-protected columns (stage, amount, etc.).

    Note: account_id and opportunity_name are in the INSERT (for new
    pipeline-created rows) but NOT in the ON CONFLICT UPDATE SET — we
    never overwrite CRM-set values on existing rows.

    Args:
        deal: Deal model with extraction data
        source_user_id: Auth0 user ID from Envelope.user_id (provenance)
    """
    dim_params = _ontology_dim_params(deal.ontology_scores)

    # Build the SET clause for dim_* columns dynamically
    dim_col_names = sorted(dim_params.keys())
    dim_insert_cols = ', '.join(f'"{c}"' for c in dim_col_names)
    dim_insert_vals = ', '.join(f':{c}' for c in dim_col_names)
    dim_update_set = ', '.join(f'"{c}" = :{c}' for c in dim_col_names)

    sql = f"""
    INSERT INTO opportunities (
        graph_opportunity_id, tenant_id, account_id, opportunity_name, deal_ref,
        currency, actual_close_date,
        latest_ai_summary, ai_evolution_summary,
        meddic_metrics, meddic_metrics_confidence,
        meddic_economic_buyer, meddic_economic_buyer_confidence,
        meddic_decision_criteria, meddic_decision_criteria_confidence,
        meddic_decision_process, meddic_decision_process_confidence,
        meddic_identified_pain, meddic_identified_pain_confidence,
        meddic_champion, meddic_champion_confidence,
        meddic_completeness, meddic_paper_process, meddic_competition,
        {dim_insert_cols},
        ontology_scores_json, ontology_completeness, ontology_version,
        extraction_embedding, extraction_embedding_current,
        extraction_confidence, extraction_version, source_interaction_id,
        qualification_status, source_user_id,
        ai_workflow_metadata
    ) VALUES (
        :graph_opportunity_id, :tenant_id, :account_id, :opportunity_name, :deal_ref,
        :currency, :actual_close_date,
        :latest_ai_summary, :ai_evolution_summary,
        :meddic_metrics, :meddic_metrics_confidence,
        :meddic_economic_buyer, :meddic_economic_buyer_confidence,
        :meddic_decision_criteria, :meddic_decision_criteria_confidence,
        :meddic_decision_process, :meddic_decision_process_confidence,
        :meddic_identified_pain, :meddic_identified_pain_confidence,
        :meddic_champion, :meddic_champion_confidence,
        :meddic_completeness, :meddic_paper_process, :meddic_competition,
        {dim_insert_vals},
        :ontology_scores_json, :ontology_completeness, :ontology_version,
        :extraction_embedding, :extraction_embedding_current,
        :extraction_confidence, :extraction_version, :source_interaction_id,
        :qualification_status, :source_user_id,
        :ai_workflow_metadata
    )
    ON CONFLICT (graph_opportunity_id) DO UPDATE SET
        deal_ref = :deal_ref,
        latest_ai_summary = :latest_ai_summary,
        ai_evolution_summary = :ai_evolution_summary,
        meddic_metrics = :meddic_metrics,
        meddic_metrics_confidence = :meddic_metrics_confidence,
        meddic_economic_buyer = :meddic_economic_buyer,
        meddic_economic_buyer_confidence = :meddic_economic_buyer_confidence,
        meddic_decision_criteria = :meddic_decision_criteria,
        meddic_decision_criteria_confidence = :meddic_decision_criteria_confidence,
        meddic_decision_process = :meddic_decision_process,
        meddic_decision_process_confidence = :meddic_decision_process_confidence,
        meddic_identified_pain = :meddic_identified_pain,
        meddic_identified_pain_confidence = :meddic_identified_pain_confidence,
        meddic_champion = :meddic_champion,
        meddic_champion_confidence = :meddic_champion_confidence,
        meddic_completeness = :meddic_completeness,
        meddic_paper_process = :meddic_paper_process,
        meddic_competition = :meddic_competition,
        {dim_update_set},
        ontology_scores_json = :ontology_scores_json,
        ontology_completeness = :ontology_completeness,
        ontology_version = :ontology_version,
        extraction_embedding = :extraction_embedding,
        extraction_embedding_current = :extraction_embedding_current,
        extraction_confidence = :extraction_confidence,
        extraction_version = :extraction_version,
        source_interaction_id = :source_interaction_id,
        qualification_status = :qualification_status,
        source_user_id = :source_user_id,
        ai_workflow_metadata = :ai_workflow_metadata,
        updated_at = now()
    """

    params = {
        'graph_opportunity_id': _to_pg_uuid(deal.opportunity_id),
        'tenant_id': _to_pg_uuid(deal.tenant_id),
        'account_id': _to_pg_uuid(deal.account_id) if deal.account_id else None,
        'opportunity_name': deal.name or None,  # INSERT only — don't overwrite CRM name
        'currency': deal.currency or 'USD',  # INSERT only — don't overwrite CRM currency
        'actual_close_date': deal.closed_at,  # INSERT only — don't overwrite CRM close date
        'deal_ref': deal.deal_ref,
        'latest_ai_summary': deal.opportunity_summary or None,
        'ai_evolution_summary': deal.evolution_summary or None,
        # MEDDIC
        'meddic_metrics': deal.meddic.metrics,
        'meddic_metrics_confidence': deal.meddic.metrics_confidence,
        'meddic_economic_buyer': deal.meddic.economic_buyer,
        'meddic_economic_buyer_confidence': deal.meddic.economic_buyer_confidence,
        'meddic_decision_criteria': deal.meddic.decision_criteria,
        'meddic_decision_criteria_confidence': deal.meddic.decision_criteria_confidence,
        'meddic_decision_process': deal.meddic.decision_process,
        'meddic_decision_process_confidence': deal.meddic.decision_process_confidence,
        'meddic_identified_pain': deal.meddic.identified_pain,
        'meddic_identified_pain_confidence': deal.meddic.identified_pain_confidence,
        'meddic_champion': deal.meddic.champion,
        'meddic_champion_confidence': deal.meddic.champion_confidence,
        'meddic_completeness': deal.meddic.completeness_score,
        'meddic_paper_process': deal.meddic.paper_process,
        'meddic_competition': deal.meddic.competition,
        # Ontology
        'ontology_scores_json': _ontology_to_jsonb(deal.ontology_scores),
        'ontology_completeness': deal.ontology_scores.completeness_score,
        'ontology_version': deal.ontology_version,
        # Embeddings (extraction_ prefix to distinguish from CRM embeddings)
        'extraction_embedding': _embedding_to_pgvector(deal.embedding),
        'extraction_embedding_current': _embedding_to_pgvector(deal.embedding_current),
        # Extraction metadata
        'extraction_confidence': deal.confidence,
        'extraction_version': deal.version,
        'source_interaction_id': _to_pg_uuid(deal.source_interaction_id),
        # Future slots
        'qualification_status': deal.qualification_status,
        'source_user_id': source_user_id,  # Auth0 user ID from Envelope.user_id
        # Workflow metadata (JSONB with extraction-specific fields)
        'ai_workflow_metadata': json.dumps({
            'source': 'deal_graph',
            'deal_name': deal.name,
            'stage_assessment': deal.stage.value if isinstance(deal.stage, DealStage) else deal.stage,
            'amount_estimate': deal.amount,
            'currency': deal.currency,
            'deal_ref': deal.deal_ref,
            'expected_close_date': deal.expected_close_date.isoformat() if deal.expected_close_date else None,
            'closed_at': deal.closed_at.isoformat() if deal.closed_at else None,
        }),
        # Dim columns
        **dim_params,
    }

    async with self.engine.begin() as conn:
        await conn.execute(text(sql), params)

    logger.info(
        'postgres_client.upsert_deal',
        opportunity_id=str(deal.opportunity_id),
        tenant_id=str(deal.tenant_id),
    )
```

**Step 6: Add `insert_deal_version()` method**

```python
async def insert_deal_version(self, version: DealVersion) -> None:
    """Insert a DealVersion snapshot into deal_versions table.

    ON CONFLICT (tenant_id, opportunity_id, version_number) DO NOTHING
    to ensure idempotency.
    """
    sql = """
    INSERT INTO deal_versions (
        id, tenant_id, opportunity_id, version_number,
        name, stage, amount,
        opportunity_summary, evolution_summary,
        meddic_metrics, meddic_economic_buyer, meddic_decision_criteria,
        meddic_decision_process, meddic_identified_pain, meddic_champion,
        meddic_completeness,
        ontology_scores_json, ontology_completeness, ontology_version,
        change_summary, changed_fields, change_source_interaction_id,
        created_at, valid_from, valid_until
    ) VALUES (
        :id, :tenant_id, :opportunity_id, :version_number,
        :name, :stage, :amount,
        :opportunity_summary, :evolution_summary,
        :meddic_metrics, :meddic_economic_buyer, :meddic_decision_criteria,
        :meddic_decision_process, :meddic_identified_pain, :meddic_champion,
        :meddic_completeness,
        :ontology_scores_json, :ontology_completeness, :ontology_version,
        :change_summary, :changed_fields, :change_source_interaction_id,
        :created_at, :valid_from, :valid_until
    )
    ON CONFLICT (tenant_id, opportunity_id, version_number) DO NOTHING
    """

    params = {
        'id': _to_pg_uuid(version.version_id),
        'tenant_id': _to_pg_uuid(version.tenant_id),
        'opportunity_id': _to_pg_uuid(version.deal_opportunity_id),
        'version_number': version.version,
        'name': version.name,
        'stage': version.stage.value if isinstance(version.stage, DealStage) else version.stage,
        'amount': version.amount,
        'opportunity_summary': version.opportunity_summary,
        'evolution_summary': version.evolution_summary,
        'meddic_metrics': version.meddic_metrics,
        'meddic_economic_buyer': version.meddic_economic_buyer,
        'meddic_decision_criteria': version.meddic_decision_criteria,
        'meddic_decision_process': version.meddic_decision_process,
        'meddic_identified_pain': version.meddic_identified_pain,
        'meddic_champion': version.meddic_champion,
        'meddic_completeness': version.meddic_completeness,
        'ontology_scores_json': version.ontology_scores_json,
        'ontology_completeness': version.ontology_completeness,
        'ontology_version': version.ontology_version,
        'change_summary': version.change_summary,
        'changed_fields': json.dumps(version.changed_fields),
        'change_source_interaction_id': _to_pg_uuid(version.change_source_interaction_id),
        'created_at': _to_pg_ts(version.created_at),
        'valid_from': _to_pg_ts(version.valid_from),
        'valid_until': _to_pg_ts(version.valid_until),
    }

    async with self.engine.begin() as conn:
        await conn.execute(text(sql), params)

    logger.info(
        'postgres_client.insert_deal_version',
        version_id=str(version.version_id),
        opportunity_id=str(version.deal_opportunity_id),
    )
```

**Step 7: Add `persist_deal_full()` convenience method**

```python
async def link_deal_to_interaction(
    self,
    tenant_id: UUID,
    interaction_id: UUID,
    opportunity_id: UUID,
) -> None:
    """Link an interaction to an opportunity via interaction_links table.

    Preserves the provenance chain: which interactions contributed to a deal.
    ON CONFLICT DO NOTHING for idempotency.
    """
    sql = """
    INSERT INTO interaction_links (tenant_id, interaction_id, entity_type, entity_id)
    VALUES (:tenant_id, :interaction_id, 'opportunity', :entity_id)
    ON CONFLICT DO NOTHING
    """
    async with self.engine.begin() as conn:
        await conn.execute(text(sql), {
            'tenant_id': _to_pg_uuid(tenant_id),
            'interaction_id': _to_pg_uuid(interaction_id),
            'entity_id': _to_pg_uuid(opportunity_id),
        })

async def persist_deal_full(
    self,
    deal: Deal,
    version: DealVersion | None = None,
    interaction_id: UUID | None = None,
    source_user_id: str | None = None,
) -> None:
    """Write a Deal + optional DealVersion to Postgres with failure isolation.

    Calls upsert_deal() first. Then optionally:
    - insert_deal_version() if version provided
    - link_deal_to_interaction() if interaction_id provided
    Each sub-write is in its own try/except.

    Args:
        deal: Deal model with extraction data
        version: DealVersion snapshot (if a version was created during merge)
        interaction_id: Interaction UUID for provenance linking
        source_user_id: Auth0 user ID from Envelope.user_id
    """
    await self.upsert_deal(deal, source_user_id=source_user_id)

    if version is not None:
        try:
            await self.insert_deal_version(version)
        except Exception:
            logger.exception(
                'postgres_client.persist_deal_version_failed',
                opportunity_id=str(deal.opportunity_id),
            )

    if interaction_id is not None:
        try:
            await self.link_deal_to_interaction(
                tenant_id=deal.tenant_id,
                interaction_id=interaction_id,
                opportunity_id=deal.opportunity_id,
            )
        except Exception:
            logger.exception(
                'postgres_client.link_deal_interaction_failed',
                opportunity_id=str(deal.opportunity_id),
                interaction_id=str(interaction_id),
            )
```

**Step 8: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_deal_postgres_client.py -v
```

Expected: All tests PASS.

**Step 9: Run full test suite for regressions**

```bash
uv run python -m pytest tests/ -x -q
```

Expected: 401+ tests pass, no regressions.

**Step 10: Commit**

```bash
git add src/action_item_graph/clients/postgres_client.py tests/test_deal_postgres_client.py
git commit -m "feat: add Deal dual-write methods to PostgresClient (Phase B)"
```

---

## Task 4: Wire PostgresClient into DealPipeline — Failing Tests

**Files:**
- Create: `tests/test_deal_pipeline_postgres.py`

**Step 1: Write failing tests**

```python
"""Tests for DealPipeline Postgres dual-write wiring.

Verifies that DealPipeline accepts an optional postgres_client and calls
persist_deal_full() after Neo4j merging, with failure isolation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from action_item_graph.models.envelope import (
    ContentFormat,
    ContentPayload,
    EnvelopeV1,
    InteractionType,
    SourceType,
)
from deal_graph.models.deal import Deal, DealStage, DealVersion
from deal_graph.pipeline.pipeline import DealPipeline


@pytest.fixture
def mock_neo4j():
    client = AsyncMock()
    client.driver = MagicMock()
    return client


@pytest.fixture
def mock_openai():
    return AsyncMock()


@pytest.fixture
def mock_postgres():
    pg = AsyncMock()
    pg.persist_deal_full = AsyncMock()
    return pg


class TestDealPipelinePostgresWiring:
    """DealPipeline should accept and use an optional postgres_client."""

    def test_init_accepts_postgres_client(self, mock_neo4j, mock_openai, mock_postgres):
        """DealPipeline.__init__ should accept postgres_client parameter."""
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
            postgres_client=mock_postgres,
        )
        assert pipeline.postgres is mock_postgres

    def test_init_postgres_defaults_to_none(self, mock_neo4j, mock_openai):
        """postgres_client should default to None (backward compatible)."""
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
        )
        assert pipeline.postgres is None


class TestDealPipelineDualWrite:
    """DealPipeline._dual_write_postgres() tests."""

    @pytest.mark.asyncio
    async def test_dual_write_skipped_when_no_postgres(self, mock_neo4j, mock_openai):
        """When postgres_client is None, dual write should be a no-op."""
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
        )
        # Should not raise
        await pipeline._dual_write_postgres([], str(uuid4()))

    @pytest.mark.asyncio
    async def test_dual_write_failure_does_not_raise(self, mock_neo4j, mock_openai, mock_postgres):
        """Postgres failure should be logged but not propagated."""
        mock_postgres.persist_deal_full = AsyncMock(side_effect=Exception('pg down'))
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
            postgres_client=mock_postgres,
        )
        # Should not raise
        await pipeline._dual_write_postgres(
            [MagicMock(opportunity_id=str(uuid4()), action='created', was_new=True, version_created=False)],
            str(uuid4()),
        )
```

**Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_deal_pipeline_postgres.py -v
```

Expected: FAIL — DealPipeline doesn't accept `postgres_client` yet.

---

## Task 5: Implement DealPipeline Dual-Write Wiring

**Files:**
- Modify: `src/deal_graph/pipeline/pipeline.py`

**Step 1: Add import**

At the top of the file, add:

```python
from action_item_graph.clients.postgres_client import PostgresClient
```

**Step 2: Add `postgres_client` parameter to `__init__`**

Modify `DealPipeline.__init__()`:

```python
def __init__(
    self,
    neo4j_client: DealNeo4jClient,
    openai_client: OpenAIClient,
    postgres_client: PostgresClient | None = None,
):
    self.extractor = DealExtractor(openai_client)
    self.matcher = DealMatcher(neo4j_client, openai_client)
    self.merger = DealMerger(neo4j_client, openai_client)
    self.repository = DealRepository(neo4j_client)
    self.postgres = postgres_client
```

**Step 3: Add `_dual_write_postgres()` method**

After `_enrich_interaction()`:

```python
async def _dual_write_postgres(
    self,
    merge_results: list[DealMergeResult],
    interaction_id: str | None,
    source_user_id: str | None = None,
) -> None:
    """Write Deals to Postgres after Neo4j merge completes.

    Failure-isolated: any exception is logged and swallowed.
    Neo4j (source of truth) is never affected.

    Args:
        merge_results: Results from match+merge stage
        interaction_id: Interaction that triggered processing
        source_user_id: Auth0 user ID from Envelope.user_id
    """
    if self.postgres is None:
        return

    for merge_result in merge_results:
        try:
            # Fetch post-merge Deal state from Neo4j
            deal_node = await self.repository.get_deal(
                tenant_id=UUID(merge_result.details.get('tenant_id', '')),
                opportunity_id=merge_result.opportunity_id,
            )
            if deal_node is None:
                logger.warning(
                    'deal_pipeline.postgres_dual_write.deal_not_found',
                    opportunity_id=merge_result.opportunity_id,
                )
                continue

            deal = _neo4j_node_to_deal(deal_node)

            # Build version if one was created during merge
            version = None
            if merge_result.version_created:
                version = await self._fetch_latest_version(
                    tenant_id=deal.tenant_id,
                    opportunity_id=merge_result.opportunity_id,
                )

            await self.postgres.persist_deal_full(
                deal=deal,
                version=version,
                interaction_id=UUID(interaction_id) if interaction_id else None,
                source_user_id=source_user_id,
            )

        except Exception:
            logger.exception(
                'deal_pipeline.postgres_dual_write.deal_failed',
                opportunity_id=merge_result.opportunity_id,
            )
```

**Step 4: Add `_neo4j_node_to_deal()` helper**

Before the `DealPipeline` class:

```python
def _neo4j_node_to_deal(node: dict[str, Any]) -> Deal:
    """Convert a Neo4j Deal node dict to a Deal model instance."""
    from deal_graph.models.deal import MEDDICProfile, OntologyScores
    from deal_graph.pipeline.merger import TRANSCRIPT_EXTRACTED_DIMENSIONS

    meddic = MEDDICProfile(
        metrics=node.get('meddic_metrics'),
        metrics_confidence=node.get('meddic_metrics_confidence', 0.0),
        economic_buyer=node.get('meddic_economic_buyer'),
        economic_buyer_confidence=node.get('meddic_economic_buyer_confidence', 0.0),
        decision_criteria=node.get('meddic_decision_criteria'),
        decision_criteria_confidence=node.get('meddic_decision_criteria_confidence', 0.0),
        decision_process=node.get('meddic_decision_process'),
        decision_process_confidence=node.get('meddic_decision_process_confidence', 0.0),
        identified_pain=node.get('meddic_identified_pain'),
        identified_pain_confidence=node.get('meddic_identified_pain_confidence', 0.0),
        champion=node.get('meddic_champion'),
        champion_confidence=node.get('meddic_champion_confidence', 0.0),
        paper_process=node.get('meddic_paper_process'),
        competition=node.get('meddic_competition'),
    )

    scores = {}
    confidences = {}
    for dim_id in TRANSCRIPT_EXTRACTED_DIMENSIONS:
        score = node.get(f'dim_{dim_id}')
        if score is not None:
            scores[dim_id] = score
        conf = node.get(f'dim_{dim_id}_confidence')
        if conf is not None:
            confidences[dim_id] = conf

    ontology = OntologyScores(scores=scores, confidences=confidences)

    return Deal(
        tenant_id=UUID(node['tenant_id']),
        opportunity_id=UUID(node['opportunity_id']),
        deal_ref=node.get('deal_ref'),
        name=node.get('name', ''),
        stage=node.get('stage', 'prospecting'),
        amount=node.get('amount'),
        account_id=node.get('account_id'),
        currency=node.get('currency', 'USD'),
        meddic=meddic,
        ontology_scores=ontology,
        ontology_version=node.get('ontology_version'),
        opportunity_summary=node.get('opportunity_summary', ''),
        evolution_summary=node.get('evolution_summary', ''),
        embedding=node.get('embedding'),
        embedding_current=node.get('embedding_current'),
        version=node.get('version', 1),
        confidence=node.get('confidence', 1.0),
        source_interaction_id=UUID(node['source_interaction_id']) if node.get('source_interaction_id') else None,
    )
```

**Step 5: Add `_fetch_latest_version()` helper**

```python
async def _fetch_latest_version(
    self,
    tenant_id: UUID,
    opportunity_id: str,
) -> DealVersion | None:
    """Fetch the most recent DealVersion from Neo4j for a given Deal."""
    version_node = await self.repository.get_latest_version(
        tenant_id=tenant_id,
        opportunity_id=opportunity_id,
    )
    if version_node is None:
        return None
    return _neo4j_node_to_deal_version(version_node)
```

Note: `DealRepository.get_latest_version()` may need to be added if it doesn't exist. Check the repository first.

**Step 6: Call `_dual_write_postgres()` in `process_envelope()`**

In `process_envelope()`, after Step 6 (enrich interaction) and before Step 7 (finalize), add:

```python
# ------------------------------------------------------------------
# Step 6b: Dual-write to Postgres (failure-isolated)
# ------------------------------------------------------------------
try:
    await self._dual_write_postgres(
        result.merge_results,
        interaction_id,
        source_user_id=getattr(envelope, 'user_id', None),
    )
except Exception:
    logger.exception('deal_pipeline.postgres_dual_write_failed')
```

**Step 7: Run tests**

```bash
uv run python -m pytest tests/test_deal_pipeline_postgres.py tests/test_deal_postgres_client.py -v
```

Expected: All tests PASS.

**Step 8: Run full test suite**

```bash
uv run python -m pytest tests/ -x -q
```

Expected: 401+ tests pass.

**Step 9: Commit**

```bash
git add src/deal_graph/pipeline/pipeline.py tests/test_deal_pipeline_postgres.py
git commit -m "feat: wire PostgresClient into DealPipeline with failure isolation (Phase B)"
```

---

## Task 6: Update Dispatcher + Live E2E Script

**Files:**
- Modify: `src/dispatcher/dispatcher.py` (no change needed — DealPipeline is passed pre-built)
- Modify: `scripts/run_live_e2e.py`

**Step 1: Check dispatcher wiring**

The dispatcher receives pre-built pipeline instances. No changes needed — the caller (E2E script, API service) is responsible for passing `postgres_client` to `DealPipeline`.

**Step 2: Update `run_live_e2e.py`**

Find where `DealPipeline` is constructed and pass `postgres_client`:

```python
# Existing (approximate):
deal_pipeline = DealPipeline(neo4j_client=deal_neo4j, openai_client=openai_client)

# Change to:
deal_pipeline = DealPipeline(
    neo4j_client=deal_neo4j,
    openai_client=openai_client,
    postgres_client=postgres_client,  # Same instance used by ActionItemPipeline
)
```

**Step 3: Add Postgres Deal verification after E2E run**

After the existing Postgres verification section (action items), add:

```python
# --- Deal dual-write verification ---
if postgres_client:
    async with postgres_client.engine.begin() as conn:
        # Count deals with graph cross-reference
        result = await conn.execute(text(
            "SELECT COUNT(*) FROM opportunities WHERE graph_opportunity_id IS NOT NULL"
        ))
        pg_deal_count = result.scalar()

        # Count deal versions
        result = await conn.execute(text(
            "SELECT COUNT(*) FROM deal_versions"
        ))
        pg_version_count = result.scalar()

        print(f"\n--- Postgres Deal Verification ---")
        print(f"Opportunities with graph_opportunity_id: {pg_deal_count}")
        print(f"Deal versions: {pg_version_count}")
```

**Step 4: Commit**

```bash
git add scripts/run_live_e2e.py
git commit -m "feat: wire Deal dual-write into live E2E script (Phase B)"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `docs/SMOKE_TEST_GUIDE.md`

**Step 1: Add Deal dual-write verification section**

After the existing "Postgres Dual-Write Verification" section, add or extend with:

```markdown
### Deal Dual-Write Verification

After E2E run, verify Deal data landed in Postgres:

```sql
-- Count deals with graph cross-reference
SELECT COUNT(*) FROM opportunities WHERE graph_opportunity_id IS NOT NULL;

-- Check MEDDIC fields populated
SELECT graph_opportunity_id, meddic_completeness, meddic_metrics IS NOT NULL as has_metrics
FROM opportunities WHERE graph_opportunity_id IS NOT NULL;

-- Count deal versions
SELECT COUNT(*) FROM deal_versions;

-- Verify ontology dimensions
SELECT graph_opportunity_id, ontology_completeness, ontology_scores_json IS NOT NULL as has_scores
FROM opportunities WHERE graph_opportunity_id IS NOT NULL;
```
```

**Step 2: Commit**

```bash
git add docs/SMOKE_TEST_GUIDE.md
git commit -m "docs: add Deal dual-write verification to Smoke Test Guide (Phase B)"
```

---

## Task 8: Live E2E Verification

**Step 1: Run live E2E**

```bash
cd /Users/peteroneil/action-item-graph
uv run python scripts/run_live_e2e.py
```

**Step 2: Verify via Neon MCP**

```sql
-- Deals landed
SELECT COUNT(*) FROM opportunities WHERE graph_opportunity_id IS NOT NULL;

-- MEDDIC populated
SELECT graph_opportunity_id, meddic_completeness, meddic_metrics
FROM opportunities WHERE graph_opportunity_id IS NOT NULL LIMIT 5;

-- Ontology dimensions
SELECT graph_opportunity_id, ontology_completeness,
       dim_champion_strength, dim_competitive_position
FROM opportunities WHERE graph_opportunity_id IS NOT NULL LIMIT 5;

-- Deal versions exist
SELECT COUNT(*) FROM deal_versions;

-- Spot check a version
SELECT opportunity_id, version_number, change_summary
FROM deal_versions LIMIT 3;
```

**Step 3: Confirm counts match Neo4j**

Neo4j Deal count should match `opportunities WHERE graph_opportunity_id IS NOT NULL` count.

---

## Task 9: Final Commit & Push

**Step 1: Run full test suite**

```bash
uv run python -m pytest tests/ -x -q
```

Expected: All tests pass (401+ original + ~15 new = 416+).

**Step 2: Commit any remaining changes**

```bash
git add -A
git commit -m "feat: Phase B Deal dual-write complete — MEDDIC, ontology, embeddings to Postgres"
```

**Step 3: Push to main**

```bash
git push origin main
```

---

## Verification Checklist

- [ ] Prisma migration applied to Neon (57+ new columns on opportunities, deal_versions table)
- [ ] `upsert_deal()`, `insert_deal_version()`, `link_deal_to_interaction()`, `persist_deal_full()` methods on PostgresClient
- [ ] DealPipeline accepts optional `postgres_client` (backward compatible)
- [ ] Dual-write stage runs after Neo4j merge, failure-isolated
- [ ] No writes to trigger-protected columns (stage, amount, close_date, etc.)
- [ ] No writes to forecast-owned columns (latest_forecast_*, forecast_count, etc.)
- [ ] `deal_ref`, `qualification_status`, `source_user_id` columns included
- [ ] Embedding columns use `extraction_` prefix (`extraction_embedding`, `extraction_embedding_current`)
- [ ] Entity linking: `interaction_links` populated for opportunity ↔ interaction provenance
- [ ] Unit tests for all new PostgresClient methods
- [ ] Integration tests for DealPipeline dual-write wiring
- [ ] Live E2E passes with Postgres Deal verification
- [ ] Neon MCP confirms MEDDIC + ontology + embeddings landed
- [ ] Full test suite passes (416+ tests)
- [ ] Smoke Test Guide updated
- [ ] Pushed to main

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Forecast trigger fires on our writes | We only write to non-trigger columns. Verified by test. |
| New columns break eq-frontend queries | All new columns are nullable, no Prisma SDL changes needed |
| DealPipeline breaks without Postgres | `postgres_client` defaults to None; all existing tests unaffected |
| Ontology dimensions change | JSONB (`ontology_scores_json`) is always authoritative; individual columns are convenience |
| Connection pooling conflict with ActionItemPipeline | Both share same `PostgresClient` instance with 5+5 pool |
| `updated_at` race with forecast pipeline | Both pipelines write `updated_at = now()`. Last writer wins — acceptable for a timestamp. Not trigger-protected. |
| `ontology_completeness` write conflict | Forecast pipeline never writes it (confirmed via code audit). We are the sole writer. |

---

## Follow-Up Tasks (Not in This Plan)

These items from the original spec are deferred to Phase C or separate PRs:

### eq-frontend: Prisma Schema Updates
- Add `DealVersion` Prisma model
- Add `dealVersions` relation on `Opportunity` model
- These are not required for the dual-write to function (raw SQL migration creates the table)

### eq-frontend: View Updates
- Extend `opportunity_pipeline` view: add `meddic_completeness`, `extraction_version`
- Extend `rpt_account_health`: add deal extraction metrics
- Extend `account_360`: add `meddic_completeness` for deal health
- Create `action_item_topics_dashboard` view (agent-friendly)

### Reconciliation (Phase C)
- Periodic Neo4j vs Postgres count comparison per tenant
- Replay missed writes for drift detection
- Backfill script for existing Neo4j Deals → Postgres
