# Production RLS invariant — tenant-scoped Postgres writes

_Added for the prod repoint (EQ-123, 2026-06-11). Companion to the identical
invariant docs in live-transcription-fastapi and eq-email-pipeline._

## The invariant

**Every Postgres statement that reads or writes a tenant-owned table MUST run
inside a transaction whose FIRST statement sets the tenant GUC:**

```sql
SELECT set_config('app.tenant_id', '<tenant-uuid>', true);
```

In this service that means: never call `engine.begin()` directly for a
tenant-scoped statement — use `PostgresClient.scoped_begin(tenant_id)`
(`src/action_item_graph/clients/postgres_client.py`), which opens the
transaction and sets the GUC before yielding the connection. All ten
`PostgresClient` writers and the `agent_action_outbox` write in
`pipeline.py` go through it.

## Why

- In **production** the service connects as a least-privilege, NOBYPASSRLS
  role. Several tables this service writes carry tenant-isolation row-level
  security policies keyed on `current_setting('app.tenant_id')` — at the time
  of writing: `opportunities` (FORCE), `deal_versions`,
  `action_item_versions`, `action_item_topics`, `action_item_topic_versions`,
  `action_item_topic_memberships`, `action_item_owners`. Without the GUC those
  statements **fail closed** — and because the dual-write steps are
  deliberately failure-isolated (they swallow exceptions into warnings), the
  failure would be **silent**: Neo4j keeps writing, Postgres silently stops.
- In **dev** the service connects as the database owner (BYPASSRLS), so the
  GUC is a harmless no-op. The code is identical in both environments by
  design (the prod-mirrors-dev parity principle).
- `set_config(..., is_local := true)` scopes the setting to the enclosing
  transaction only; nothing leaks across pooled connections.

## Rules for future changes

1. New tenant-scoped statement → run it via `scoped_begin(tenant_id)`. Do not
   add bare `engine.begin()` writes.
2. Never weaken this to "set once on the connection" — the pool reuses
   connections across tenants.
3. The authoritative list of which tables are strict is the live database's
   `pg_policies` catalog, not this document — when adding a table, check it.
4. The DBOS system database (`eq_aig_dbos_sys`) is intentionally exempt: it is
   a separate, non-tenant database holding workflow state for all tenants by
   design. RLS is not enabled there.
