"""Regression tests for the GAP-067 class: SQLAlchemy ``text()`` does NOT
recognize ``:name`` immediately followed by ``::cast`` as a bind parameter —
the token passes through as literal SQL, PostgreSQL raises ``syntax error at
or near ":"``, and a fail-open writer swallows the failure silently.

Two guards:

1. ``test_no_param_cast_tokens_in_source`` — the dangerous ``:param::cast``
   spelling must never appear under ``src/``. Write ``CAST(:param AS type)``.

2. ``test_every_supplied_parameter_is_a_recognized_bind`` — for every
   statically extractable ``conn.execute(<sql>, <params>)`` pair in ``src/``,
   every supplied parameter key must be recognized by ``text()`` as a bind
   parameter. A dropped bind means the statement cannot execute at all, so
   this is the rows-not-statuses guarantee at the unit level.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import text

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

# `:name` immediately followed by `::` — the exact shape text() fails to bind.
PARAM_CAST_RE = re.compile(r":[A-Za-z_][A-Za-z0-9_]*::")

_FUNC_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _python_sources() -> list[Path]:
    files = sorted(SRC_ROOT.rglob("*.py"))
    assert files, f"no python sources found under {SRC_ROOT}"
    return files


def test_no_param_cast_tokens_in_source() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if PARAM_CAST_RE.search(line):
                offenders.append(f"{path.relative_to(SRC_ROOT.parent)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "':param::cast' binds are silently dropped by SQLAlchemy text() — "
        "use CAST(:param AS type) instead:\n" + "\n".join(offenders)
    )


def _const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _static_sql_text(node: ast.expr | None) -> str | None:
    """SQL text statically visible in *node*: a string literal verbatim, or an
    f-string's constant fragments joined (interpolated segments dropped — any
    binds they generate at runtime are out of static reach, but the literal
    binds in the static fragments remain checkable; upsert_deal's shape)."""
    s = _const_str(node)
    if s is not None:
        return s
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        if parts:
            return "".join(parts)
    return None


def _text_call_sql(node: ast.expr) -> str | None:
    """Return the SQL string if *node* is ``text(<string literal>)``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "text"
        and node.args
    ):
        return _static_sql_text(node.args[0])
    return None


def _dict_const_keys(node: ast.expr) -> set[str] | None:
    """The LITERAL constant keys of a dict literal; ``**spread`` entries are
    skipped (their keys are dynamic — out of static reach), computed keys
    bail out. Checking only the literal keys keeps the assertion sound: every
    key we can see supplied must still be a recognized bind."""
    if not isinstance(node, ast.Dict):
        return None
    keys: set[str] = set()
    for k in node.keys:
        if k is None:  # **spread entry
            continue
        s = _const_str(k)
        if s is None:
            return None
        keys.add(s)
    return keys or None


def _walk_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Walk *node* in depth-first source order WITHOUT descending into nested
    function scopes (each function is analyzed as its own scope)."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, _FUNC_SCOPES):
            yield from _walk_scope(child)


def _scope_maps(scope: ast.AST) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Per-scope candidate maps: name -> {candidate SQL strings} and
    name -> {supplied dict keys} (union across branches; subscript-augmented).
    Candidate SETS make branch-assigned names safe: a parameter only fails if
    NO candidate SQL recognizes it."""
    sql_by_name: dict[str, set[str]] = {}
    keys_by_name: dict[str, set[str]] = {}
    for node in _walk_scope(scope):
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        if target is not None and isinstance(node, (ast.Assign, ast.AnnAssign)):
            assert node.value is not None
            if isinstance(target, ast.Name):
                sql = _text_call_sql(node.value) or _static_sql_text(node.value)
                if sql is not None and ":" in sql:
                    sql_by_name.setdefault(target.id, set()).add(sql)
                keys = _dict_const_keys(node.value)
                if keys is not None:
                    keys_by_name.setdefault(target.id, set()).update(keys)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in keys_by_name
            ):
                key = _const_str(target.slice)
                if key is not None:
                    keys_by_name[target.value.id].add(key)
    return sql_by_name, keys_by_name


def _collect_pairs(tree: ast.Module) -> list[tuple[set[str], set[str]]]:
    """Extract (candidate SQLs, supplied param keys) for every resolvable
    ``.execute(sql, params)`` call. Covers inline ``execute(text("..."),
    {...})`` as well as the postgres-client shape (``sql``/``params`` locals,
    ``execute(text(sql), params)``); module/class-level constants (e.g.
    ``SET_TENANT_SCOPE_SQL``) resolve via the module scope as a fallback."""
    module_sql, module_keys = _scope_maps(tree)
    pairs: list[tuple[set[str], set[str]]] = []

    scopes: list[ast.AST] = [tree]
    scopes += [n for n in ast.walk(tree) if isinstance(n, _FUNC_SCOPES)]

    for scope in scopes:
        if scope is tree:
            sql_by_name, keys_by_name = module_sql, module_keys
        else:
            local_sql, local_keys = _scope_maps(scope)
            sql_by_name = {**module_sql, **local_sql}
            keys_by_name = {**module_keys, **local_keys}

        for node in _walk_scope(scope):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "execute" or len(node.args) < 2:
                continue
            sql_arg, params_arg = node.args[0], node.args[1]

            sqls: set[str] = set()
            inline = _text_call_sql(sql_arg)
            if inline is not None:
                sqls = {inline}
            elif (
                isinstance(sql_arg, ast.Call)
                and isinstance(sql_arg.func, ast.Name)
                and sql_arg.func.id == "text"
                and sql_arg.args
                and isinstance(sql_arg.args[0], ast.Name)
            ):
                sqls = sql_by_name.get(sql_arg.args[0].id, set())
            elif isinstance(sql_arg, ast.Name):
                sqls = sql_by_name.get(sql_arg.id, set())

            keys = _dict_const_keys(params_arg)
            if keys is None and isinstance(params_arg, ast.Name):
                keys = keys_by_name.get(params_arg.id)

            if sqls and keys:
                pairs.append((sqls, keys))

    return pairs


def test_every_supplied_parameter_is_a_recognized_bind() -> None:
    all_pairs: list[tuple[Path, set[str], set[str]]] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for sqls, keys in _collect_pairs(tree):
            all_pairs.append((path, sqls, keys))

    # Extraction sanity: the matcher must keep seeing real writer SQL — in
    # particular the agent_action_outbox INSERT (the original GAP-067 site).
    assert len(all_pairs) >= 5, (
        f"only {len(all_pairs)} execute(sql, params) pairs extracted — "
        "the AST matcher no longer sees the writers; fix the test, do not delete it"
    )
    assert any(
        {"account_id", "interaction_id", "dedup_key"} <= keys for _, _, keys in all_pairs
    ), "the agent_action_outbox INSERT is no longer covered by the extractor"
    assert any(
        {"graph_opportunity_id", "meddic_metrics"} <= keys for _, _, keys in all_pairs
    ), "the upsert_deal writer (f-string SQL + **spread params) is no longer covered by the extractor"

    failures: list[str] = []
    for path, sqls, keys in all_pairs:
        recognized: set[str] = set()
        for sql in sqls:
            recognized |= set(text(sql)._bindparams.keys())
        dropped = keys - recognized
        if dropped:
            failures.append(
                f"{path.relative_to(SRC_ROOT.parent)}: supplied parameter(s) "
                f"{sorted(dropped)} are NOT recognized as binds by text() — "
                "PostgreSQL will see a literal ':' (GAP-067 class)"
            )
    assert not failures, "\n".join(failures)
