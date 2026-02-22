"""POST /process — validate envelope and dispatch through pipelines."""

import structlog
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from action_item_graph.models.envelope import EnvelopeV1
from action_item_graph.pipeline.pipeline import ActionItemPipeline
from deal_graph.pipeline.pipeline import DealPipeline
from dispatcher.dispatcher import EnvelopeDispatcher

from ..auth import verify_worker_token

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/process")
async def process_envelope(
    envelope_data: dict[str, Any],
    request: Request,
    _auth: None = Depends(verify_worker_token),
):
    """Validate an EnvelopeV1 payload and dispatch through both pipelines."""
    # Validate envelope
    try:
        envelope = EnvelopeV1.model_validate(envelope_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    log = logger.bind(
        tenant_id=str(envelope.tenant_id),
        interaction_type=envelope.interaction_type.value,
        interaction_id=str(envelope.interaction_id) if envelope.interaction_id else None,
    )
    log.info("process.received")

    # Build pipelines from shared clients
    try:
        action_item_pipeline = ActionItemPipeline(
            openai_client=request.app.state.openai,
            neo4j_client=request.app.state.neo4j,
        )
        deal_pipeline = DealPipeline(
            neo4j_client=request.app.state.deal_neo4j,
            openai_client=request.app.state.openai,
        )
        dispatcher = EnvelopeDispatcher(
            action_item_pipeline=action_item_pipeline,
            deal_pipeline=deal_pipeline,
        )
        result = await dispatcher.dispatch(envelope)
    except Exception as e:
        log.error("process.failed", error=str(e), error_type=type(e).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "overall_success": False},
        )

    log.info(
        "process.complete",
        overall_success=result.overall_success,
        dispatch_time_ms=result.dispatch_time_ms,
    )
    return result.to_dict()
