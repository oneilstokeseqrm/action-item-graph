"""Health check endpoint."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Check Neo4j connectivity."""
    try:
        await request.app.state.neo4j.verify_connectivity()
        return {"status": "ok"}
    except Exception:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"status": "unhealthy"})
