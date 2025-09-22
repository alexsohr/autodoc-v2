"""Health check endpoints"""

import os
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, status
from pydantic import BaseModel

# Import database manager for health checks
try:
    from src.utils.database import db_manager
except ImportError:
    db_manager = None

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    timestamp: datetime
    version: str
    python_version: str
    environment: str


@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="2.0.0",
        python_version=sys.version.split()[0],
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness check for Kubernetes/ECS"""
    checks = {"database": False, "timestamp": datetime.now(timezone.utc)}

    # Check database connection
    if db_manager:
        try:
            checks["database"] = await db_manager.health_check()
        except Exception:
            checks["database"] = False

    # Determine overall status
    all_ready = all([checks["database"]])

    status_code = (
        status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": checks["timestamp"],
    }


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """Liveness check for Kubernetes/ECS"""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc)}
