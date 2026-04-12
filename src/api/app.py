"""FastAPI application factory (SRP: wires routes, nothing else)."""
from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.workflow import router as workflow_router
from src.api.test_endpoints.routes.testing import router as testing_router


def create_app() -> FastAPI:
    """
    Construct and return the configured FastAPI application.

    Responsibilities
    ----------------
    - Register all route modules.
    - Set metadata (title, description, version).
    - This function is the *only* place that knows FastAPI exists.
    """
    app = FastAPI(
        title="AI Dev System",
        description=(
            "Multi-agent AI system that takes a development ticket, writes a "
            "technical spec, implements code, runs tests in a Docker sandbox, "
            "and opens a Pull/Merge Request."
        ),
        version="1.0.0",
    )

    app.include_router(health_router)
    app.include_router(workflow_router)

    # Testing routes present inside the test_endpoints package
    app.include_router(testing_router)

    return app
