"""Entry point — loads environment and starts the FastAPI server.

All route and app configuration lives in src/api/. This file is intentionally
minimal: it only bootstraps the environment and hands off to the app factory.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from src.api.app import create_app  # noqa: E402 (load_dotenv must run first)

app = create_app()

if __name__ == "__main__":
    import uvicorn
    adress = os.getenv("IP_ADDRESS")
    uvicorn.run("main:app", host=adress, reload=True)
