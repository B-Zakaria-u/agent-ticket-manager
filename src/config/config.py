from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

domain_name = os.getenv("DOMAIN_NAME")

origins = [
    "http://localhost",
    f"http://{domain_name}",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://[IP_ADDRESS]",
    "http://0.0.0.0"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)