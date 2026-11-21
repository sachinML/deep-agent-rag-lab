
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv(Path(__file__).resolve().parent / ".env")

langfuse = get_client()

print("Public key loaded:", bool(os.getenv("LANGFUSE_PUBLIC_KEY")))
print("Secret key loaded:", bool(os.getenv("LANGFUSE_SECRET_KEY")))

print("Langfuse authentication:", langfuse.auth_check())
