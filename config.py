import os
from pathlib import Path
from dotenv import load_dotenv 

BASE_DIR = Path(__file__).resolve().parent 
load_dotenv(BASE_DIR / ".env")

def get_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Add OPENAI_API_KEY to your .env file to use this mode")
    from openai import OpenAI

    return OpenAI(timeout=60, max_retries=1)

"""
timeout = 60 sets a 60-second network timeout for API requests.
max_retries = 1 lets the client try one more time if a request fails with a retryable error. That means at most two attempts for one request.
"""




