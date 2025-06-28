# modules/llm_setup.py
import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def get_llm():
    """Initializes and returns the Gemini LLM with fallback handling."""
    try:
        return LLM(
            model="gemini/gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return None
