# Filename: modules/config.py

import os
from dotenv import load_dotenv

load_dotenv()

def get_config():
    """
    Returns a dictionary of configuration values used across the chatbot system.
    Pulls values from environment variables where appropriate.
    """
    return {
        # API Keys
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),

        # LLM settings
        "llm_model": os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash"),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
        "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "llm_timeout": int(os.getenv("LLM_TIMEOUT", "30")),
        "llm_max_retries": int(os.getenv("LLM_MAX_RETRIES", "2")),

        # Tool model settings
        "crisis_model": os.getenv("CRISIS_MODEL", "sentinet/suicidality"),

        # Questionnaire path
        "questionnaire_file": os.getenv("QUESTIONNAIRE_FILE", "questionnaire.json"),

        # Default profile for anonymous or test users
        "default_user_profile": {
            "id": "anon_user",
            "age": 30,
            "location": "Thimphu",
            "history": "Some prior anxiety episodes",
            "preferences": "Prefers meditation"
        }
    }
