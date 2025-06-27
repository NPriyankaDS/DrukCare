import streamlit as st
import json

# --- Load Questionnaires from JSON ---
QUESTIONNAIRES_FILE = "questionnaires.json"

@st.cache_data
def load_questionnaires():
    try:
        # In a real deployed environment, ensure this file is accessible
        with open(QUESTIONNAIRES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: {QUESTIONNAIRES_FILE} not found. Please ensure it's in the same directory.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {QUESTIONNAIRES_FILE}. Check file format.")
        return {}

QUESTIONS = load_questionnaires()


# --- Scoring Logic Functions ---

def score_phq9(answers):
    """Calculates the PHQ-9 score based on user answers."""
    score_map = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
    }
    total_score = 0
    # Questions are 1-indexed in the JSON list, skip the first item (instruction)
    for i in range(1, 10):
        question_text_prefix = QUESTIONS["PHQ-9"][i].split('. ', 1)[0] # e.g., "1" from "1. Little interest..."
        # Find the answer using the question's text or a derived key
        # In conversational flow, `answers` will have keys like "Q1: Little interest..."
        # We need to ensure matching. Let's assume the question as posed to user is the key.
        found_question_key = None
        for key in answers.keys():
            if QUESTIONS["PHQ-9"][i] in key: # Check if the official question string is part of the key
                found_question_key = key
                break
        
        if found_question_key:
            answer = answers.get(found_question_key, "").strip()
            total_score += score_map.get(answer, 0) # Default to 0 if answer not found or invalid
    return total_score

def score_gad7(answers):
    """Calculates the GAD-7 score based on user answers."""
    score_map = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
    }
    total_score = 0
    for i in range(1, 8):
        question_text_prefix = QUESTIONS["GAD-7"][i].split('. ', 1)[0]
        found_question_key = None
        for key in answers.keys():
            if QUESTIONS["GAD-7"][i] in key:
                found_question_key = key
                break
        
        if found_question_key:
            answer = answers.get(found_question_key, "").strip()
            total_score += score_map.get(answer, 0)
    return total_score

def score_dast10(answers):
    """Calculates the DAST-10 score based on user answers."""
    score_map = {
        "Yes": 1,
        "No": 0
    }
    total_score = 0
    for i in range(1, 11):
        question_text_prefix = QUESTIONS["DAST-10"][i].split('. ', 1)[0]
        found_question_key = None
        for key in answers.keys():
            if QUESTIONS["DAST-10"][i] in key:
                found_question_key = key
                break
        
        if found_question_key:
            answer = answers.get(found_question_key, "").strip()
            total_score += score_map.get(answer, 0)
    return total_score