# modules/questionnaire.py
import json
from typing import Dict, Any

# Path to your questionnaire file
QUESTIONNAIRES_FILE = "questionnaire.json"

def load_questionnaires() -> Dict[str, Any]:
    """Load questionnaires from a file or fallback to defaults."""
    try:
        with open(QUESTIONNAIRES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️ Could not load {QUESTIONNAIRES_FILE}. Using default questions.")
        return create_default_questionnaires()

def create_default_questionnaires() -> Dict[str, Any]:
    """Default fallback questions."""
    return {
        "PHQ-9": [
            "Over the last 2 weeks, how often have you been bothered by any of the following problems? (0-3)",
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling asleep or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself",
            "Trouble concentrating",
            "Moving/speaking slowly or being fidgety",
            "Thoughts of self-harm or death"
        ],
        "GAD-7": [
            "Over the last 2 weeks, how often have you been bothered by the following problems? (0-3)",
            "Feeling nervous or on edge",
            "Not being able to stop worrying",
            "Worrying too much",
            "Trouble relaxing",
            "Restlessness",
            "Irritability",
            "Feeling something awful might happen"
        ],
        "DAST-10": [
            "The following questions are about drug use in the past year. Answer Yes or No.",
            "Used drugs not prescribed?",
            "Abused more than one drug at once?",
            "Tried and failed to stop using?",
            "Experienced blackouts or flashbacks?",
            "Felt guilty about drug use?",
            "Had family complain about your use?",
            "Neglected responsibilities?",
            "Committed illegal acts for drugs?",
            "Had withdrawal symptoms?",
            "Had medical problems due to use?"
        ]
    }

def conduct_assessment(condition: str) -> Dict[str, Any]:
    """Run questionnaire and return answers, score, and interpretation."""
    questions = load_questionnaires().get(condition, [])
    if not questions:
        return {"answers": {}, "score": "N/A", "interpretation": "No questions found."}

    print(f"\n📝 Starting {condition} assessment:\n")
    answers = {}
    for i, q in enumerate(questions[1:], 1):  # skip instructions
        user_input = input(f"Q{i}. {q} ").strip().lower()
        answers[q] = user_input

    score = score_questionnaire(condition, answers)
    interpretation = interpret_score(condition, score)

    return {
        "answers": answers,
        "score": score,
        "interpretation": interpretation
    }

def score_questionnaire(condition: str, answers: Dict[str, str]) -> int:
    """Score PHQ-9, GAD-7, DAST-10 answers."""
    score = 0
    if condition in ["PHQ-9", "GAD-7"]:
        scale = {"0": 0, "not at all": 0, "1": 1, "several days": 1, "2": 2, "more than half": 2, "3": 3, "nearly every": 3}
        for ans in answers.values():
            score += scale.get(ans.strip().lower(), 0)
    elif condition == "DAST-10":
        for ans in answers.values():
            score += 1 if ans.lower() in ["yes", "y", "true", "1"] else 0
    return score

def interpret_score(condition: str, score: int) -> str:
    """Interpret the score based on condition."""
    if condition == "PHQ-9":
        if score <= 4: return "Minimal depression"
        elif score <= 9: return "Mild depression"
        elif score <= 14: return "Moderate depression"
        elif score <= 19: return "Moderately severe depression"
        return "Severe depression"

    if condition == "GAD-7":
        if score <= 4: return "Minimal anxiety"
        elif score <= 9: return "Mild anxiety"
        elif score <= 14: return "Moderate anxiety"
        return "Severe anxiety"

    if condition == "DAST-10":
        if score == 0: return "No problems reported"
        elif score <= 2: return "Low level of problems"
        elif score <= 5: return "Moderate problems"
        elif score <= 8: return "Substantial problems"
        return "Severe problems"

    return "Score interpreted"
