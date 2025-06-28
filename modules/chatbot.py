# Refactored DrukCare Mental Health Chatbot
# Filename: drukcare_chatbot.py

import os
import json
import random
from typing import Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langsmith import traceable

# Load environment variables
load_dotenv()

# ======================= CONFIGURATION =======================
from new_flow.new_agents.tools import MentalHealthTools, TextClassifierTool
from modules.llm_setup import get_llm
from modules.questionnaire import load_questionnaires, conduct_assessment, score_questionnaire, interpret_score
from modules.config import get_config

# Load config values
config = get_config()

# LLM Initialization
llm = get_llm()

# Tool Setup
mental_health_tools = MentalHealthTools()
crisis_classifier_tool = TextClassifierTool(model=config["crisis_model"])

# ======================= ASSESSMENT QUESTIONNAIRES =======================
QUESTIONS = load_questionnaires()

# ======================= OUTPUT SCHEMAS =======================
class CrisisDetectionOutput(BaseModel):
    is_crisis: bool = Field(description="True if the query indicates a mental health crisis.")
    explanation: str = Field(description="Reason for classifying as crisis or not.")

class MentalConditionOutput(BaseModel):
    condition: str = Field(description="The diagnosed mental condition or concern.")
    rationale: str = Field(description="Why the classification was made.")

# ======================= AGENT FACTORY =======================
def create_agent(role: str, goal: str, backstory: str, tools=None,**kwargs) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools or [],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        **kwargs
    )

# ======================= AGENTS =======================
crisis_detection_agent = create_agent(
    "Crisis Detection Specialist",
    "Identify immediate crisis situations and escalate if needed.",
    "Trained to detect signs of suicidal ideation and mental health emergencies.",
    tools=[crisis_classifier_tool]
)

mental_condition_classifier_agent = create_agent(
    "Mental Health Condition Classifier",
    "Classify user's mental health condition.",
    "Analyzes text for stress, anxiety, depression and matches with PHQ-9, GAD-7, DAST-10."
)

data_retriever_agent = create_agent(
    "User Profile Data Retriever",
    "Retrieve user profile details.",
    "Pulls demographic and background mental health info."
)

recommendation_agent = create_agent(
    "Personalized Recommendation Generator",
    "Provide tailored mental health recommendations based on all gathered information, including questionnaire scores.",
    "You are a compassionate and knowledgeable AI dedicated to offering "\
    "actionable and personalized advice. You synthesize user queries, "\
    "profile data, assessment answers, and quantitative scores from assessments "\
    "to deliver helpful recommendations, including suggesting professional help when appropriate.",
    tools=[mental_health_tools.get_bhutanese_helplines],
    reasoning=True
)

# ======================= TASKS =======================
crisis_detection_task = Task(
    description="Analyze the user's input for crisis indicators using the classifier tool.",
    expected_output="JSON object with is_crisis and explanation fields.",
    output_json=CrisisDetectionOutput,
    input_variables=["user_query"],
    agent=crisis_detection_agent
)

mental_condition_classification_task = Task(
    description="Given the following user query: {user_query}, classify the user's mental health condition and match the assessment.",
    expected_output="JSON object with condition and rationale fields.",
    output_json=MentalConditionOutput,
    input_variables=["user_query", "user_profile"],
    agent=mental_condition_classifier_agent
)

data_retriever_task = Task(
    description="Fetch user profile data in structured JSON.",
    expected_output="User demographic and background profile as JSON.",
    input_variables=["user_query", "user_profile"],
    agent=data_retriever_agent
)

recommendation_task = Task(
    description="Provide tailored mental health recommendation based on all context.",
    expected_output="A complete recommendation for user in plain text.",
    input_variables=["user_query", "user_profile", "classified_condition", "assessment_answers", "questionnaire_score", "is_crisis"],
    agent=recommendation_agent
)

# ======================= CREWS =======================
crisis_management_crew = Crew(agents=[crisis_detection_agent], tasks=[crisis_detection_task], verbose=True)
mental_condition_crew = Crew(agents=[mental_condition_classifier_agent], tasks=[mental_condition_classification_task], verbose=True)
data_retrieval_crew = Crew(agents=[data_retriever_agent], tasks=[data_retriever_task], verbose=True)
recommendation_crew = Crew(agents=[recommendation_agent], tasks=[recommendation_task], verbose=True)

# ======================= EXPORTABLE API =======================
def run_crisis_check(user_query: str) -> dict:
    result = crisis_management_crew.kickoff({"user_query": user_query})
    return result.return_values if hasattr(result, "return_values") else {}

def run_condition_classification(user_query: str, user_profile: str) -> dict:
    return mental_condition_crew.kickoff({
        "user_query": user_query,
        "user_profile": user_profile
    })

def run_user_profile_retrieval(user_query: str, user_profile: str):
    return data_retrieval_crew.kickoff({
        "user_query": user_query,
        "user_profile": user_profile
    })

def run_recommendations(user_query: str, user_profile: str, condition: str, answers: str, score: str, is_crisis: str):
    return recommendation_crew.kickoff({
        "user_query": user_query,
        "user_profile": user_profile,
        "classified_condition": condition,
        "assessment_answers": answers,
        "questionnaire_score": score,
        "is_crisis": is_crisis
    })

# ======================= FULL CHAT FLOW =======================
@traceable(name= "Druckare Chatbot full flow")
def full_chat_flow(user_query: str, user_id: str = "anon_user"):
    print("📄 Fetching user profile...")
    dummy_profile = {
        "id": user_id,
        "age": 30,
        "location": "Thimphu",
        "history": "Some prior anxiety episodes",
        "preferences": "Prefers meditation"
    }

    print("🔍 Checking for crisis...")
    crisis_result = run_crisis_check(user_query)
    is_crisis = crisis_result.get("is_crisis", False)
    explanation = crisis_result.get("explanation", "")

    if is_crisis:
        print(f"🚨 Crisis Detected: {explanation}")
        rec = run_recommendations(user_query, user_profile=json.dumps(dummy_profile), condition="Crisis", answers="{}", score="N/A", is_crisis="true")
        print("\n🆘 Crisis Support Recommendation:\n", rec)
        return

    print("🔎 Classifying condition...")
    condition_result = run_condition_classification(user_query, json.dumps(dummy_profile))
    condition = condition_result.get("condition", "General Well-being")

    print(f"🧠 Detected condition: {condition}")
    if condition not in QUESTIONS:
        print("Skipping assessment as condition is general or unknown.")
        score = "N/A"
        answers = {}
        interpretation = "Not applicable"
    else:
        assessment = conduct_assessment(condition)
        answers = assessment["answers"]
        score = assessment["score"]
        interpretation = assessment["interpretation"]

    print("💡 Generating recommendations...")
    final_rec = run_recommendations(
        user_query,
        json.dumps(dummy_profile),
        condition,
        json.dumps(answers),
        str(score),
        is_crisis="false"
    )

    print("\n📋 Final Recommendation:\n", final_rec)
    print("📊 Score Interpretation:", interpretation)

if __name__ == "__main__":
    query = input("👤 Enter your mental health query: ")
    full_chat_flow(query)
