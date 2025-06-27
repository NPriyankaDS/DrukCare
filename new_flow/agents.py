import os
from crewai import Agent, LLM
from tools import MentalHealthTools, TextClassifierTool
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Instantiate the tools
mental_health_tools = MentalHealthTools()
sentiment_classifier_tool = TextClassifierTool(model_name="sentinet/suicidality")

llm = LLM(
          model="gemini/gemini-2.0-flash",
          api_key=os.getenv("GOOGLE_API_KEY"),
          temperature=0.3,
          max_tokens=None,
          timeout=None,
          max_retries=2,
        )

# --- Agents ---
crisis_detection_agent = Agent(
    role='Crisis Detection Specialist',
    goal='Identify immediate crisis situations in user input and provide emergency helplines.',
    backstory=(
        "You are a highly empathetic and vigilant AI assistant trained to detect signs of "
        "severe distress, suicidal ideation, or other mental health emergencies. "
        "Your primary responsibility is to classify the query into crisis or no-crisis."
    ),
    tools=[sentiment_classifier_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm 
)

mental_condition_classifier_agent = Agent(
    role='Mental Health Condition Classifier',
    goal='Accurately classify the user\'s mental health concern or condition, specifically aiming to identify if a PHQ-9, GAD-7, or DAST-10 assessment is most appropriate.',
    backstory=(
        "You are a meticulous AI specialized in understanding various mental health states. "
        "You analyze user input and their historical profile to categorize their current concern, "
        "with a preference for matching it to a standard assessment like PHQ-9, GAD-7, or DAST-10, "
        "or to 'General Well-being' or 'Other'."
    ),
    tools=[],
    llm=llm,
    verbose=False,
    allow_delegation=False
)

data_retriever_agent = Agent(
    role='User Profile Data Retriever',
    goal='Fetch relevant user profile information from the database.',
    backstory="You are an efficient data access specialist.",
    tools=[],
    llm=llm, # Even though it's simulated, an LLM is required by CrewAI
    verbose=False,
    allow_delegation=False
)

rag_agent = Agent(
    role='Knowledge Base Manager & Query Refiner', 
    goal='Interpret user queries, formulate specific search terms, and manage/query the mental health knowledge base using RAG.', # Updated goal
    backstory=(
        "You are responsible for intelligently understanding user needs, even from vague inputs. "
        "You will formulate precise search queries or identify relevant mental health keywords "
        "before efficiently retrieving relevant information from the vector database. "
        "You ensure that the knowledge base is always up-to-date and accessible for generating "
        "informed recommendations, and that relevant information is always found, even for general queries."
    ),
    tools=[],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

personalized_recommendation_agent = Agent(
    role='Personalized Recommendation Engine',
    goal='Generate tailored mental health recommendations based on user profile and retrieved knowledge.',
    backstory=(
        "You are the final stage in providing valuable assistance. Leveraging the user's "
        "profile information from the Behavioral Agent and the retrieved insights from "
        "the RAG Agent, you craft highly personalized, empathetic, and actionable "
        "mental health recommendations relevant to the Bhutanese context."
    ),
    tools=[mental_health_tools.get_bhutanese_helplines], # This agent primarily synthesizes information, might not need new tools but processes info from previous tasks
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_retry_limit=2,
    reasoning = True,
    max_reasoning_attempts=2
)
