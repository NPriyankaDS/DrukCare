import os
from crewai import Agent
from tools import MentalHealthTools
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Instantiate the tools
mental_health_tools = MentalHealthTools()

llm = ChatGroq(
          model="groq/llama-3.3-70b-versatile",
          base_url="https://api.groq.com/openai/v1/",
          api_key=os.getenv("OPENAI_API_KEY"),
          temperature=0.3,
          max_tokens=200,
          timeout=None,
          max_retries=2,
        )

crisis_detection_agent = Agent(
    role='Crisis Detection Specialist',
    goal='Identify immediate crisis situations in user input and provide emergency helplines.',
    backstory=(
        "You are a highly empathetic and vigilant AI assistant trained to detect signs of "
        "severe distress, suicidal ideation, or other mental health emergencies. "
        "Your primary responsibility is to ensure the user's immediate safety by providing "
        "relevant emergency contacts for Bhutan and responding with compassion."
    ),
    tools=[mental_health_tools.get_bhutanese_helplines],
    verbose=True,
    allow_delegation=False,
    llm=llm # Uncomment and set if you need a specific LLM for this agent
)

behavioral_agent = Agent(
    role='Behavioral Profile Analyst',
    goal='Collect user profile information (age, gender, location, ethnicity) with explicit consent.',
    backstory=(
        "You are an AI assistant specialized in understanding user behavior and preferences. "
        "Your goal is to politely and clearly ask for user demographic information, "
        "ensuring consent is obtained. You must also provide an option to skip these questions. "
        "You are skilled at using Natural Language Processing (NLP) to extract relevant details."
    ),
    tools=[mental_health_tools.manage_user_profile],
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_retry_limit=2,
    memory=True # Added short-term conversational memory for this agent
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
    tools=[mental_health_tools.vector_db_operations],
    verbose=True,
    allow_delegation=False,
    llm=llm
)


assessment_agent = Agent(
    role='Mental Health Assessment Specialist',
    goal=(
        "Conditionally administer and manage appropriate mental health questionnaires (e.g., PHQ-9, GAD-7) "
        "to gauge severity, but only with explicit consent. "
        "You must use the 'Administer Questionnaire' tool and follow its lead for questions/consent. "
        "Based on the 'identified_condition' from the RAG agent's output, select the correct questionnaire to offer."
    ),
    backstory=(
        "You are an empathetic and professional AI, skilled in guiding users through sensitive "
        "mental health assessments. Your expertise lies in ensuring user comfort and privacy, "
        "while collecting crucial information to refine the understanding of their condition. "
        "You strictly adhere to consent protocols and adapt the assessment based on initial condition identification."
    ),
    tools=[mental_health_tools.administer_questionnaire], # Updated tool name
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
    tools=[], # This agent primarily synthesizes information, might not need new tools but processes info from previous tasks
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_retry_limit=2,
    reasoning = True,
    max_reasoning_attempts=2
)


