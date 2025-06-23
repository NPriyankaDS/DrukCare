import os
from crewai import Agent, LLM
from tools import MentalHealthTools
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Instantiate the tools
mental_health_tools = MentalHealthTools()

llm = LLM(
          model="gemini/gemini-2.5-flash",
          api_key=os.getenv("GOOGLE_API_KEY"),
          temperature=0.3,
          max_tokens=None,
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
    goal=(
        "**Interact step-by-step with the user to collect their profile information (age, gender, location, ethnicity) with explicit consent.** "
        "You MUST use the 'User Profile Manager' tool, passing the user's latest input '{user_query}' and the `current_profile_str` to it. " # Crucial change here
        "**Crucially, after each tool call, you MUST analyze the tool's output JSON.** "
        "If the `status` from the tool's output is 'consent_pending', 'age_pending', 'gender_pending', 'location_pending', or 'ethnicity_pending', "
        "you MUST output the exact string: 'QUESTION_FOR_USER: ' followed by the value of `next_question_for_user` from the tool's output. "
        "This tells the outer loop to prompt the human user with this question. "
        "If the tool's `status` is 'complete', 'skipped_all', or 'consent_denied', output a final natural language message "
        "summarizing the profile collection outcome (e.g., 'Profile collection completed, I have your age as 30.') "
        "followed by a unique tag: 'PROFILE_COMPLETED', 'PROFILE_SKIPPED', or 'CONSENT_DENIED' at the very end of your output. "
        "Ensure the output JSON from the tool is still part of the task's final output for subsequent tasks to use as context."
    ),
    backstory=(
        "You are an AI assistant specialized in understanding user behavior and preferences. "
        "Your goal is to politely and clearly ask for user demographic information, "
        "ensuring consent is obtained. You must also provide an option to skip these questions. "
        "You are skilled at using the 'User Profile Manager' tool to guide a rule-based "
        "questionnaire and relay the exact questions or status messages to the user. "
        "You are aware of prior crisis detection status and should adapt your initial greeting accordingly."
    ),
    tools=[mental_health_tools.manage_user_profile],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retry_limit=2, 
    llm=llm 
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
        "**Conditionally administer and manage appropriate mental health questionnaires (e.g., PHQ-9, GAD-7, DAST-10) "
        "to gauge severity, but only with explicit consent from the user.** "
        "You MUST use the 'Administer Questionnaire' tool, passing the user's latest input '{user_query}' and the `current_assessment_state_str` to it. " # Crucial change here
        "**Crucially, after each tool call, you MUST analyze the tool's output JSON.** "
        "If the `status` from the tool's output is 'consent_pending' or 'q_pending', you MUST output the exact string: 'QUESTION_FOR_USER: ' "
        "followed by the `next_question_for_user` from the tool. This tells the outer loop to prompt the human user. "
        "If the tool's `status` is 'complete', 'skipped', or 'consent_denied', output a final natural language message "
        "summarizing the assessment outcome (e.g., 'Assessment completed, your score is X.') "
        "followed by a unique tag: 'ASSESSMENT_COMPLETED', 'ASSESSMENT_SKIPPED', or 'ASSESSMENT_DENIED' at the very end of your output. "
        "4. If no specific assessment is triggered (e.g., for 'general well-being' or 'stress'), the task should output "
        "   a natural language message followed by 'NO_ASSESSMENT_NEEDED'."
        "Ensure the output JSON from the tool is still part of the task's final output for subsequent tasks to use as context."
    ),
    backstory=(
        "You are an empathetic and professional AI, skilled in guiding users through sensitive "
        "mental health assessments. Your expertise lies in ensuring user comfort and privacy, "
        "while collecting crucial information to refine the understanding of their condition. "
        "You strictly adhere to consent protocols and adapt the assessment based on initial condition identification. "
        "You are aware that this might be part of a multi-turn conversation and must always ask the explicit next question from the tool."
    ),
    tools=[mental_health_tools.administer_questionnaire], # Updated tool name
    verbose=True,
    allow_delegation=False,
    llm=llm # Explicitly assign LLM for demonstration
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


