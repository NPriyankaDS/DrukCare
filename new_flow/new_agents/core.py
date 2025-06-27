import os
from crewai import Agent, Task, Crew, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from new_agents.tools import MentalHealthTools, TextClassifierTool
from textwrap import dedent

load_dotenv()

mental_health_tools = MentalHealthTools()
crisis_classifier_tool = TextClassifierTool(model='sentinet/suicidality')

# Initialize the Langchain Google Generative AI model

llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Pydantic Models for Structured Output ---
class CrisisDetectionOutput(BaseModel):
    is_crisis: str = Field(description="'YES' if the query indicates a mental health crisis or emergency, 'NO' otherwise.")
    explanation: str = Field(description="A brief explanation for the crisis detection.")

class MentalConditionOutput(BaseModel):
    condition: str = Field(description="The classified mental health condition or concern (e.g., 'Anxiety', 'Depression', 'Substance Abuse', 'General Well-being', 'Other').")
    rationale: str = Field(description="A brief rationale for the classification.")

# --- Agents ---

crisis_detection_agent = Agent(
    role='Crisis Detection Specialist',
    goal='Identify immediate crisis situations in user input and provide emergency helplines.',
    backstory=(
        "You are a highly empathetic and vigilant AI assistant trained to detect signs of "
        "severe distress, suicidal ideation, or other mental health emergencies. "
        "Your primary responsibility is to classify the query as crisis or no-crisis situation using the tool you have."
        "If the output is 'is_crisis=True', then it is CRISIS situation otherwise it is NO CRISIS situation."
    ),
    tools=[crisis_classifier_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm # Uncomment and set if you need a specific LLM for this agent
)

mental_condition_classifier_agent = Agent(
    role='Mental Health Condition Classifier',
    goal='Classify the user\'s mental health concern or condition, specifically aiming to identify the relevant questionnaire based on the condition detected.',
    backstory=(
        "You are a meticulous AI specialized in understanding various mental health "
        "states. You analyze user input and identify the keywords for stress, anxiety, depression, substance abuse etc. and their historical profile to categorize "
        "their current concern, with a preference for matching it to a standard assessment "
        "like PHQ-9, GAD-7, or DAST-10, or to 'General Well-being' or 'Other'."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False
)

data_retriever_agent = Agent(
    role='User Profile Data Retriever',
    goal='Fetch relevant user profile information from the database.',
    backstory=(
        "You are an efficient data access specialist. Your role is to securely "
        "retrieve user-specific data that is crucial for personalized interactions. "
        "For this simulation, you 'fetch' data from a provided dictionary."
    ),
    llm=llm, # Even though it's simulated, an LLM is required by CrewAI
    verbose=False,
    allow_delegation=False
)

recommendation_agent = Agent(
    role='Personalized Recommendation Generator',
    goal='Provide tailored mental health recommendations based on all gathered information, including questionnaire scores.',
    backstory=(
        "You are a compassionate and knowledgeable AI dedicated to offering "
        "actionable and personalized advice. You synthesize user queries, "
        "profile data, assessment answers, and quantitative scores from assessments "
        "to deliver helpful recommendations, including suggesting professional help when appropriate."
    ),
    tools=[mental_health_tools.get_bhutanese_helplines],
    llm=llm,
    verbose=False,
    allow_delegation=False,
    reasoning=True
)

# --- Tasks ---
data_retriever_task = Task(
    description=(
        "Query the database to retrieve the user_profile given the user_id"
    ),
    expected_output="A user profile in JSON format.",
    agent=data_retriever_agent
)

crisis_detection_task = Task(
    description=(
        "Analyze the user's current query to determine if it indicates a mental health crisis or emergency. "
        "Input: {user_query}. Output MUST be a JSON string adhering to the CrisisDetectionOutput schema. "
        "Example: {'is_crisis': True, 'explanation': 'User expressed suicidal ideation.'}"
    ),
    expected_output=f"The output {CrisisDetectionOutput}",
    agent=crisis_detection_agent,
    context=[data_retriever_task],
    output_json=CrisisDetectionOutput                               # This is where the Pydantic model is passed
)

mental_condition_classification_task = Task(
    description=(
        "Given the user's initial query '{user_query}' and the collected user profile '{user_profile}' "
        "from the Behavioral Agent, perform the following steps: "
        "1. Parse the `user_profile_data_json` string into a Python dictionary. "
        "2. **Analyze the '{user_query}'. If it is general or vague (e.g., 'I'm feeling down', 'I need some advice'), "
        "   use your intelligence to formulate a more specific query or identify potential mental health keywords "
        "   (e.g., 'stress', 'anxiety', 'depression', 'general well-being') that reflect the user's potential "
        "   underlying condition."
    ),
    expected_output=f"The output {MentalConditionOutput}",
    agent=mental_condition_classifier_agent,
    output_json=MentalConditionOutput                     # This is where the Pydantic model is passed
)

recommendation_task = Task(
    description=dedent("""
        You are an expert in Medicine Buddha, embodying the principles of healing and compassion. Your purpose is to guide users through their mental health challenges, such as depression, anxiety, 
        stress-related disorders, and schizophrenia, by providing user-friendly, empathetic, and culturally resonant recommendations for recovery.

        You will be provided with a {user_query} from the user describing their mental health struggle.

        Synthesize the user's initial query '{user_query}', the collected user profile '{user_profile}', 
        the retrieved information '{retrieved_data}' from the RAG agent (which includes recommendations and identified condition), 
        and the assessment results '{questionnaire_score}' and '{assessment_answers}'from the Assessment Agent.
        1. Parse all input JSON strings into Python dictionaries. 
        2. Generate a highly personalized, empathetic, and actionable mental health recommendation. 
        3. Ensure the language is culturally sensitive to Bhutan and the recommendations are practical. 
        4. If `user_profile_data_json` indicates consent was denied or profile was skipped, provide general, but still helpful, recommendations.
        5. If `assessment_result_json` indicates a completed assessment, incorporate the score and interpretation into the recommendation. 
          For example, if 'Mild depression' was assessed, suggest interventions relevant to mild depression.
          If assessment was skipped or denied, or not needed, proceed with recommendations based on the `identified_condition` from RAG only.
        "The final response should be a well-structured message providing the personalized recommendations,
        "acknowledging any assessments made or skipped.
        6.Personalize and Empathize: Tailor your recommendations by thoughtfully considering the user's {user_profile}. For instance, an older user might benefit from different social connection suggestions than a younger one, or location might inform community resources.
        7.Align with Bhutanese Cultural Values: Ensure all responses and interactions deeply align with Bhutanese cultural values. This includes:
        8.Gross National Happiness (GNH): Frame recommendations within the holistic pursuit of well-being, acknowledging both material and spiritual aspects of happiness.
        9.Compassion: Express genuine empathy and kindness in your language and suggestions.
        10.Interconnectedness: Emphasize the importance of community, family, and the natural world in healing, reflecting the interconnectedness of all beings.
        11.Respect for Tradition: Integrate traditional Bhutanese wisdom, practices (e.g., mindfulness, simple rituals, connection to nature), and the role of spiritual guidance in your advice, where appropriate. Avoid language that might dismiss traditional beliefs about illness.
        12.User-Friendly Language: Keep the language clear, encouraging, and easy to understand for someone in distress.
        13.Actionable Steps: Provide concrete, gentle, and practical measures the user can consider. 
        NOTE: Provide the helplines only when necessary. """
    ),
    expected_output="A comprehensive, personalized, and empathetic mental health recommendation for the user, "
                    "tailored by profile, RAG results and assessment result (if available)",
    agent=recommendation_agent,
    context=[mental_condition_classification_task, data_retriever_task]
)

# --- Crews ---

crisis_management_crew = Crew(
    agents=[crisis_detection_agent],
    tasks=[crisis_detection_task],
    verbose=True # Set to True to see CrewAI's internal thoughts and execution
)

mental_condition_classifier_crew = Crew(
    agents=[mental_condition_classifier_agent],
    tasks=[mental_condition_classification_task],
    verbose=True
)

data_retrieval_crew = Crew(
    agents=[data_retriever_agent],
    tasks=[data_retriever_task],
    verbose=True
)

recommendation_crew = Crew(
    agents=[recommendation_agent],
    tasks=[recommendation_task],
    verbose=True
)
