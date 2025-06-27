import streamlit as st
import json
import random
from crewai import Crew
from agents import *
from tasks import *
from utils import *

# --- Load Questionnaires from JSON ---
QUESTIONNAIRES_FILE = "new_flow\questionnaire.json"

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

# --- Database Simulation (Replace with actual PostgreSQL connection) ---
def get_postgresql_connection():
    """
    Placeholder for establishing a PostgreSQL database connection.
    In a real application, you would use a library like `psycopg2` or `SQLAlchemy`.
    Example:
    import psycopg2
    conn = psycopg2.connect(
        host="your_host",
        database="your_database",
        user="your_user",
        password="your_password"
    )
    return conn
    """
    st.info("Simulating PostgreSQL database connection.")
    # For demonstration, we'll use a dummy in-memory data store
    pass

def fetch_user_profile_from_db(user_id):
    """
    Simulates fetching user profile data from a PostgreSQL database.
    In a real application, you would execute SQL queries here.
    """
    # conn = get_postgresql_connection()
    # cursor = conn.cursor()
    # cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
    # user_profile = cursor.fetchone()
    # conn.close()

    # Dummy user profiles for demonstration
    dummy_profiles = {
        "user123": {"name": "Alice", "age": 30, "history": "Anxiety, seeking stress management."},
        "user456": {"name": "Bob", "age": 25, "history": "Feeling overwhelmed with work."},
        "anon_user": {"name": "Guest", "age": "N/A", "history": "First-time user."}
    }
    st.session_state['user_id'] = st.session_state.get('user_id', "anon_user") # Default to anon if not set
    return dummy_profiles.get(st.session_state['user_id'], dummy_profiles["anon_user"])


# --- CrewAI Setup ---
# --- Crews ---
crisis_management_crew = Crew(
    agents=[crisis_detection_agent],
    tasks=[crisis_detection_task],
    verbose=1
)

mental_condition_classifier_crew = Crew(
    agents=[mental_condition_classifier_agent],
    tasks=[mental_condition_classification_task],
    verbose=1
)

rag_recommendation_crew = Crew(
    agents=[data_retriever_agent, rag_agent, personalized_recommendation_agent],
    tasks=[query_vector_db_task, personalize_and_recommend_task],
    verbose=1
)

# --- Streamlit App ---

st.set_page_config(page_title="DrukCare Chatbot", layout="centered")

st.title("DrukCare Chatbot 🤖")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    st.session_state['stage'] = "welcome"
    st.session_state['current_user_query'] = ""
    st.session_state['classified_condition'] = ""
    st.session_state['assessment_consent_asked'] = False # New: to track if consent has been asked
    st.session_state['assessment_started'] = False # New: to track if consent was given and assessment started
    st.session_state['current_question_index'] = 0 # New: to track current question for conversational flow
    st.session_state['assessment_questions_list'] = [] # New: to store questions for current assessment
    st.session_state['assessment_answers'] = {}
    st.session_state['questionnaire_score'] = None
    st.session_state['user_id'] = "user" + str(random.randint(100, 999))

# --- Display Chat History ---
for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Main Workflow Logic ---

def handle_welcome():
    st.session_state['chat_history'].append({"role": "bot", "content": "Welcome to DrukCare Chatbot! How can I assist you with your mental well-being today?"})
    st.session_state['stage'] = "query"

def handle_user_query(user_input):
    st.session_state['current_user_query'] = user_input
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    st.session_state['stage'] = "crisis_check"

def handle_crisis_check():
    st.session_state['chat_history'].append({"role": "bot", "content": "Running crisis detection..."})
    inputs = {"user_query": st.session_state['current_user_query']}
    try:
        result = crisis_management_crew.kickoff(inputs=inputs)
        if isinstance(result, CrisisDetectionOutput):
            is_crisis = result.is_crisis
            explanation = result.explanation
        else:
            is_crisis = False
            explanation = "Could not parse crisis detection output."
            st.warning(f"Unexpected crisis detection output format: {result}")

        st.session_state['chat_history'].append({"role": "bot", "content": f"Crisis detection result: {'YES' if is_crisis else 'NO'}. Reason: {explanation}"})

        if is_crisis:
            st.session_state['chat_history'].append({"role": "bot", "content": "It seems like you might be going through a crisis. Please consider reaching out to a mental health professional or emergency services immediately. We will still try to provide a general recommendation."})
            st.session_state['stage'] = "recommend"
        else:
            st.session_state['chat_history'].append({"role": "bot", "content": "No immediate crisis detected. Proceeding to mental condition classification."})
            st.session_state['stage'] = "mental_classify"
    except Exception as e:
        st.error(f"Error during crisis detection CrewAI kickoff: {e}")
        st.session_state['chat_history'].append({"role": "bot", "content": "An error occurred during crisis detection. Proceeding as no crisis detected."})
        st.session_state['stage'] = "mental_classify"


def handle_mental_classify():
    st.session_state['chat_history'].append({"role": "bot", "content": "Analyzing your input to classify mental condition..."})
    user_profile = fetch_user_profile_from_db(st.session_state['user_id'])
    inputs = {
        "user_query": st.session_state['current_user_query'],
        "user_profile": json.dumps(user_profile)
    }
    try:
        result = mental_condition_classifier_crew.kickoff(inputs=inputs)
        if isinstance(result, MentalConditionOutput):
            condition = result.condition
            rationale = result.rationale
        else:
            condition = "General Well-being"
            rationale = "Could not parse mental condition output."
            st.warning(f"Unexpected mental condition classification output format: {result}")

        st.session_state['classified_condition'] = condition
        st.session_state['chat_history'].append({"role": "bot", "content": f"Based on our analysis, your concern seems related to: **{condition}**. Rationale: {rationale}"})
        
        # Reset assessment flags for a new potential assessment
        st.session_state['assessment_consent_asked'] = False
        st.session_state['assessment_started'] = False
        st.session_state['current_question_index'] = 0
        st.session_state['assessment_questions_list'] = QUESTIONS.get(st.session_state['classified_condition'], QUESTIONS["Other"])
        st.session_state['assessment_answers'] = {} # Clear previous answers

        st.session_state['stage'] = "assessment_consent" # New stage for consent
    except Exception as e:
        st.error(f"Error during mental condition classification CrewAI kickoff: {e}")
        st.session_state['chat_history'].append({"role": "bot", "content": "An error occurred during mental condition classification. Defaulting to general questions."})
        st.session_state['classified_condition'] = "General Well-being"
        st.session_state['stage'] = "assessment_consent" # Still go for consent even if default


def handle_assessment_consent():
    if not st.session_state['assessment_consent_asked']:
        consent_message = f"I've identified your concern as related to **{st.session_state['classified_condition']}**. Would you like to answer a few standard questions to help me provide a more accurate recommendation? Please type 'yes' or 'no'."
        if st.session_state['classified_condition'] in ["PHQ-9", "GAD-7", "DAST-10"]:
            consent_message = f"Based on your input, a **{st.session_state['classified_condition']}** assessment might be helpful. This typically involves {len(st.session_state['assessment_questions_list']) - 1 if st.session_state['classified_condition'] in ['PHQ-9', 'GAD-7', 'DAST-10'] and len(st.session_state['assessment_questions_list']) > 0 else 'a few'} questions. Would you like to proceed? Please type 'yes' or 'no'."
        st.session_state['chat_history'].append({"role": "bot", "content": consent_message})
        st.session_state['assessment_consent_asked'] = True
    # User's input will be processed by the main chat_input handler

def handle_ask_question(user_input):
    # This function is called when a user submits an answer during the assessment flow.
    # It means the previous input was an answer to a question.

    # Store the user's previous answer
    if st.session_state['current_question_index'] > 0: # Only store if it's an actual question answer
        previous_question_text = st.session_state['assessment_questions_list'][st.session_state['current_question_index'] - 1]
        st.session_state['assessment_answers'][previous_question_text] = user_input.strip()

    # Move to the next question
    st.session_state['current_question_index'] += 1

    questions_to_ask = st.session_state['assessment_questions_list']
    
    # Adjust starting index for questionnaires with an initial instruction
    start_idx_offset = 1 if st.session_state['classified_condition'] in ["PHQ-9", "GAD-7", "DAST-10"] else 0

    if st.session_state['current_question_index'] < len(questions_to_ask):
        # If there's an instruction at the beginning, display it once
        if start_idx_offset == 1 and st.session_state['current_question_index'] == 1:
            st.session_state['chat_history'].append({"role": "bot", "content": f"**Instructions:** {questions_to_ask[0]}"})

        current_q_display_index = st.session_state['current_question_index'] # Use this for Q1, Q2 etc.
        question_text = questions_to_ask[st.session_state['current_question_index']]
        
        st.session_state['chat_history'].append({"role": "bot", "content": f"Q{current_q_display_index}: {question_text}"})
        st.session_state['stage'] = "ask_question" # Stay in this stage to await next answer
    else:
        # All questions answered, calculate score and proceed to recommendation
        st.session_state['chat_history'].append({"role": "bot", "content": "Thank you for completing the assessment."})
        
        score = None
        if st.session_state['classified_condition'] == "PHQ-9":
            score = score_phq9(st.session_state['assessment_answers'])
            st.session_state['chat_history'].append({"role": "bot", "content": f"Your PHQ-9 score is: **{score}**."})
        elif st.session_state['classified_condition'] == "GAD-7":
            score = score_gad7(st.session_state['assessment_answers'])
            st.session_state['chat_history'].append({"role": "bot", "content": f"Your GAD-7 score is: **{score}**."})
        elif st.session_state['classified_condition'] == "DAST-10":
            score = score_dast10(st.session_state['assessment_answers'])
            st.session_state['chat_history'].append({"role": "bot", "content": f"Your DAST-10 score is: **{score}**."})
        
        st.session_state['questionnaire_score'] = score
        st.session_state['stage'] = "recommend"
        st.rerun() # Rerun to trigger recommendation flow


def handle_recommendation():
    st.session_state['chat_history'].append({"role": "bot", "content": "Generating personalized recommendations..."})
    user_profile = fetch_user_profile_from_db(st.session_state['user_id'])

    inputs = {
        "user_query": st.session_state['current_user_query'],
        "user_profile": json.dumps(user_profile),
        "assessment_answers": json.dumps(st.session_state['assessment_answers']),
        "questionnaire_score": str(st.session_state['questionnaire_score']) if st.session_state['questionnaire_score'] is not None else "N/A"
    }
    try:
        final_recommendation = rag_recommendation_crew.kickoff(inputs=inputs)
        st.session_state['chat_history'].append({"role": "bot", "content": f"**Final Recommendation:**\n\n{final_recommendation}"})
        st.session_state['chat_history'].append({"role": "bot", "content": "Is there anything else I can help you with today? Type your next query or say 'reset' to start over."})
        st.session_state['stage'] = "query"
    except Exception as e:
        st.error(f"Error during recommendation CrewAI kickoff: {e}")
        st.session_state['chat_history'].append({"role": "bot", "content": "An error occurred while generating recommendations. Please try again or rephrase your query."})
        st.session_state['stage'] = "query"


# --- Trigger Workflow based on Stage ---
if st.session_state['stage'] == "welcome":
    st.session_state['chat_history'] = []
    handle_welcome()
elif st.session_state['stage'] == "crisis_check":
    with st.spinner("Checking for crisis..."):
        handle_crisis_check()
elif st.session_state['stage'] == "mental_classify":
    with st.spinner("Classifying your mental condition..."):
        handle_mental_classify()
elif st.session_state['stage'] == "assessment_consent":
    handle_assessment_consent()
elif st.session_state['stage'] == "ask_question":
    # If we are in 'ask_question' stage, and there's no new user input,
    # it means the question was just posted, so we just wait for input.
    pass # Awaiting user input
elif st.session_state['stage'] == "recommend":
    with st.spinner("Generating your recommendation..."):
        handle_recommendation()

# --- User Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Handle 'reset' command first
    if user_input.lower() == "reset":
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    elif st.session_state['stage'] == "query":
        handle_user_query(user_input)
        st.rerun()
    elif st.session_state['stage'] == "assessment_consent":
        if user_input.lower() == "yes":
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['assessment_started'] = True
            st.session_state['current_question_index'] = 0 # Start from the first question
            # Immediately ask the first question
            # The `handle_ask_question` will now handle question display and index increment.
            handle_ask_question("") # Pass empty string for initial call as it's not an answer
            st.rerun()
        elif user_input.lower() == "no":
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['assessment_started'] = False
            st.session_state['chat_history'].append({"role": "bot", "content": "No problem! We'll proceed with a general recommendation based on your initial query and profile."})
            st.session_state['stage'] = "recommend"
            st.rerun()
        else:
            st.warning("Please type 'yes' or 'no' to proceed with the assessment.")
            st.session_state['chat_history'].append({"role": "user", "content": user_input}) # Echo user's invalid input
            st.session_state['chat_history'].append({"role": "bot", "content": "Please type 'yes' or 'no'."}) # Re-prompt
    elif st.session_state['stage'] == "ask_question":
        st.session_state['chat_history'].append({"role": "user", "content": user_input}) # Echo user's answer
        handle_ask_question(user_input) # Pass the answer to the handler
        st.rerun()
    else:
        st.warning("Please complete the current step before typing a new message or type 'reset' to start over.")

st.markdown("---")
st.markdown(f"**Current User ID:** `{st.session_state['user_id']}` (for demo purposes)")
st.markdown("This is a simulated DrukCare Chatbot. AI responses are generated by the Gemini API via Langchain/CrewAI.")

