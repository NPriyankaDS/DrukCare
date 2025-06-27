import os
import json
import random
from langsmith import traceable
# Import everything from the 'agents' package
from new_agents.core import (
    mental_condition_classifier_agent,
    data_retriever_agent,
    recommendation_agent,
    crisis_detection_task,
    mental_condition_classification_task,
    recommendation_task,
    crisis_detection_agent,
    crisis_management_crew,
    data_retrieval_crew,
    mental_condition_classifier_crew,
    recommendation_crew,
    CrisisDetectionOutput,
    MentalConditionOutput,
)

# --- Load Questionnaires from JSON ---
QUESTIONNAIRES_FILE = "questionnaire.json"

def load_questionnaires():
    """Loads questionnaire data from a JSON file."""
    try:
        with open(QUESTIONNAIRES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {QUESTIONNAIRES_FILE} not found. Please ensure it's in the same directory.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {QUESTIONNAIRES_FILE}. Check file format.")
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
    for i in range(1, min(10, len(QUESTIONS.get("PHQ-9", [])))):
        if i < len(QUESTIONS["PHQ-9"]):
            question_text_from_json = QUESTIONS["PHQ-9"][i]
            answer = answers.get(question_text_from_json, "").strip()
            total_score += score_map.get(answer, 0)
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
    for i in range(1, min(8, len(QUESTIONS.get("GAD-7", [])))):
        if i < len(QUESTIONS["GAD-7"]):
            question_text_from_json = QUESTIONS["GAD-7"][i]
            answer = answers.get(question_text_from_json, "").strip()
            total_score += score_map.get(answer, 0)
    return total_score

def score_dast10(answers):
    """Calculates the DAST-10 score based on user answers."""
    score_map = {
        "Yes": 1,
        "No": 0
    }
    total_score = 0
    for i in range(1, min(11, len(QUESTIONS.get("DAST-10", [])))):
        if i < len(QUESTIONS["DAST-10"]):
            question_text_from_json = QUESTIONS["DAST-10"][i]
            answer = answers.get(question_text_from_json, "").strip()
            total_score += score_map.get(answer, 0)
    return total_score

# --- Database Simulation (Replace with actual PostgreSQL connection) ---
def fetch_user_profile_from_db(user_id):
    """
    Simulates fetching user profile data from a PostgreSQL database.
    In a real application, you would execute SQL queries here.
    """
    # For demonstration, we'll use a dummy in-memory data store
    dummy_profiles = {
        "user123": {"name": "Alice", "age": 30, "history": "Anxiety, seeking stress management."},
        "user456": {"name": "Bob", "age": 25, "history": "Feeling overwhelmed with work."},
        "anon_user": {"name": "Guest", "age": "N/A", "history": "First-time user."}
    }
    # Simulate user_id if not set (for initial anonymous access)
    return dummy_profiles.get(user_id, dummy_profiles["anon_user"])

# --- Import or Create Data Retrieval Crew (to match workflow) ---
try:
    from new_agents.core import data_retrieval_crew
except ImportError:
    print("Warning: data_retrieval_crew not found. Please implement it in new_agents.core")
    data_retrieval_crew = None

# Try to import mental health condition classifier crew (separate from mental_condition_classifier_crew_obj)
try:
    from new_agents.core import mental_health_condition_classifier_crew
except ImportError:
    print("Warning: mental_health_condition_classifier_crew not found. Using mental_condition_classifier_crew_obj instead")
    mental_health_condition_classifier_crew = mental_condition_classifier_crew

# --- Interactive Chatbot Logic (Following Workflow) ---
def chat_interface():
    """Runs the interactive chatbot following the workflow diagram."""
    chat_history = []
    user_id = "user" + str(random.randint(100, 999))

    def print_message(role, content):
        prefix = "🤖 Bot:" if role == "bot" else "👤 You:"
        print(f"\n{prefix} {content}")

    def reset_session():
        """Reset all session variables"""
        return {
            'current_stage': "welcome",
            'current_user_query': "",
            'classified_condition': "",
            'assessment_consent_asked': False,
            'assessment_started': False,
            'current_question_index': 0,
            'assessment_questions_list': [],
            'assessment_answers': {},
            'questionnaire_score': None,
            'retrieved_data': None,
            'user_profile_data': None,
            'processing_complete': False
        }

    def process_user_query_automatically(user_query, session_vars):
        """Process user query through all automatic stages"""
        session_vars['current_user_query'] = user_query
        session_vars['user_profile_data'] = fetch_user_profile_from_db(user_id)
        
        # STAGE 1: Data Retrieval
        print_message("bot", "Retrieving your profile and relevant data...")
        data_inputs = {
            "user_query": user_query,
            "user_profile": json.dumps(session_vars['user_profile_data'])
        }
        
        # try:
        #     if data_retrieval_crew:
        #         session_vars['retrieved_data'] = data_retrieval_crew.kickoff(inputs=data_inputs)
        #         print_message("bot", "✅ Data retrieval completed.")
        #     else:
        #         session_vars['retrieved_data'] = "Basic mental health resources and general guidelines."
        #         print_message("bot", "✅ Using default resources.")
        # except Exception as e:
        #     print(f"Error during data retrieval: {e}")
        #     session_vars['retrieved_data'] = "Basic mental health resources and general guidelines."
        #     print_message("bot", "⚠️ Using default resources due to retrieval error.")

        # STAGE 2: Crisis Detection
        print_message("bot", "🔍 Analyzing for crisis indicators...")
        inputs = {"user_query": user_query}
        
        crisis_detected = False
        try:
            result = crisis_management_crew.kickoff(inputs=inputs)
            print(f"Type of result: {type(result)}")
            print(f"Result: {result}")
            print(result.get('is_crisis'))
            if isinstance(result, CrisisDetectionOutput):
                crisis_detected = result.content.get('is_crisis')
                print(crisis_detected)
                explanation = result.content.get('explanation')            
            else:
                query_str = str(user_query).lower()
                crisis_keywords = ['crisis', 'emergency', 'urgent', 'immediate', 'suicide', 'harm']
                crisis_detected = any(keyword in query_str for keyword in crisis_keywords)
                

            if crisis_detected:
                print_message("bot", "🚨 CRISIS SITUATION DETECTED")
                print_message("bot", f"Analysis: {explanation}")
                
                # Skip to recommendations for crisis
                return generate_crisis_recommendations(session_vars)
            else:
                print_message("bot", f"✅ No crisis detected. {explanation}")
        
        except Exception as e:
            print(f"Crisis detection error: {e}")
            # Fallback crisis detection
            query_lower = user_query.lower()
            crisis_keywords = ['suicide', 'kill myself', 'want to die', 'hurt myself', 'emergency']
            if any(keyword in query_lower for keyword in crisis_keywords):
                print_message("bot", "🚨 POTENTIAL CRISIS DETECTED (Backup Analysis)")
                print_message("bot", "⚠️ Emergency contacts: 988, Text HOME to 741741")
                return generate_crisis_recommendations(session_vars)
            else:
                print_message("bot", "⚠️ Crisis detection had an error, but no obvious crisis language detected.")

        # STAGE 3: Mental Health Classification
        print_message("bot", "🧠 Analyzing your mental health indicators...")
        
        classification_inputs = {
            "user_query": user_query,
            "user_profile": json.dumps(session_vars['user_profile_data']),
            "retrieved_data": str(session_vars['retrieved_data']) if session_vars['retrieved_data'] else ""
        }
        
        try:
            result = mental_health_condition_classifier_crew.kickoff(inputs=classification_inputs)
            if isinstance(result, MentalConditionOutput):
                session_vars['classified_condition'] = result.condition
                rationale = result.rationale
            else:
                result_str = str(result)
                # Extract condition from result
                for cond in ["PHQ-9", "GAD-7", "DAST-10"]:
                    if cond in result_str:
                        session_vars['classified_condition'] = cond
                        break
                else:
                    session_vars['classified_condition'] = "General Well-being"
                rationale = result_str if result_str else "Analysis completed."

            print_message("bot", f"✅ Analysis complete: **{session_vars['classified_condition']}**")
            print_message("bot", f"Reasoning: {rationale}")
            
        except Exception as e:
            print(f"Classification error: {e}")
            session_vars['classified_condition'] = "General Well-being"
            print_message("bot", "⚠️ Using general assessment due to classification error.")

        # Setup assessment variables
        session_vars['assessment_questions_list'] = QUESTIONS.get(session_vars['classified_condition'], QUESTIONS.get("Other", []))
        session_vars['assessment_answers'] = {}
        session_vars['current_question_index'] = 0
        
        # Move to assessment consent stage
        session_vars['current_stage'] = "assessment_consent"
        session_vars['processing_complete'] = True
        return session_vars

    def generate_crisis_recommendations(session_vars):
        """Generate immediate crisis recommendations"""
        print_message("bot", "🚨 Generating immediate crisis support recommendations...")
        
        recommendation_inputs = {
            "user_query": session_vars['current_user_query'],
            "user_profile": json.dumps(session_vars['user_profile_data']),
            "retrieved_data": str(session_vars['retrieved_data']) if session_vars['retrieved_data'] else "",
            "classified_condition": "Crisis",
            "assessment_answers": json.dumps({"crisis_detected": "true"}),
            "questionnaire_score": "N/A",
            "chat_history": json.dumps(chat_history[-5:]),
            "is_crisis": "true"
        }
        
        try:
            final_recommendation = recommendation_crew.kickoff(inputs=recommendation_inputs)
            print_message("bot", "🆘 **IMMEDIATE CRISIS SUPPORT RECOMMENDATIONS:**")
            print_message("bot", f"{final_recommendation}")
        except Exception as e:
            print(f"Crisis recommendation error: {e}")
            print_message("bot", "🆘 **IMMEDIATE CRISIS SUPPORT:**")
            print_message("bot", "Please reach out to emergency services or a crisis hotline immediately. You don't have to go through this alone.")
        
        print_message("bot", "---")
        print_message("bot", "💬 Please prioritize getting professional help. Is there anything else I can assist you with?")
        
        session_vars['current_stage'] = "query"
        session_vars['processing_complete'] = True
        return session_vars

    # Initialize session
    session_vars = reset_session()
    print_message("bot", "Welcome to DrukCare Chatbot! How can I assist you with your mental well-being today?")

    while True:
        try:
            # Only ask for input when we're actually waiting for user input
            if session_vars['current_stage'] in ["welcome", "query", "assessment_consent"] or \
               (session_vars['current_stage'] == "ask_question"):
                user_input = input("\n👤 You: ")
            else:
                # This should not happen with the new logic, but just in case
                print("ERROR: Unexpected stage requiring input:", session_vars['current_stage'])
                user_input = input("\n👤 You: ")
                
        except (KeyboardInterrupt, EOFError):
            print_message("bot", "Goodbye! Take care of yourself.")
            break

        # Handle special commands
        if user_input.lower() in ["quit", "exit", "bye"]:
            print_message("bot", "Goodbye! Take care of yourself.")
            break

        if user_input.lower() == "reset":
            print_message("bot", "Resetting the conversation...")
            chat_history = []
            session_vars = reset_session()
            print_message("bot", "Welcome to DrukCare Chatbot! How can I assist you with your mental well-being today?")
            continue
        
        chat_history.append({"role": "user", "content": user_input})

        # MAIN WORKFLOW LOGIC
        if session_vars['current_stage'] in ["welcome", "query"]:
            # Process the user query automatically through all stages
            session_vars = process_user_query_automatically(user_input, session_vars)
            # After processing, we should be at assessment_consent stage (or back to query if crisis)
            
        elif session_vars['current_stage'] == "assessment_consent":
            if not session_vars['assessment_consent_asked']:
                consent_message = f"Based on the analysis, I'd like to conduct a **{session_vars['classified_condition']}** assessment to provide you with the most accurate recommendations."
                if session_vars['classified_condition'] in ["PHQ-9", "GAD-7", "DAST-10"]:
                    num_questions = max(0, len(session_vars['assessment_questions_list']) - 1)
                    consent_message += f" This involves {num_questions} standardized questions."
                consent_message += " Would you like to proceed with the assessment? (yes/no)"
                
                print_message("bot", consent_message)
                session_vars['assessment_consent_asked'] = True
                continue
            
            # Handle consent response
            if user_input.lower() == "yes":
                session_vars['assessment_started'] = True
                session_vars['current_question_index'] = 0
                print_message("bot", "Great! Let's begin the assessment.")
                
                # Handle instructions for standardized assessments
                actual_questions_start_idx = 1 if session_vars['classified_condition'] in ["PHQ-9", "GAD-7", "DAST-10"] else 0
                if actual_questions_start_idx == 1 and len(session_vars['assessment_questions_list']) > 0:
                    print_message("bot", f"**Instructions:** {session_vars['assessment_questions_list'][0]}")
                    session_vars['current_question_index'] = 1
                
                # Ask first question
                if session_vars['current_question_index'] < len(session_vars['assessment_questions_list']):
                    current_q_text = session_vars['assessment_questions_list'][session_vars['current_question_index']]
                    display_q_number = session_vars['current_question_index'] + 1 - actual_questions_start_idx
                    print_message("bot", f"Question {display_q_number}: {current_q_text}")
                    session_vars['current_stage'] = "ask_question"
                else:
                    print_message("bot", "No specific questions available. Generating recommendations...")
                    session_vars = generate_final_recommendations(session_vars)

            elif user_input.lower() == "no":
                print_message("bot", "That's perfectly fine. I'll provide recommendations based on your initial query and profile.")
                session_vars = generate_final_recommendations(session_vars)
            else:
                print_message("bot", "Please respond with 'yes' or 'no' to proceed.")

        elif session_vars['current_stage'] == "ask_question":
            # Store the current answer
            if session_vars['current_question_index'] < len(session_vars['assessment_questions_list']):
                question_text_key = session_vars['assessment_questions_list'][session_vars['current_question_index']]
                session_vars['assessment_answers'][question_text_key] = user_input.strip()

            session_vars['current_question_index'] += 1
            actual_questions_start_idx = 1 if session_vars['classified_condition'] in ["PHQ-9", "GAD-7", "DAST-10"] else 0

            if session_vars['current_question_index'] < len(session_vars['assessment_questions_list']):
                # Ask next question
                current_q_text = session_vars['assessment_questions_list'][session_vars['current_question_index']]
                display_q_number = session_vars['current_question_index'] + 1 - actual_questions_start_idx
                print_message("bot", f"Question {display_q_number}: {current_q_text}")
            else:
                # All questions completed
                print_message("bot", "Assessment completed! Calculating your results...")
                
                # Calculate score
                score = None
                if session_vars['classified_condition'] == "PHQ-9":
                    score = score_phq9(session_vars['assessment_answers'])
                    print_message("bot", f"Your PHQ-9 score: **{score}** (0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-19: Moderately Severe, 20-27: Severe)")
                elif session_vars['classified_condition'] == "GAD-7":
                    score = score_gad7(session_vars['assessment_answers'])
                    print_message("bot", f"Your GAD-7 score: **{score}** (0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-21: Severe)")
                elif session_vars['classified_condition'] == "DAST-10":
                    score = score_dast10(session_vars['assessment_answers'])
                    print_message("bot", f"Your DAST-10 score: **{score}** (0: No problems, 1-2: Low level, 3-5: Moderate, 6-8: Substantial, 9-10: Severe)")
                
                session_vars['questionnaire_score'] = score
                session_vars = generate_final_recommendations(session_vars)

    def generate_final_recommendations(session_vars):
        """Generate final recommendations and return to query stage"""
        print_message("bot", "📋 Generating your personalized recommendations...")
        
        recommendation_inputs = {
            "user_query": session_vars['current_user_query'],
            "user_profile": json.dumps(session_vars['user_profile_data']),
            "retrieved_data": str(session_vars['retrieved_data']) if session_vars['retrieved_data'] else "No specific data retrieved",
            "classified_condition": session_vars['classified_condition'],
            "assessment_answers": json.dumps(session_vars['assessment_answers']) if session_vars['assessment_answers'] else "{}",
            "questionnaire_score": str(session_vars['questionnaire_score']) if session_vars['questionnaire_score'] is not None else "N/A",
            "chat_history": json.dumps(chat_history[-5:]),
            "is_crisis": "false"
        }
        
        try:
            final_recommendation = recommendation_crew.kickoff(inputs=recommendation_inputs)
            print_message("bot", "📋 **Your Personalized Mental Health Recommendation:**")
            print_message("bot", f"{final_recommendation}")
        except Exception as e:
            print(f"Recommendation generation error: {e}")
            print_message("bot", "I apologize, but there was an error generating your recommendations. Please try rephrasing your concern.")
        
        print_message("bot", "---")
        print_message("bot", "💡 Is there anything else I can help you with? Feel free to ask another question or type 'reset' to start over.")
        
        session_vars['current_stage'] = "query"
        return session_vars
    

# Run the chatbot
if __name__ == "__main__":
    try:
        chat_interface()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Please check your configuration and try again.")