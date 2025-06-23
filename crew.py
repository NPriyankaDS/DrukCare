from crewai import Crew, Process
from agents import *
from tasks import *
import json
from langsmith import traceable
from typing import Optional

# Define the Crew with a sequential process
bhutan_mental_health_crew = Crew(
    agents=[
        crisis_detection_agent,
        behavioral_agent,
        rag_agent,
        assessment_agent,
        personalized_recommendation_agent
    ],
    tasks=[
        crisis_detection_task,
        collect_user_profile_task,
        # Ingest data might be a separate background process or an initial setup task
        # ingest_data_task,
        query_vector_db_task,
        conduct_assessment_task,
        personalize_and_recommend_task
    ],
    process=Process.sequential, # Execute tasks in the order defined
    verbose=True,
    output_log_file="output.txt",
    manager_llm=None # Only necessary for hierarchical process
)

# Function to run a single turn of the mental health assistant crew
@traceable
def run_crew_turn(user_input: str, current_profile_state: dict, current_assessment_state: dict, rag_query_result: Optional[str]=None, retrieved_info: Optional[str]=None) -> dict:
    """
    Runs one turn of the mental health assistance crew.

    Returns a dictionary containing the AI's response and updated states.
    """
    inputs = {
        "user_query": user_input,
        "user_profile_data_json": json.dumps(current_profile_state),
        "assessment_result_json": json.dumps(current_assessment_state),
        "rag_query_result_json": rag_query_result,
        "retrieved_info_json": retrieved_info
    }

    try:
        # CrewAI's kickoff returns the final output of the last task that runs
        raw_output = bhutan_mental_health_crew.kickoff(inputs=inputs)
        raw_output_string = raw_output.raw
        
        # CrewAI's output is often a string directly from the last agent.
        # We need to parse it if it's expected to be JSON.
        # The expected_output of the tasks return JSON strings.
        # So we try to parse them here.

        response_for_user = ""
        updated_profile_state = current_profile_state
        updated_assessment_state = current_assessment_state
        
        # Attempt to parse the raw_output as JSON
        try:
            parsed_output = json.loads(raw_output_string)
            
            # Logic to determine response and update states based on task outputs
            if "status" in parsed_output:
                status = parsed_output["status"]
                
                # Handle output from User Profile Manager (via collect_user_profile_task)
                if "profile" in parsed_output and "message_for_agent" in parsed_output:
                    updated_profile_state = parsed_output["profile"]
                    if status == "consent_pending" or "pending" in status:
                        response_for_user = parsed_output["next_question_for_user"]
                    elif status == "complete":
                        response_for_user = "Thank you for providing your profile. I'll use it to tailor recommendations."
                    elif status == "skipped_all":
                        response_for_user = "You chose to skip profile collection. I'll provide general recommendations."
                    elif status == "consent_denied":
                        response_for_user = "You chose not to share your profile. I'll provide general recommendations."
                    
                # Handle output from Administer Questionnaire (via conduct_assessment_task)
                elif "assessment_name" in parsed_output and "total_score" in parsed_output:
                    # This means conduct_assessment_task completed or is pending
                    updated_assessment_state = {
                        "consent_given": parsed_output.get("consent_given"),
                        "current_q_idx": parsed_output.get("current_q_idx"),
                        "scores": parsed_output.get("scores")
                    } # Reconstruct state for next turn if needed
                    
                    if status == "q_pending" or status == "consent_pending":
                        response_for_user = parsed_output["next_question_for_user"]
                    elif status == "complete":
                        response_for_user = f"You completed the {parsed_output['assessment_name']} questionnaire. Your score is {parsed_output['total_score']} ({parsed_output['interpretation']}). I'll use this information."
                    elif status == "consent_denied" or status == "skipped":
                        response_for_user = f"You chose to skip the {parsed_output['assessment_name']} questionnaire. That's fine."
                    elif status == "no_assessment_needed":
                        response_for_user = "No specific assessment was needed at this time."
                
                # If it's a final recommendation, it won't have a 'status' like the tools do,
                # it will be the direct output of the personalized_recommendation_agent.
                # This needs to be handled if the final agent's output is not JSON.
                else:
                    # Default to raw_output if not a structured JSON from known tool workflows
                    response_for_user = raw_output 
            else:
                # If raw_output is not a JSON with 'status', it's likely the final recommendation
                response_for_user = raw_output
        except json.JSONDecodeError:
            # If the output isn't JSON, it's likely the final human-readable response
            response_for_user = raw_output

        return {
            "response": response_for_user,
            "updated_profile_state": updated_profile_state,
            "updated_assessment_state": updated_assessment_state
        }
    except Exception as e:
        return {
            "response": f"An error occurred: {e}. Please try again.",
            "updated_profile_state": current_profile_state,
            "updated_assessment_state": current_assessment_state
        }

# @traceable
# def run_mental_health_assistant(user_input: str, current_profile_state: str = '{}', current_assessment_state: str = '{}'):
#     """
#     Function to run the mental health assistance crew.
#     This function now takes current_profile_state and current_assessment_state to simulate multi-turn interaction.
#     """
#     inputs = {
#         "user_query": user_input,
#         "user_profile_data_json": current_profile_state,
#         "rag_query_result_json": " ",
#         "retrieved_info_json": " ",
#         "assessment_result_json": current_assessment_state # Pass generic assessment state
#     }

#     print("### Running the Mental Health Assistance Crew ###")
#     result = bhutan_mental_health_crew.kickoff(inputs=inputs)
#     print("\n### Crew execution complete. Final Response: ###")
#     print(result)
#     return result


# --- How to use it ---
if __name__ == "__main__":
    
    
    print("\n--- DrukCare AI Chatbot ---")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("---")

    current_profile_state = {}  # Stores profile data across turns
    current_assessment_state = {} # Stores assessment data across turns
    is_conversation_active = True

    while is_conversation_active:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("DrukCare AI: Goodbye! Take care.")
            is_conversation_active = False
            break

        # Run one turn of the Crew and get the response and updated states
        turn_output = run_crew_turn(user_input, current_profile_state, current_assessment_state).get('response')
        
        # Update states for the next turn
        current_profile_state = turn_output["updated_profile_state"]
        current_assessment_state = turn_output["updated_assessment_state"]

        # Print the AI's response to the user
        print(f"DrukCare AI: {turn_output.raw}")

        # Optionally, add a short delay to simulate thinking time, especially with a real LLM
        # time.sleep(0.5)

        # Check if the conversation should end (e.g., after crisis helplines or final recommendation)
        # This logic needs to be refined based on what constitutes a "final" response from your agents.
        # For now, we assume if the response is a comprehensive recommendation, the conversation implicitly concludes,
        # but the loop continues until the user types 'quit'.
        # A more robust check would involve parsing the final agent's JSON output for a 'final_status' field.
        # Example check:
        # if "final_recommendation_status" in turn_output['response']: # Requires agent to output this
        #     print("DrukCare AI: Conversation concluded. Type 'quit' to exit or continue with a new query.")
