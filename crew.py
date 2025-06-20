from crewai import Crew, Process
from agents import *
from tasks import *
import json

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

def run_mental_health_assistant(user_input: str, current_profile_state: str = '{}', current_assessment_state: str = '{}'):
    """
    Function to run the mental health assistance crew.
    This function now takes current_profile_state and current_assessment_state to simulate multi-turn interaction.
    """
    inputs = {
        "user_query": user_input,
        "user_profile_data_json": current_profile_state,
        "rag_query_result_json": " ",
        "retrieved_info_json": " ",
        "assessment_result_json": current_assessment_state # Pass generic assessment state
    }

    print("### Running the Mental Health Assistance Crew ###")
    result = bhutan_mental_health_crew.kickoff(inputs=inputs)
    print("\n### Crew execution complete. Final Response: ###")
    print(result)
    return result

# --- How to use it ---
if __name__ == "__main__":
   
    
    user_input = "I am feeling to not live. I do not want to live anymore."
    result_flow = run_mental_health_assistant(user_input)

