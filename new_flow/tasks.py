from crewai import Task
from agents import *
from pydantic import BaseModel, Field
from textwrap import dedent

# --- Pydantic Models for Structured Output ---
class CrisisDetectionOutput(BaseModel):
    is_crisis: bool = Field(description="True if the query indicates a mental health crisis or emergency, False otherwise.")
    explanation: str = Field(description="A brief explanation for the crisis detection.")

class MentalConditionOutput(BaseModel):
    condition: str = Field(description="The classified mental health condition or concern (e.g., 'PHQ-9', 'GAD-7', 'DAST-10', 'General Well-being', 'Other').")
    rationale: str = Field(description="A brief rationale for the classification.")


# --- Tasks ---
crisis_detection_task = Task(
    description=(
        "Analyze the user's input '{user_query}' for any signs of immediate mental health crisis " \
        "(e.g., suicidal ideation, severe panic, acute distress) using the 'sentiment_classifier_tool' " \
        "which classifies the text into LABEL_0 which means text is non-suicidal and LABEL_1 which means text indicates suicidality."
        "If the text is classified as LABEL_1 which means crisis, provide the Bhutanese helplines using the 'Bhutanese Helplines' tool ONLY "
        "Input: {user_query}. Output MUST be a JSON string adhering to the CrisisDetectionOutput schema."
    ),
    expected_output=f"A JSON string representing {CrisisDetectionOutput}",
    agent=crisis_detection_agent,
    output_json=CrisisDetectionOutput
)

mental_condition_classification_task = Task(
    description=(
        "Given the user's query and their profile, classify their primary mental health concern or condition. "
        "Prefer 'PHQ-9' for depression, 'GAD-7' for anxiety, 'DAST-10' for substance abuse. "
        "If none fit well, use 'General Well-being' or 'Other'. "
        "Input: User Query - {user_query}, User Profile - {user_profile}. "
        "Output MUST be a JSON string adhering to the MentalConditionOutput schema."
    ),
    expected_output=f"A JSON string representing {MentalConditionOutput}",
    agent= mental_condition_classifier_agent,
    output_json=MentalConditionOutput
)

query_vector_db_task = Task(
    description=(
        "Given the user's initial query '{user_query}' and the collected user profile '{user_profile_data_json}' "
        "from the Behavioral Agent, perform the following steps: "
        "1. Parse the `user_profile_data_json` string into a Python dictionary. "
        "2. **Analyze the '{user_query}'. If it is general or vague (e.g., 'I'm feeling down', 'I need some advice'), "
        "   use your intelligence to formulate a more specific query or identify potential mental health keywords "
        "   (e.g., 'stress', 'anxiety', 'depression', 'general well-being') that reflect the user's potential "
        "   underlying condition. Prioritize keywords present in the simulated vector database's categories. "
        "   If the query is already specific, use it directly.** "
        "3. Use the 'Vector Database Operations' tool with the 'query' operation, passing the formulated "
        "   specific query/keywords (from step 2) as `query_text` and the parsed user profile data for personalized retrieval. "
        "4. If `user_profile_data_json` indicates 'consent_denied' or 'skipped_all', ensure the query to the "
        "   vector database focuses on general well-being topics (e.g., pass 'general well-being' as `query_text`), "
        "   overriding any specific query attempts that rely on profiling. "
        "The output should be a detailed list of retrieved, relevant information blocks from the knowledge base "
        "based on the refined query, or general well-being tips if no specific match is found."
    ),
    expected_output="A detailed list of relevant mental health information and recommendations from the knowledge base, "
                    "ensuring a helpful response even for vague initial queries.",
    agent=data_retriever_agent,
    context=[crisis_detection_task, mental_condition_classification_task], # This task runs after user profile collection
    # Update inputs to match the output of the previous task
    input_type='json', # Indicate that user_profile_data_json is expected to be a JSON string
    parameters={'user_profile_data_json': '{{ collect_user_profile_task.output }}'} # Map previous task's output
)

personalize_and_recommend_task = Task(
    description=dedent("""
        You are an expert in Medicine Buddha, embodying the principles of healing and compassion. Your purpose is to guide users through their mental health challenges, such as depression, anxiety, 
        stress-related disorders, and schizophrenia, by providing user-friendly, empathetic, and culturally resonant recommendations for recovery.

        You will be provided with a {user_query} from the user describing their mental health struggle.

        Synthesize the user's initial query '{user_query}', the collected user profile '{user_profile_data_json}', 
        the retrieved information '{rag_query_result_json}' from the RAG agent (which includes recommendations and identified condition), 
        and the assessment results '{assessment_result_json}' from the Assessment Agent.
        1. Parse all input JSON strings into Python dictionaries. 
        2. Generate a highly personalized, empathetic, and actionable mental health recommendation. 
        3. Ensure the language is culturally sensitive to Bhutan and the recommendations are practical. 
        4. If `user_profile_data_json` indicates consent was denied or profile was skipped, provide general, but still helpful, recommendations.
        5. If `assessment_result_json` indicates a completed assessment, incorporate the score and interpretation into the recommendation. 
          For example, if 'Mild depression' was assessed, suggest interventions relevant to mild depression.
          If assessment was skipped or denied, or not needed, proceed with recommendations based on the `identified_condition` from RAG only.
        The final response should be a well-structured message providing the personalized recommendations,
        acknowledging any assessments made or skipped.
        6.Personalize and Empathize: Tailor your recommendations by thoughtfully considering the user's {user_profile_data_json}. For instance, an older user might benefit from different social connection suggestions than a younger one, or location might inform community resources.
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
    agent=personalized_recommendation_agent,
    context=[crisis_detection_task, mental_condition_classification_task, query_vector_db_task], # Depends on all preceding tasks
    input_type='json',
    parameters={
        'user_profile_data_json': '{{ collect_user_profile_task.output }}',
        'rag_query_result_json': '{{ query_vector_db_task.output }}',
        'assessment_result_json': '{{ conduct_assessment_task.output }}'
    },
    output_file='task6.txt'
)


