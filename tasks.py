from crewai import Task
from textwrap import dedent
from agents import crisis_detection_agent, behavioral_agent, rag_agent, personalized_recommendation_agent, assessment_agent

crisis_detection_task = Task(
    description=dedent(
        "Analyze the user's input '{user_query}' for any signs of immediate mental health crisis "
        "(e.g., suicidal ideation, severe panic, acute distress). "
        "If a crisis is detected, provide the Bhutanese helplines using the 'Bhutanese Helplines' tool ONLY"
        "and respond in a deeply empathetic and supportive manner, urging them to contact the helplines in Bhutan. "
        "Integrate traditional Bhutanese wisdom, practices (e.g., mindfulness, simple rituals, connection to nature), and the role of spiritual guidance in your advice, where appropriate"
        "If no crisis is detected, clearly state that and pass control to the Behavioral Agent."
        "NOTE: Use ONLY the helplines fetched using the tool you have."
    ),
    expected_output="An empathetic message with helplines if crisis detected, or a 'no crisis' message.",
    agent=crisis_detection_agent,
    output_file='task1.txt'
)

collect_user_profile_task = Task(
    description=dedent(
        "Initiate and manage the user profile collection process. "
        "Your first action should be to use the 'User Profile Manager' tool with the initial '{user_query}' "
        "and an empty `current_profile_str`. "
        "Based on the tool's output (specifically 'next_question_for_user' and 'status'), "
        "either ask the precise next question to the user, or finalize the profile collection. "
        "If the tool's status is 'complete', 'skipped_all', or 'consent_denied', this task is done. "
        "You MUST pass the 'profile' dictionary (as a JSON string) from the tool's output "
        "as the `current_profile_str` in subsequent calls to the tool within this task's conceptual loop "
        "(i.e., for a multi-turn interaction if this task were broken down into sub-tasks or handled by an external loop). "
        "For this single-turn task, assume the LLM will interpret the 'next_question_for_user' "
        "and this task will output the final profile state as a JSON string."
        "If consent is denied or all questions are skipped, ensure the output reflects this so the next agent can proceed."
    ),
    expected_output="A JSON string representing the final collected user profile (e.g., '{\"age\": 30, \"gender\": \"female\", \"consent_given\": true}') "
                    "or an indication of no consent: '{\"consent_given\": false}' or skipped: '{\"status\": \"skipped_all\"}'.",
    agent=behavioral_agent,
    context=[crisis_detection_task], # This task runs after crisis detection
    output_file='task2.txt'
)

ingest_data_task = Task(
    description=dedent(
        "Ingest the relevant general mental health data and specific Bhutanese context information "
        "into the simulated vector database using the 'Vector Database Operations' tool with the 'ingest' operation. "
        "This task should only run if new data needs to be added or updated to the knowledge base. "
        "For this workflow, assume some initial data is 'ingested' for demonstration."
        "The output should confirm data ingestion."
    ),
    expected_output="Confirmation message of data ingestion.",
    agent=rag_agent,
    # This task is more for setup/maintenance. For a live user query, it runs implicitly
    # or before the system starts taking user queries.
    # We'll simulate a query later in the personalized recommendation task.
)

query_vector_db_task = Task(
    description=dedent(
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
    agent=rag_agent,
    context=[crisis_detection_task, collect_user_profile_task], # This task runs after user profile collection
    # Update inputs to match the output of the previous task
    input_type='json', # Indicate that user_profile_data_json is expected to be a JSON string
    parameters={'user_profile_data_json': '{{ collect_user_profile_task.output }}'} # Map previous task's output
)

conduct_assessment_task = Task(
    description=dedent(
        "Based on the 'identified_condition' from '{rag_query_result_json}' (which contains both recommendations and the identified condition) "
        "and the user's initial query '{user_query}', determine if a detailed assessment is appropriate. "
        "1. Parse the `rag_query_result_json` string to get the `identified_condition`. "
        "2. If the `identified_condition` is 'depression' or 'anxiety' or 'substance_abuse', " # Now checking for multiple conditions
        "   initiate the 'Administer Questionnaire' tool with the corresponding `condition_type`. "
        "3. ** Crucially: First, ask the user for consent to take the questionnaire. If they provide 'yes', proceed. "
        "   If they say 'no' or 'skip', or if the questionnaire is completed or skipped, respect their choice and proceed to the next stage. ** "
        "4. If administering the questionnaire, follow the prompts from the 'Administer Questionnaire' tool. "
        "   You will need to pass the user's response and the `current_assessment_state_str` back to the tool. "
        "   (Note: In a true interactive multi-turn Crew, this task would loop or delegate back to the main app "
        "   for user input. For a sequential Crew, you are simulating a continuous interaction within this task's scope.)"
        "5. If no specific assessment is triggered (e.g., for 'general well-being' or 'stress'), the task should output "
        "   a status indicating 'no_assessment_needed' or 'assessment_skipped_by_condition', and proceed."
        "6. The final output of this task should be a JSON string indicating the assessment status (e.g., 'completed_phq9', 'consent_denied', 'skipped_assessment', 'no_assessment_needed') "
        "   and relevant assessment results (e.g., total_score, interpretation) if completed."
    ),
    expected_output="A JSON string containing 'assessment_status' (e.g., 'completed_phq9', 'completed_gad7', 'consent_denied', 'skipped_assessment', 'no_assessment_needed'), "
                    "'assessment_name' (e.g., 'PHQ-9'), 'total_score' (if applicable, or null), and 'interpretation' (if applicable, or null).",
    agent=assessment_agent,
    context=[query_vector_db_task],
    input_type='json',
    parameters={'rag_query_result_json': '{{ query_vector_db_task.output }}'},
    output_file='task5.txt'
)

personalize_and_recommend_task = Task(
    description=dedent("""
        You are an expert in Medicine Buddha, embodying the principles of healing and compassion. Your purpose is to guide users through their mental health challenges, such as depression, anxiety, 
        stress-related disorders, and schizophrenia, by providing user-friendly, empathetic, and culturally resonant recommendations for recovery.

        You will be provided with a {user_query} from the user describing their mental health struggle.

        Synthesize the user's initial query '{user_query}', the collected user profile '{user_profile_data_json}', 
        the retrieved information '{retrieved_info_json}' from the RAG agent (which includes recommendations and identified condition), 
        and the assessment results '{assessment_result_json}' from the Assessment Agent.
        1. Parse all input JSON strings into Python dictionaries. 
        2. Generate a highly personalized, empathetic, and actionable mental health recommendation. 
        3. Ensure the language is culturally sensitive to Bhutan and the recommendations are practical. 
        4. If `user_profile_data_json` indicates consent was denied or profile was skipped, provide general, but still helpful, recommendations.
        5. If `assessment_result_json` indicates a completed assessment, incorporate the score and interpretation into the recommendation. 
          For example, if 'Mild depression' was assessed, suggest interventions relevant to mild depression.
          If assessment was skipped or denied, or not needed, proceed with recommendations based on the `identified_condition` from RAG only.
        "The final response should be a well-structured message providing the personalized recommendations,
        "acknowledging any assessments made or skipped.
        6.Personalize and Empathize: Tailor your recommendations by thoughtfully considering the user's {user_profile_data_json}. For instance, an older user might benefit from different social connection suggestions than a younger one, or location might inform community resources.
        7.Align with Bhutanese Cultural Values: Ensure all responses and interactions deeply align with Bhutanese cultural values. This includes:
        8.Gross National Happiness (GNH): Frame recommendations within the holistic pursuit of well-being, acknowledging both material and spiritual aspects of happiness.
        9.Compassion: Express genuine empathy and kindness in your language and suggestions.
        10.Interconnectedness: Emphasize the importance of community, family, and the natural world in healing, reflecting the interconnectedness of all beings.
        11.Respect for Tradition: Integrate traditional Bhutanese wisdom, practices (e.g., mindfulness, simple rituals, connection to nature), and the role of spiritual guidance in your advice, where appropriate. Avoid language that might dismiss traditional beliefs about illness.
        12.User-Friendly Language: Keep the language clear, encouraging, and easy to understand for someone in distress.
        13.Actionable Steps: Provide concrete, gentle, and practical measures the user can consider. """
    ),
    expected_output="A comprehensive, personalized, and empathetic mental health recommendation for the user, "
                    "tailored by profile, RAG results and assessment result (if available)",
    agent=personalized_recommendation_agent,
    context=[crisis_detection_task, collect_user_profile_task, query_vector_db_task, conduct_assessment_task], # Depends on all preceding tasks
    input_type='json',
    parameters={
        'user_profile_data_json': '{{ collect_user_profile_task.output }}',
        'retrieved_info_json': '{{ query_vector_db_task.output }}',
        'assessment_result_json': '{{ conduct_assessment_task.output }}'
    },
    output_file='task6.txt'
)


