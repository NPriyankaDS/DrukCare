from crewai.tools import tool

class MentalHealthTools:
    """
    Tools for mental health assistance, simulating database interaction,
    NER, and information retrieval.
    """

    # Simulated storage for different questionnaires (would typically be loaded from JSON files)
    _questionnaires = {
        "depression": {
            "name": "PHQ-9",
            "questions": [
                "Little interest or pleasure in doing things?",
                "Feeling down, depressed, or hopeless?",
                "Trouble falling or staying asleep, or sleeping too much?",
                "Feeling tired or having little energy?",
                "Poor appetite or overeating?",
                "Feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
                "Trouble concentrating on things, such as reading the newspaper or watching television?",
                "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving a lot more than usual?",
                "Thoughts that you would be better off dead, or of hurting yourself in some way?"
            ],
            "response_scale": "(0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)",
            "interpretation_logic": { # Score ranges and their interpretations
                "0-4": "Minimal depression",
                "5-9": "Mild depression",
                "10-14": "Moderate depression",
                "15-19": "Moderately severe depression",
                "20-27": "Severe depression"
            }
        },
        "anxiety": {
            "name": "GAD-7",
            "questions": [
                "Feeling nervous, anxious, or on edge?",
                "Not being able to stop or control worrying?",
                "Worrying too much about different things?",
                "Trouble relaxing?",
                "Being so restless that it's hard to sit still?",
                "Becoming easily annoyed or irritable?",
                "Feeling afraid as if something awful might happen?"
            ],
            "response_scale": "(0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)",
            "interpretation_logic": {
                "0-4": "Minimal anxiety",
                "5-9": "Mild anxiety",
                "10-14": "Moderate anxiety",
                "15-21": "Severe anxiety"
            }
        },
        "substance_abuse": {
            "name": "DAST-10", # Placeholder for a substance abuse questionnaire
            "questions": [
                "Have you used drugs other than those required for medical reasons?",
                "Do you abuse more than one drug at a time?",
                "Are you unable to stop using drugs when you want to?"
                # ... more DAST-10 questions
            ],
            "response_scale": "(0=No, 1=Yes)",
            "interpretation_logic": {
                "0": "No problem indicated",
                "1-2": "Low level of drug abuse",
                "3-5": "Moderate level of drug abuse",
                "6-8": "Substantial level of drug abuse",
                "9-10": "Severe level of drug abuse"
            }
        }
    }

    @tool("Bhutanese Helplines")
    def get_bhutanese_helplines():
        """
        Provides a list of mental health helplines in Bhutan.
        """
        helplines = """
        Here are some mental health helplines in Bhutan:
        - National Mental Health Program Hotline: 1717 (24/7)
        - Jigme Dorji Wangchuck National Referral Hospital (JDWNRH) Psychiatry Department: +975-2-322137
        - Khesar Gyalpo University of Medical Sciences of Bhutan (KGUMSB) Counseling Services: Contact university administration for current numbers.
        - Youth HelpLine (for young people): 1769 / 1768
        - Druk Trace App (for general emergency contacts, including health): Available on app stores.
        """
        return helplines

    @tool("User Profile Manager")
    def manage_user_profile(user_input: str, current_profile_str: str = '{}') -> str:
        """
        Manages the collection of user profile information (age, gender, location, ethnicity)
        in a rule-based manner, allowing users to skip questions.
        This tool handles one input at a time and returns the next question or completion status.

        Expected user_input formats:
        - "yes" / "no" for consent
        - "my age is X" or "X years old" or just "X" for age
        - "I am male/female/non-binary" or just "male/female/non-binary" for gender
        - "I live in Thimphu" or just "Thimphu" for location
        - "I am Drukpa" or just "Drukpa" for ethnicity
        - "skip" to skip the current question
        - "skip all" to skip all remaining profile questions

        Args:
            user_input (str): The current input from the user.
            current_profile_str (str): A JSON string representing the current state of the collected profile.
                                       This should be passed from the task's context to maintain state.

        Returns:
            str: A JSON string containing:
                - 'profile': The updated profile dictionary (e.g., {'age': 30, 'gender': 'Female'}).
                - 'message_for_agent': A detailed message for the agent on how to proceed.
                - 'next_question_for_user': The exact question to ask the user next, or None if complete/skipped.
                - 'status': 'consent_pending', 'age_pending', 'gender_pending', 'location_pending',
                            'ethnicity_pending', 'complete', 'skipped_all', 'consent_denied'.
        """
        import json
        import re

        try:
            profile_data = json.loads(current_profile_str)
        except json.JSONDecodeError:
            profile_data = {} # Initialize if parsing fails

        user_input_lower = user_input.lower().strip()
        message_for_agent = ""
        next_question_for_user = None
        status = 'needs_processing' # Internal temporary status

        # Handle "skip all" command first, regardless of current stage
        if user_input_lower == 'skip all':
            profile_data['consent_given'] = profile_data.get('consent_given', True) # Assume consent if skipping all
            profile_data['age'] = None
            profile_data['gender'] = None
            profile_data['location'] = None
            profile_data['ethnicity'] = None
            message_for_agent = "User chose to skip all profile questions. Profile collection is complete."
            status = 'skipped_all'
            return json.dumps({
                'profile': profile_data,
                'message_for_agent': message_for_agent,
                'next_question_for_user': next_question_for_user,
                'status': status
            })


        # --- Consent Collection ---
        # If consent has not been explicitly given or denied
        if 'consent_given' not in profile_data:
            if user_input_lower == 'yes' or 'consent' in user_input_lower:
                profile_data['consent_given'] = True
                message_for_agent = "Consent received. Proceeding to collect profile details."
                status = 'age_pending' # Next logical step
            elif user_input_lower == 'no' or 'do not consent' in user_input_lower or 'don\'t consent' in user_input_lower:
                profile_data['consent_given'] = False
                message_for_agent = "User denied consent. Profile collection will not proceed."
                status = 'consent_denied'
                return json.dumps({
                    'profile': profile_data,
                    'message_for_agent': message_for_agent,
                    'next_question_for_user': next_question_for_user,
                    'status': status
                })
            else:
                # Still waiting for consent
                message_for_agent = "Please ask the user: 'To help me tailor recommendations, may I collect some basic profile information (age, gender, location, ethnicity)? Please say \'yes\' or \'no\'.'"
                next_question_for_user = "To help me tailor recommendations, may I collect some basic profile information (age, gender, location, ethnicity)? Please say 'yes' or 'no'."
                status = 'consent_pending'
                return json.dumps({
                    'profile': profile_data,
                    'message_for_agent': message_for_agent,
                    'next_question_for_user': next_question_for_user,
                    'status': status
                })

        # --- Profile Data Collection (if consent given) ---
        if profile_data.get('consent_given'):
            # Define the order of questions
            questions_order = ['age', 'gender', 'location', 'ethnicity']
            current_field_to_ask = None

            # Determine which field to process next based on existing data
            for field in questions_order:
                if field not in profile_data or profile_data[field] is None:
                    current_field_to_ask = field
                    break

            # If all fields are already collected or skipped, set status to complete
            if current_field_to_ask is None:
                status = 'complete'
                message_for_agent = "All profile information collected or explicitly skipped."
                if all(val is None for val in [profile_data.get('age'), profile_data.get('gender'), profile_data.get('location'), profile_data.get('ethnicity')]):
                    message_for_agent += " All details were skipped."
                return json.dumps({
                    'profile': profile_data,
                    'message_for_agent': message_for_agent,
                    'next_question_for_user': next_question_for_user,
                    'status': status
                })

            # Process the current field based on user input
            if current_field_to_ask == 'age':
                if user_input_lower == 'skip':
                    profile_data['age'] = None
                    message_for_agent = "Age skipped. Now proceeding to gender."
                    status = 'gender_pending'
                    next_question_for_user = "What is your gender (e.g., Male, Female, Non-binary)? You can say 'skip'."
                else:
                    age_match = re.search(r'age is (\d+)|I am (\d+)\s*years old|(\d+)', user_input, re.IGNORECASE)
                    if age_match:
                        profile_data['age'] = int(age_match.group(1) or age_match.group(2) or age_match.group(3))
                        message_for_agent = f"Noted age: {profile_data['age']}. Now proceeding to gender."
                        status = 'gender_pending'
                        next_question_for_user = "What is your gender (e.g., Male, Female, Non-binary)? You can say 'skip'."
                    else:
                        message_for_agent = "Invalid age input. Please ask the user: 'What is your age? You can say \"skip\".'"
                        next_question_for_user = "What is your age? You can say 'skip'."
                        status = 'age_pending' # Remain in this state until valid input or skip
                
            elif current_field_to_ask == 'gender':
                if user_input_lower == 'skip':
                    profile_data['gender'] = None
                    message_for_agent = "Gender skipped. Now proceeding to location."
                    status = 'location_pending'
                    next_question_for_user = "Which district in Bhutan are you located in (e.g., Thimphu, Paro)? You can say 'skip'."
                else:
                    if "male" in user_input_lower:
                        profile_data['gender'] = 'Male'
                    elif "female" in user_input_lower:
                        profile_data['gender'] = 'Female'
                    elif "non-binary" in user_input_lower:
                        profile_data['gender'] = 'Non-binary'
                    if 'gender' in profile_data and profile_data['gender'] is not None:
                        message_for_agent = f"Noted gender: {profile_data['gender']}. Now proceeding to location."
                        status = 'location_pending'
                        next_question_for_user = "Which district in Bhutan are you located in (e.g., Thimphu, Paro)? You can say 'skip'."
                    else:
                        message_for_agent = "Invalid gender input. Please ask the user: 'What is your gender (e.g., Male, Female, Non-binary)? You can say \"skip\".'"
                        next_question_for_user = "What is your gender (e.g., Male, Female, Non-binary)? You can say 'skip'."
                        status = 'gender_pending'

            elif current_field_to_ask == 'location':
                if user_input_lower == 'skip':
                    profile_data['location'] = None
                    message_for_agent = "Location skipped. Now proceeding to ethnicity."
                    status = 'ethnicity_pending'
                    next_question_for_user = "What is your ethnicity (e.g., Drukpa, Lhotshampa)? You can say 'skip'."
                else:
                    bhutanese_locations = ['thimphu', 'paro', 'phuntsholing', 'punakha', 'wangduephodrang', 'bumthang', 'trashigang', 'mongar', 'chukh', 'daga', 'gasa', 'ha', 'lhuentse', 'pemagatshel', 'samdrupjongkhar', 'samtsi', 'sarpang', 'sarpang', 'shemgang', 'tashiyangtse', 'trongsa', 'tsirang', 'dagana', 'zhemgang'] # Add more as needed
                    found_location = next((loc for loc in bhutanese_locations if loc in user_input_lower), None)
                    if found_location:
                        profile_data['location'] = found_location.capitalize()
                        message_for_agent = f"Noted location: {profile_data['location']}. Now proceeding to ethnicity."
                        status = 'ethnicity_pending'
                        next_question_for_user = "What is your ethnicity (e.g., Drukpa, Lhotshampa)? You can say 'skip'."
                    else:
                        message_for_agent = "Invalid location input. Please ask the user: 'Which district in Bhutan are you located in (e.g., Thimphu, Paro)? You can say \"skip\".'"
                        next_question_for_user = "Which district in Bhutan are you located in (e.g., Thimphu, Paro)? You can say 'skip'."
                        status = 'location_pending'

            elif current_field_to_ask == 'ethnicity':
                if user_input_lower == 'skip':
                    profile_data['ethnicity'] = None
                    message_for_agent = "Ethnicity skipped. Profile collection complete."
                    status = 'complete'
                else:
                    bhutanese_ethnicities = ['drukpa', 'lhotshampa', 'sharchop', 'nagalop', 'kheng', 'brokpa', 'lepcha', 'nepalese'] # Add more as needed
                    found_ethnicity = next((eth for eth in bhutanese_ethnicities if eth in user_input_lower), None)
                    if found_ethnicity:
                        profile_data['ethnicity'] = found_ethnicity.capitalize()
                        message_for_agent = f"Noted ethnicity: {profile_data['ethnicity']}. Profile collection complete."
                        status = 'complete'
                    else:
                        message_for_agent = "Invalid ethnicity input. Please ask the user: 'What is your ethnicity (e.g., Drukpa, Lhotshampa)? You can say \"skip\".'"
                        next_question_for_user = "What is your ethnicity (e.g., Drukpa, Lhotshampa)? You can say 'skip'."
                        status = 'ethnicity_pending'
            
            # If after processing, all fields are collected or skipped, ensure status is complete
            if status != 'complete' and all(field in profile_data and (profile_data[field] is not None or profile_data[field] == None) for field in questions_order):
                 status = 'complete'
                 message_for_agent = "All profile information collected or explicitly skipped. Profile collection complete."
                 next_question_for_user = None # No more questions

        return json.dumps({
            'profile': profile_data,
            'message_for_agent': message_for_agent,
            'next_question_for_user': next_question_for_user,
            'status': status
        })

    @tool("Vector Database Operations")
    def vector_db_operations(operation: str, data: str = None, query_text: str = None, user_profile: dict = None):
        """
        Performs operations on a simulated vector database:
        - Ingestion (operation='ingest', data='text to ingest')
        - Query (operation='query', query_text='text to query', user_profile={'age': 30, ...})

        Returns relevant recommendations based on query and user profile.
        """
        # This is a highly simplified simulation. In reality, this would connect
        # to a vector database like Pinecone, Weaviate, Milvus, etc.

        print(f"\n--- DEBUG: Vector Database Operations Tool Called ---")
        print(f"Operation: {operation}")
        print(f"Data (for ingest): {data}")
        print(f"Query Text (for query): {query_text}")
        print(f"User Profile (for query): {user_profile}")
        print(f"---------------------------------------------------\n")

        simulated_vector_db = {
            "stress": [
                "Mindfulness exercises for stress reduction in Bhutanese context.",
                "Coping strategies for work-related stress, considering local work culture.",
                "Importance of family and community support in managing stress."
            ],
            "anxiety": [
                "Breathing techniques for anxiety relief.",
                "When to seek professional help for anxiety in Bhutan.",
                "Connecting with nature for anxiety management (Bhutan's natural beauty)."
            ],
            "depression": [
                "Understanding symptoms of depression and seeking help.",
                "The role of spirituality and Buddhist principles in mental well-being.",
                "Support groups and community initiatives for depression."
            ],
            "general well-being": [
                "Practicing gratitude and compassion (Metta meditation).",
                "Maintaining a balanced diet and physical activity.",
                "The concept of Gross National Happiness and personal well-being."
            ]
        }

        try:
            if operation == 'ingest':
                # In a real scenario, 'data' would be chunked, embedded, and stored.
                print(f"Simulating ingestion of data: '{data}' into vector DB.")
                # For this demo, we'll just acknowledge ingestion.
                return "Data ingestion simulated successfully."

            elif operation == 'query':
                if not query_text:
                    return "Query text is required for vector database query operation."

                # Simple keyword matching for demo purposes
                relevant_recommendations = []
                query_lower = query_text.lower()

                for keyword, recs in simulated_vector_db.items():
                    if keyword in query_lower:
                        relevant_recommendations.extend(recs)

                # Further refine based on user profile (simulated)
                if user_profile:
                    try:
                        # Attempt to parse user_profile if it's a string, otherwise use as dict
                        if isinstance(user_profile, str):
                            import json
                            parsed_profile = json.loads(user_profile.replace("'", "\"")) # Replace single quotes for valid JSON
                        else:
                            parsed_profile = user_profile

                        if 'age' in parsed_profile and parsed_profile['age'] and int(parsed_profile['age']) < 25:
                            relevant_recommendations.append("Recommendations for youth mental health.")
                        if 'gender' in parsed_profile and parsed_profile['gender'] and parsed_profile['gender'].lower() == 'female':
                            relevant_recommendations.append("Consider resources specific to women's mental health.")
                        if 'location' in parsed_profile and parsed_profile['location'] and parsed_profile['location'].lower() == 'thimphu':
                            relevant_recommendations.append("Local Thimphu-based mental health resources might be available.")
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        print(f"--- DEBUG: Error parsing user_profile in vector_db_operations: {e}")
                        print(f"--- DEBUG: Raw user_profile received: {user_profile}")
                        relevant_recommendations.append("Could not use user profile for deeper personalization due to parsing error.")


                if relevant_recommendations:
                    result = "\n- " + "\n- ".join(list(set(relevant_recommendations))) # Remove duplicates
                    print(f"--- DEBUG: Query Result: {result}")
                    return result
                else:
                    result = "No specific recommendations found for your query. Here are some general well-being tips:\n- " + "\n- ".join(simulated_vector_db["general well-being"])
                    print(f"--- DEBUG: Query Result (General): {result}")
                    return result
            else:
                return "Invalid vector database operation."
        except Exception as e:
            print(f"--- ERROR: An error occurred in vector_db_operations: {e}")
            return f"An error occurred during vector database operation: {e}"
        
    @tool("Administer Questionnaire")
    def administer_questionnaire(user_input: str, condition_type: str, current_assessment_state_str: str = '{}') -> str:
        """
        Administers a mental health questionnaire (e.g., PHQ-9, GAD-7) based on condition_type.
        Manages consent and question-by-question flow.

        Args:
            user_input (str): The user's response (e.g., "yes", "no", "0", "1", "2", "3", "skip").
            condition_type (str): The type of condition (e.g., 'depression', 'anxiety', 'substance_abuse')
                                  to determine which questionnaire to use.
            current_assessment_state_str (str): JSON string containing current state
                                               (e.g., {'consent_given': True, 'current_q_idx': 0, 'scores': []}).

        Returns:
            str: A JSON string containing:
                - 'status': 'consent_pending', 'q_pending', 'complete', 'consent_denied', 'skipped', 'error'.
                - 'next_question_for_user': The question to ask next, or None if complete/consent denied.
                - 'assessment_name': Name of the questionnaire (e.g., "PHQ-9").
                - 'total_score': Total score if complete, otherwise None.
                - 'interpretation': Clinical interpretation if complete, otherwise None.
                - 'current_q_idx': Current question index.
                - 'scores': List of scores for each question.
                - 'error_message': Any error message.
        """
        import json

        try:
            state = json.loads(current_assessment_state_str)
        except json.JSONDecodeError:
            state = {'consent_given': None, 'current_q_idx': -1, 'scores': []}

        user_input_lower = user_input.lower().strip()

        questionnaire_data = MentalHealthTools._questionnaires.get(condition_type)
        if not questionnaire_data:
            return json.dumps({
                'status': 'error',
                'error_message': f"No questionnaire found for condition type: {condition_type}",
                'next_question_for_user': None,
                'assessment_name': None,
                'total_score': None,
                'interpretation': None,
                'current_q_idx': state['current_q_idx'],
                'scores': state['scores']
            })

        q_name = questionnaire_data["name"]
        questions = questionnaire_data["questions"]
        response_scale = questionnaire_data["response_scale"]
        interpretation_logic = questionnaire_data["interpretation_logic"]

        def interpret_score(score, logic):
            for r, interp in logic.items():
                if '-' in r:
                    low, high = map(int, r.split('-'))
                    if low <= score <= high:
                        return interp
                else: # For exact scores or single threshold (like 0)
                    if score == int(r):
                        return interp
            return "Interpretation not available for this score."

        # --- Consent Phase ---
        if state['consent_given'] is None:
            if user_input_lower == 'yes' or 'consent' in user_input_lower:
                state['consent_given'] = True
                state['current_q_idx'] = 0
                return json.dumps({
                    'status': 'q_pending',
                    'next_question_for_user': f"Okay. Here is the first question ({q_name} question 1 of {len(questions)}): {questions[0]} {response_scale}",
                    'assessment_name': q_name,
                    'total_score': None,
                    'interpretation': None,
                    'current_q_idx': state['current_q_idx'],
                    'scores': state['scores']
                })
            elif user_input_lower == 'no' or 'do not consent' in user_input_lower or 'don\'t consent' in user_input_lower:
                state['consent_given'] = False
                return json.dumps({
                    'status': 'consent_denied',
                    'next_question_for_user': f"You chose not to take the {q_name} questionnaire. That's perfectly fine.",
                    'assessment_name': q_name,
                    'total_score': None,
                    'interpretation': None,
                    'current_q_idx': state['current_q_idx'],
                    'scores': state['scores']
                })
            else:
                return json.dumps({
                    'status': 'consent_pending',
                    'next_question_for_user': f"The system has identified '{condition_type}' as a potential area. Would you like to take a brief {q_name} questionnaire to help me understand your feelings better? Please say 'yes' or 'no'.",
                    'assessment_name': q_name,
                    'total_score': None,
                    'interpretation': None,
                    'current_q_idx': state['current_q_idx'],
                    'scores': state['scores']
                })

        # --- Questionnaire Phase (if consent given and not complete) ---
        if state['consent_given'] is True and state['current_q_idx'] < len(questions):
            try:
                score = int(user_input_lower)
                # Basic validation for common scales (0-3 for PHQ/GAD, 0-1 for DAST-10)
                if (q_name in ["PHQ-9", "GAD-7"] and 0 <= score <= 3) or \
                   (q_name == "DAST-10" and 0 <= score <= 1): # Extend as needed for other scales
                    state['scores'].append(score)
                    state['current_q_idx'] += 1
                else:
                    return json.dumps({
                        'status': 'q_pending',
                        'next_question_for_user': f"Please provide a score according to the scale {response_scale}. Current question ({q_name} question {state['current_q_idx'] + 1} of {len(questions)}): {questions[state['current_q_idx']]} {response_scale}",
                        'assessment_name': q_name,
                        'total_score': None,
                        'interpretation': None,
                        'current_q_idx': state['current_q_idx'],
                        'scores': state['scores'],
                        'error_message': 'Invalid score provided.'
                    })
            except ValueError:
                if user_input_lower == 'skip':
                    # User chose to skip the questionnaire mid-way
                    return json.dumps({
                        'status': 'skipped',
                        'next_question_for_user': f"You chose to skip the {q_name} questionnaire. That's fine.",
                        'assessment_name': q_name,
                        'total_score': None,
                        'interpretation': None,
                        'current_q_idx': state['current_q_idx'], # Keep current index as is for debugging if needed
                        'scores': state['scores']
                    })
                else:
                    return json.dumps({
                        'status': 'q_pending',
                        'next_question_for_user': f"Invalid input. Please provide a score or say 'skip'. Current question ({q_name} question {state['current_q_idx'] + 1} of {len(questions)}): {questions[state['current_q_idx']]} {response_scale}",
                        'assessment_name': q_name,
                        'total_score': None,
                        'interpretation': None,
                        'current_q_idx': state['current_q_idx'],
                        'scores': state['scores'],
                        'error_message': 'Invalid input.'
                    })

            # Check if all questions are answered
            if state['current_q_idx'] == len(questions):
                total_score = sum(state['scores'])
                return json.dumps({
                    'status': 'complete',
                    'next_question_for_user': f"Thank you for completing the {q_name} questionnaire.",
                    'assessment_name': q_name,
                    'total_score': total_score,
                    'interpretation': interpret_score(total_score, interpretation_logic),
                    'current_q_idx': state['current_q_idx'],
                    'scores': state['scores']
                })
            else:
                return json.dumps({
                    'status': 'q_pending',
                    'next_question_for_user': f"{q_name} question {state['current_q_idx'] + 1} of {len(questions)}: {questions[state['current_q_idx']]} {response_scale}",
                    'assessment_name': q_name,
                    'total_score': None,
                    'interpretation': None,
                    'current_q_idx': state['current_q_idx'],
                    'scores': state['scores']
                })
        
        # Fallback for unexpected states
        return json.dumps({
            'status': 'error',
            'error_message': "An unexpected state occurred during the questionnaire.",
            'next_question_for_user': None,
            'assessment_name': q_name,
            'total_score': None,
            'interpretation': None,
            'current_q_idx': state['current_q_idx'],
            'scores': state['scores']
        })
