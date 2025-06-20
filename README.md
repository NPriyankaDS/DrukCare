# DrukCare
A skeleton for CrewAI agents for providing mental health assistance.
## Tagline: Empowering Mental Well-being with Intelligent and Culturally Sensitive Support.

# 1. About
DrukCare AI is an intelligent chatbot application designed to provide empathetic and personalized mental health assistance, specifically tailored for the context of Bhutan. Leveraging the CrewAI framework, this system orchestrates a team of specialized AI agents to guide users through various stages of support, from crisis detection and profile collection to dynamic mental health assessments and personalized recommendations.

The project aims to offer accessible, initial mental health guidance, respecting user privacy and cultural nuances, while adhering to ethical guidelines.

# 2. Features
Crisis Detection: Immediate identification of crisis situations and provision of relevant Bhutanese helplines with empathetic responses.

Consent-Based User Profiling: Secure collection of demographic information (age, gender, location, ethnicity) with explicit user consent, offering options to skip.

Intelligent Knowledge Retrieval (RAG): Utilizes Retrieval-Augmented Generation to intelligently refine vague user queries and retrieve relevant mental health information from a knowledge base.

Dynamic Mental Health Assessments: Administers context-specific questionnaires (e.g., PHQ-9 for depression, GAD-7 for anxiety, DAST-10 for substance abuse) based on identified conditions, only after obtaining user consent.

Personalized Recommendations: Generates tailored, actionable mental health recommendations synthesized from user profile, retrieved knowledge, and assessment results.

Modular Architecture: Designed for easy extension and maintenance with clear separation of Agents, Tasks, and Tools.

# 3. Workflow
The DrukCare AI operates as a sequential CrewAI process, ensuring a structured and coherent user interaction flow:

## Crisis Detection:

Input: User's initial query.

Action: The Crisis Detection Specialist agent analyzes the input for emergency signs.

Output: If a crisis is detected, provides helplines. Otherwise, passes control to the Behavioral Agent.

## User Profile Collection:

Input: User's query and status from Crisis Detection.

Action: The Behavioral Profile Analyst agent initiates a step-by-step consent-based questionnaire (age, gender, location, ethnicity). Users can consent, skip individual questions, or skip all.

Output: A structured user profile (or an indication of skipped/denied consent).

Knowledge Retrieval & Query Refinement (RAG):

Input: User's initial query and collected user profile.

Action: The Knowledge Base Manager & Query Refiner agent interprets the user's intent, formulating specific keywords for the vector database. It then retrieves relevant mental health information and identifies a potential condition (e.g., 'depression', 'anxiety').

Output: Relevant mental health recommendations and the identified condition.

## Conditional Assessment:

Input: Identified condition from the RAG agent and user's query.

Action: The Mental Health Assessment Specialist agent determines if an assessment is relevant (e.g., PHQ-9 for depression, GAD-7 for anxiety). If relevant, it seeks explicit user consent. If consent is given, it administers the questionnaire step-by-step.

Output: Assessment status (completed, skipped, denied) and results (score, interpretation) if completed.

## Personalized Recommendation:

Input: Original user query, collected user profile, RAG results, and assessment results.

Action: The Personalized Recommendation Engine synthesizes all gathered information to generate highly personalized, empathetic, and actionable mental health recommendations, culturally adapted for Bhutan.

Output: The final comprehensive recommendation to the user.

# 4. Architecture/Components
The application is built using the CrewAI framework, comprising Agents, Tasks, and Tools.

## 4.1. Agents
Crisis Detection Specialist: Focuses on immediate safety and empathetic crisis response.

Behavioral Profile Analyst: Manages consent-driven collection of user demographic data.

Memory: Enabled (memory=True) for short-term conversational context within the profile collection flow.

Knowledge Base Manager & Query Refiner: Responsible for intelligent query formulation and RAG operations.

Mental Health Assessment Specialist: Oversees the conditional and consent-based administration of mental health questionnaires.

Personalized Recommendation Engine: Synthesizes all information to generate tailored mental health advice.

## 4.2. Tasks
crisis_detection_task: Initial analysis for crisis.

collect_user_profile_task: Manages the step-by-step user profiling.

query_vector_db_task: Refines queries and retrieves knowledge from the simulated database.

conduct_assessment_task: Orchestrates the dynamic questionnaire administration.

personalize_and_recommend_task: Generates the final, comprehensive recommendation.

## 4.3. Tools
All tools are encapsulated within the MentalHealthTools class.

Bhutanese Helplines: Provides a predefined list of mental health helplines relevant to Bhutan.

User Profile Manager: Handles rule-based, step-by-step collection and parsing of user demographic data (age, gender, location, ethnicity). Simulates storage in a database.

Vector Database Operations: Simulates ingestion and querying of a vector database for mental health recommendations. It includes logic to identify specific conditions based on query keywords.

Administer Questionnaire: A generalized tool that administers various mental health questionnaires (e.g., PHQ-9, GAD-7, DAST-10) based on the condition_type identified. It manages consent, question sequencing, and score calculation.

# 5. Usage
To run the DrukCare AI application, execute the main Python script. In the __name__=="__main__" , you can experiment with various scenarios.

python crew.py

The console output will show the detailed steps of how agents interact, tools are used, and the final recommendations are generated for each simulated user input.

## 5.1. LLM API Key Setup
Crucially, DrukCare AI relies on a Language Model (LLM) to function.

You need to set up your LLM provider's API key. For example, if you are using OpenAI:

Obtain an API key from your chosen LLM provider (e.g., OpenAI API Keys).

Set it as an environment variable:

export OPENAI_API_KEY="YOUR_API_KEY_HERE" # On macOS/Linux
# Or for Windows (in Command Prompt):
# set OPENAI_API_KEY="YOUR_API_KEY_HERE"
# In PowerShell:
# $env:OPENAI_API_KEY="YOUR_API_KEY_HERE"

Alternatively, you can hardcode it in your script (for local testing, not recommended for production):

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

Make sure your selected LLM matches the model_name you are using.

# Disclaimer
This DrukCare AI chatbot is designed for informational and initial supportive purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified mental health professional for any questions you may have regarding a medical condition. If you are in a crisis situation, please contact the provided helplines immediately.

# License
This project is open-sourced under the MIT License.
