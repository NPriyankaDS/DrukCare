import streamlit as st
from agents import *
from tasks import *
from crew import *


# --- Streamlit App UI ---
st.set_page_config(page_title="DrukCare AI Chatbot", layout="centered")

st.title("🤖 DrukCare AI Chatbot")
st.write("Your personal mental health assistant for Bhutan.")

# Initialize chat history and state in Streamlit's session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! How can I assist you with your mental well-being today?"}]
if 'current_profile_state' not in st.session_state:
    st.session_state.current_profile_state = {}
if 'current_assessment_state' not in st.session_state:
    st.session_state.current_assessment_state = {}

# Display chat messages from history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("What's on your mind?"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("DrukCare AI is thinking..."):
            # Call the run_crew_turn function from your main script
            turn_output = run_crew_turn(
                user_input,
                st.session_state.current_profile_state,
                st.session_state.current_assessment_state
            )
            
            # Update session states from the turn output
            st.session_state.current_profile_state = turn_output["updated_profile_state"]
            st.session_state.current_assessment_state = turn_output["updated_assessment_state"]
            
            ai_response = turn_output["response"].raw
            st.markdown(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

# Sidebar for debugging/monitoring states
with st.sidebar:
    st.title("Debug Info")
    st.subheader("Profile State")
    st.json(st.session_state.current_profile_state)
    st.subheader("Assessment State")
    st.json(st.session_state.current_assessment_state)
    st.markdown("---")
    if st.button("Start New Conversation"):
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! How can I assist you with your mental well-being today?"}]
        st.session_state.current_profile_state = {}
        st.session_state.current_assessment_state = {}
        st.rerun()

