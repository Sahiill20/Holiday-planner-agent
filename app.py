"""
Streamlit Web Dashboard for the Personalized Holiday Management Agent.
A premium, conversational interface allowing travelers to plan, validate,
and iteratively refine their trip itineraries in Rupees (₹).
"""

import streamlit as st
import json
import asyncio
import os
from src.graph import compiled_graph

# Set up page configurations
st.set_page_config(
    page_title="Personalized Holiday Agent",
    page_icon="✈️",
    layout="centered"
)

# Custom premium styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Apply modern font */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #f7f9fc;
}

/* Sidebar styled styling */
[data-testid="stSidebar"] {
    background-color: rgba(243, 244, 246, 0.9);
    backdrop-filter: blur(10px);
}

/* Beautiful main header gradient */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    margin-top: 0.5rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #555555;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Thinking status box styling */
.status-box {
    padding: 1.2rem;
    border-radius: 10px;
    background-color: #eef2f7;
    border-left: 5px solid #2a5298;
    margin-bottom: 1.5rem;
    font-weight: 500;
}

/* Card layout container for the output */
.itinerary-card {
    padding: 2rem;
    border-radius: 16px;
    background-color: white;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    border: 1px solid #e1e4e8;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<h1 class="main-title">✈️ Personalized Holiday Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Set your parameters in the panel, then chat to build and revise your plan.</p>', unsafe_allow_html=True)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_itinerary" not in st.session_state:
    st.session_state["current_itinerary"] = None

# -------------------------------------------------------------
# Sidebar Configuration Inputs (No Defaults)
# -------------------------------------------------------------
st.sidebar.header("🎯 Trip Parameters")

max_budget = st.sidebar.number_input(
    "Maximum Budget (₹ - Rupees)",
    min_value=0.0,
    value=None,  # No default value (empty)
    placeholder="Enter budget in ₹",
    step=5000.0,
    help="Your maximum budget constraint in Indian Rupees"
)

dates = st.sidebar.date_input(
    "📅 Travel Date Range",
    value=(),  # No default dates (empty range)
    help="Select the start and end dates of your trip"
)

# Model selection selector
st.sidebar.subheader("🤖 Hugging Face Model")
selected_model_option = st.sidebar.selectbox(
    "Active LLM Model",
    options=[
        "Qwen 2.5 72B (Qwen/Qwen2.5-72B-Instruct:fastest)",
        "Llama 3.1 8B (meta-llama/Llama-3.1-8B-Instruct:fastest)",
        "Qwen 2.5 Coder 32B (Qwen/Qwen2.5-Coder-32B-Instruct:fastest)",
        "Mistral 7B (mistralai/Mistral-7B-Instruct-v0.3:fastest)"
    ],
    index=0,
    help="Select the serverless LLM model to build your itinerary."
)

# Extract repo id and set it to HF_MODEL_NAME environment variable
selected_model = selected_model_option.split("(")[-1].replace(")", "").strip()
os.environ["HF_MODEL_NAME"] = selected_model

st.sidebar.markdown("---")

# Sidebar reset button to clear history
if st.sidebar.button("🧹 Clear Chat & Reset"):
    st.session_state["messages"] = []
    st.session_state["current_itinerary"] = None
    st.rerun()

# -------------------------------------------------------------
# Date Parsing & Verification
# -------------------------------------------------------------
travel_dates_str = None
if isinstance(dates, tuple) and len(dates) == 2:
    start_date, end_date = dates
    travel_dates_str = f"{start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"

# -------------------------------------------------------------
# Output Panel (Dedicated itinerary container at the top)
# -------------------------------------------------------------
if st.session_state["current_itinerary"]:
    st.subheader("📋 Current Verified Travel Plan")
    st.markdown('<div class="itinerary-card">', unsafe_allow_html=True)
    st.markdown(st.session_state["current_itinerary"])
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Chat Interface
# -------------------------------------------------------------
# Render historical messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Empty status placeholder for dynamic thinking updates
status_placeholder = st.empty()

# Capture Chat Input
user_input = st.chat_input("Plan a new trip or modify the plan (e.g. 'Add a visit to the Louvre' or 'Make it cheaper')")

if user_input:
    # 1. Validation Check: Budget and Dates must be configured
    if max_budget is None or travel_dates_str is None:
        # Prompt user in the chat
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.warning("⚠️ Please specify both a **Maximum Budget** and a valid **Travel Date Range** in the left sidebar panel before we can plan your trip.")
        # Store warning message exchange
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": "⚠️ Please specify both a **Maximum Budget** and a valid **Travel Date Range** in the left sidebar panel before we can plan your trip."})
    else:
        # User details are set, run the multi-agent system
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        # Prepare inputs for the LangGraph state machine
        initial_state = {
            "user_request": user_input,
            "max_budget": float(max_budget),
            "travel_month": travel_dates_str,
            "draft_itinerary": st.session_state["current_itinerary"] or "",
            "validation_errors": [],
            "budget_errors": [],
            "iteration_count": 0,
            "final_itinerary_markdown": "",
            "final_itinerary_json": ""
        }
        
        final_state = initial_state.copy()
        
        # Execute LangGraph and update a clean status text in real-time
        with st.spinner("Our AI Agent Team is processing your request..."):
            try:
                for event in compiled_graph.stream(initial_state):
                    for node_name, state_update in event.items():
                        final_state.update(state_update)
                        
                        # Update user-friendly thinking status
                        if node_name == "security_scrubber":
                            status_placeholder.markdown(
                                '<div class="status-box">🔒 <b>Safety screening:</b> Redacting names, emails, and sensitive info...</div>', 
                                unsafe_allow_html=True
                            )
                        elif node_name == "planner":
                            it_count = final_state["iteration_count"]
                            if it_count > 0:
                                status_placeholder.markdown(
                                    f'<div class="status-box">🔄 <b>Refining:</b> Correcting schedule and budget overlaps (Adjustment loop {it_count})...</div>', 
                                    unsafe_allow_html=True
                                )
                            else:
                                status_placeholder.markdown(
                                    '<div class="status-box">📝 <b>Planning:</b> Customizing your travel itinerary schedule...</div>', 
                                    unsafe_allow_html=True
                                )
                        elif node_name == "validator":
                            status_placeholder.markdown(
                                '<div class="status-box">🔍 <b>Validating:</b> Checking opening hours and route feasibility...</div>', 
                                unsafe_allow_html=True
                            )
                        elif node_name == "weather_adaptor":
                            status_placeholder.markdown(
                                '<div class="status-box">🌤️ <b>Seasonality:</b> Adapting activities to local weather conditions...</div>', 
                                unsafe_allow_html=True
                            )
                        elif node_name == "budget_tracker":
                            status_placeholder.markdown(
                                '<div class="status-box">💰 <b>Budget check:</b> Verifying all cost estimates fit your budget limit...</div>', 
                                unsafe_allow_html=True
                            )
                        elif node_name == "loop_back":
                            status_placeholder.markdown(
                                '<div class="status-box">🔄 <b>Re-routing:</b> Submitting adjustments back to the planner...</div>', 
                                unsafe_allow_html=True
                            )
                        elif node_name == "formatter_exporter":
                            status_placeholder.markdown(
                                '<div class="status-box">✨ <b>Finalizing:</b> Formatting your beautiful travel guide...</div>', 
                                unsafe_allow_html=True
                            )
                
                # Clear the thinking status block
                status_placeholder.empty()
                
                # Save output guide to session state
                st.session_state["current_itinerary"] = final_state["final_itinerary_markdown"]
                
                # Output conversation confirmation
                assistant_reply = "I've successfully generated/updated your travel plan! You can review the fully verified itinerary in the panel above. Let me know if you would like to adjust any details."
                
                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
                
            except Exception as e:
                status_placeholder.empty()
                error_msg = str(e)
                if "rate limit" in error_msg.lower() or "429" in error_msg or "too many requests" in error_msg.lower() or "503" in error_msg:
                    friendly_error = (
                        "⚠️ **Rate Limit Exceeded or Model Loading:** The active Hugging Face Serverless API is currently rate-limited or loading the model.\n\n"
                        "Please check your `.env` file token, or wait 1-2 minutes for the Hugging Face serverless endpoint limits to reset or for the model to finish loading."
                    )
                else:
                    friendly_error = f"⚠️ An error occurred during graph execution: {error_msg}"
                
                with st.chat_message("assistant"):
                    st.markdown(friendly_error)
                st.session_state["messages"].append({"role": "assistant", "content": friendly_error})
            
            # Rerun to show update
            st.rerun()
