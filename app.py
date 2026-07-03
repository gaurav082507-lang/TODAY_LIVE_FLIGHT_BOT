from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# ---------------------- API KEY ---------------------- #
API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

# ---------------------- TOOL ---------------------- #
@tool
def get_flight_data(origin, destination):
    """Get CURRENT/real-time Flight Data between two airports.
    Note: this tool only returns flights that are currently active or scheduled for today.
    It cannot look up future or past dates due to API plan restrictions.
    origin and destination must be Airport IATA Codes (e.g. DEL, BOM)."""

    url = (
        f"http://api.aviationstack.com/v1/flights?"
        f"access_key={API_KEY}&dep_iata={origin}&arr_iata={destination}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        flight_data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch flight data: {e}"}

    if "data" not in flight_data:
        return {"error": flight_data.get("error", "No flight data found")}

    flight_details = []

    for flight in flight_data.get("data", []):
        flight_details.append(
            {
                "airline": flight.get("airline", {}).get("name"),
                "flight_no": flight.get("flight", {}).get("iata"),
                "status": flight.get("flight_status"),
                "dep_time": flight.get("departure", {}).get("scheduled"),
                "dep_delay": flight.get("departure", {}).get("delay"),
                "arr_time": flight.get("arrival", {}).get("scheduled"),
                "arr_delay": flight.get("arrival", {}).get("delay"),
                "gate": flight.get("departure", {}).get("gate"),
                "terminal": flight.get("departure", {}).get("terminal"),
            }
        )

    if not flight_details:
        return {
            "message": f"No current flights found from {origin} to {destination}"
        }

    return flight_details


# ---------------------- LLM ---------------------- #
LLM = ChatMistralAI(model="mistral-small-2603")

agent = create_agent(
    model=LLM,
    tools=[get_flight_data],
    system_prompt="""
You are a Helpful Flight Manager who shows users flight information.

Important:
- Your tool only returns CURRENT/real-time flights (today's active/scheduled flights).
- It cannot look up future dates like tomorrow or past dates because of API plan restrictions.
- Convert city names into their correct 3-letter IATA airport codes before calling the tool.
  Example:
  Jaipur -> JAI
  Kolkata -> CCU

If the user asks for a future or past date,
politely explain that only today's live flights are available and offer today's flights instead.
""",
)

# ---------------------- STREAMLIT UI ---------------------- #

st.set_page_config(
    page_title="Flight Manager",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AI Flight Manager")

st.markdown("### Developed by **Gaurav Gupta**")

st.markdown(
    "[LinkedIn Profile](https://www.linkedin.com/in/gaurav-gupta-79754a377)"
)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask about today's flights...")

if query:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Fetching flight information..."):
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                }
            )

            answer = response["messages"][-1].content

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

# Sidebar
with st.sidebar:
    st.header("Flight Manager")

    st.info(
        """
This assistant provides **today's live flight information**.

It **cannot** retrieve:
- Future flights
- Historical flights

because of the AviationStack API plan limitations.
"""
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Gaurav Gupta**")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/gaurav-gupta-79754a377)"
    )
