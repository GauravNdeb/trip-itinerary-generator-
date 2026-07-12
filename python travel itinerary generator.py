import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key , temperature=0.7)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("human", "Create a detailed day trip itinerary for {city} based on these interests: {interests}")
])

class PlannerState(TypedDict):
    messages: List
    city: str
    interests: List[str]
    itinerary: str

def create_itinerary(state: PlannerState) -> PlannerState:
    city = state.get('city', '')
    interests_list = state.get('interests', [])
    interests_str = ", ".join(interests_list) if interests_list else "general sightseeing"
    
    formatted_messages = itinerary_prompt.format_messages(city=city, interests=interests_str)
    response = llm.invoke(formatted_messages)
    
    current_messages = state.get("messages", [])
    if current_messages is None:
        current_messages = []
        
    return {
        **state,
        "itinerary": response.content,
        "messages": current_messages + [response]
    }
workflow = StateGraph(PlannerState)
workflow.add_node("create_itinerary", create_itinerary)
workflow.add_edge(START, "create_itinerary")
workflow.add_edge("create_itinerary", END)
app = workflow.compile()

# ==========================================
# 5. Streamlit Frontend UI
# ==========================================
st.title("SIMPLE TRAVEL PLANNER")

city_input = st.text_input("Destination city:")
interests_input = st.text_input("Enter your interests (comma separated):")

if st.button("Generate plan"):
    if city_input and interests_input:
        list_of_interests = [i.strip() for i in interests_input.split(",")]
        
        # Initial state setup
        initial_state = {
            "messages": [HumanMessage(content=f"Plan a trip to {city_input}")],
            "city": city_input,
            "interests": list_of_interests,
            "itinerary": ""
        }
        
        with st.spinner("Planning....."):
            # Graph execution
            final_output = app.invoke(initial_state)
            
        
        st.subheader("Your Itinerary:")
        st.write(final_output["itinerary"])
    else:
        st.error("Please fill both the fields!")
