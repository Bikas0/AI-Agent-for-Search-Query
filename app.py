import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import sys
from io import StringIO
import contextlib
import time

load_dotenv()

class StreamlitOutputCapture:
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.placeholder = container.empty()

    def write(self, text):
        self.text += text
        self.placeholder.markdown(self.text)

    def flush(self):
        pass

def initialize_session_state():
    # Available models
    available_models = [
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "llama-3.3-70b-versatile",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview"
    ]
    
    if 'web_agent_model' not in st.session_state:
        st.session_state.web_agent_model = available_models[0]
    if 'finance_agent_model' not in st.session_state:
        st.session_state.finance_agent_model = available_models[0]
    if 'team_agent_model' not in st.session_state:
        st.session_state.team_agent_model = available_models[0]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0

def create_single_agent(model_id, agent_type="web"):
    try:
        if agent_type == "web":
            return Agent(
                name="Web Agent",
                model=Groq(
                    id=model_id,
                    temperature=0.7,
                    max_tokens=1024,
                    retry_on_error=True,
                    retry_count=3
                ),
                tools=[DuckDuckGo()],
                instructions=["Always include sources"],
                show_tool_calls=True,
                markdown=True
            )
        else:
            return Agent(
                name="Finance Agent",
                role="Get financial data",
                model=Groq(
                    id=model_id,
                    temperature=0.7,
                    max_tokens=1024,
                    retry_on_error=True,
                    retry_count=3
                ),
                tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
                instructions=["Use tables to display data"],
                show_tool_calls=True,
                markdown=True
            )
    except Exception as e:
        st.error(f"Error creating agent with model {model_id}: {str(e)}")
        return None

def create_agents():
    try:
        web_agent = create_single_agent(st.session_state.web_agent_model, "web")
        finance_agent = create_single_agent(st.session_state.finance_agent_model, "finance")
        
        if web_agent is None or finance_agent is None:
            raise Exception("Failed to create individual agents")

        agent_team = Agent(
            model=Groq(
                id=st.session_state.team_agent_model,
                temperature=0.7,
                max_tokens=1024,
                retry_on_error=True,
                retry_count=3
            ),
            team=[web_agent, finance_agent],
            instructions=["Always include sources", "Use tables to display data"],
            show_tool_calls=True,
            markdown=True,
        )
        return agent_team
    except Exception as e:
        st.error(f"Error creating agent team: {str(e)}")
        return None

@contextlib.contextmanager
def capture_output(container):
    output_capture = StreamlitOutputCapture(container)
    old_stdout = sys.stdout
    sys.stdout = output_capture
    try:
        yield output_capture
    finally:
        sys.stdout = old_stdout

def handle_query(agent_team, query, response_container):
    try:
        with capture_output(response_container) as output:
            response = agent_team.print_response(query, stream=True)
            return output.text
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        st.error(error_message)
        st.session_state.error_count += 1
        
        fallback_response = (
            "I apologize, but I encountered an error while processing your request. "
            "This might be due to API limitations or temporary issues. "
            "Please try again in a moment or rephrase your query."
        )
        return fallback_response

def main():
    st.title("AI Agent Team Interface")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    
    # Available models
    available_models = [
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "llama-3.3-70b-versatile",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview"
    ]
    
    # Model selection dropdowns - now allowing any model for any agent
    web_model = st.sidebar.selectbox(
        "Web Agent Model",
        available_models,
        index=available_models.index(st.session_state.web_agent_model),
        key="web_model_select"
    )
    
    finance_model = st.sidebar.selectbox(
        "Finance Agent Model",
        available_models,  # Using full list of models
        index=available_models.index(st.session_state.finance_agent_model),
        key="finance_model_select"
    )
    
    team_model = st.sidebar.selectbox(
        "Team Agent Model",
        available_models,  # Using full list of models
        index=available_models.index(st.session_state.team_agent_model),
        key="team_model_select"
    )
    
    # Add temperature sliders for each agent
    st.sidebar.header("Model Parameters")
    web_temp = st.sidebar.slider("Web Agent Temperature", 0.0, 1.0, 0.7, key="web_temp")
    finance_temp = st.sidebar.slider("Finance Agent Temperature", 0.0, 1.0, 0.7, key="finance_temp")
    team_temp = st.sidebar.slider("Team Agent Temperature", 0.0, 1.0, 0.7, key="team_temp")
    
    # Update session state with selected models
    st.session_state.web_agent_model = web_model
    st.session_state.finance_agent_model = finance_model
    st.session_state.team_agent_model = team_model
    
    # Create agent team with current model selections
    agent_team = create_agents()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Query input at the bottom
    query = st.chat_input("Enter your query here...")
    
    if query and agent_team:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Get and display agent response
        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Processing your query..."):
                response = handle_query(agent_team, query, response_container)
                
                if response:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })

    # Display error count in sidebar
    if st.session_state.error_count > 0:
        st.sidebar.warning(f"Errors encountered: {st.session_state.error_count}")

    # Add a clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()