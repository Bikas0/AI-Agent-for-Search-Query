from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

agent = Agent(
    model = Groq(id= "llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations = True, stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display data."]
)

agent.print_response("Summarize and compare analyst recommendations and fundamentals for TESLA and NVIDIA")