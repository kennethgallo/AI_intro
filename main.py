# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

def get_weather(city: str)-> dict:
    """Get weather for a given city"""

    return {'condition': 'sunny', 'temperature': '25'}

def get_location() ->str:
    """Get the user's location. Use this when the user asks about the weather without specifying a city"""

    return "Rome, Italy"

# initialize llm
llm = ChatOpenAI(
    model="gpt-5.4-nano-2026-03-17",
    temperature=0.7
)

system_prompt="""
You are a helpful weather assistant. 
YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location, you MUST:
    - First call get_location() to find their location
    - Then call get_weather(city) with that location
    
2. If the user provides a city, call get_weather(city) directly.
"""
agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

while True:
    user_query = input("Enter your query: ")
    if user_query in ['bye', 'quit', 'exit']:
        break
    response = agent.invoke({"messages": [{'role': 'user', 'content': user_query}]},
                             {"configurable" : {"thread_id": "1"} })

    print(response['messages'][-1].content)
