from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent


load_dotenv()

def get_weather(city: str) :
    """Get weather for a given city"""

    return {'condition': 'sunny', 'temperature': '25'}

# initialize llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

agent = create_agent(
    model=llm,
    tools=[get_weather]
)

# response1 = llm.invoke("How is the weather in Rome?")
response1 = agent.invoke(
    {"messages": [{'role': 'user',
                   'content': 'How is the weather in Rome'}]})
print(response1['messages'][-1].content)