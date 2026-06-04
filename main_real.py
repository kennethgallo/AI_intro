import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
import os

load_dotenv()


def get_weather(city: str):
    """Get weather for a given city.
    Return the temperature_value in Fahrenheit label for locations such as US, Liberia, Burma"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    temperature_celsius = data['main']['temp']
    temperature_fahrenheit = temperature_celsius * 9/5 + 32

    return data, {'temperature_fahrenheit': temperature_fahrenheit}


def get_location():
    """Get the user's location. Use this when the user asks about the weather without specifying a city"""
    ip_api_url = "https://ipapi.co/json"
    response = requests.get(ip_api_url, headers={'User-agent': 'your-bot 0.1'})
    data = response.json()
    city = data['city']
    country = data['country_name']
    return f"{city}, {country}"


# initialize llm
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

system_prompt = """
You are a helpful weather assistant. 
YOUR WORKFLOW:
1. If the user asks about weather WITHOUT specifying a location, you MUST:
    - First call get_location() to find their location
    - Then call get_weather(city) with that location

2. If the user provides a city, call get_weather(city) directly.

3. Use your knowledge to determine which temperature unit is standard for the given location.
"""
agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt
)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    # response1 = llm.invoke("How is the weather in Rome?")
    response1 = agent.invoke(
        {"messages": [{'role': 'user',
                       'content': user_query}]})
    print(response1['messages'][-1].content)