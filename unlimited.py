from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent,AgentExecutor
load_dotenv()

history = []

gemini_api_key = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature = 0.3
)
@tool
def wifi():
    """the Wifi password is cooking 123  """



@tool
def kayak():
    """ the kayak is located in the other gates by the shed"""

tools = [wifi,kayak]
system_prompt = "Your Ai chatbot designed for airbnb"
user_input = input("You the User:")

prompt = ChatPromptTemplate([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history",optional=True),
    ("user","{input}"),
    MessagesPlaceholder("agent_scratchpad")])

#chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm,tools,prompt)
agent_excutor = AgentExecutor(agent = agent,tools = tools, verbose = False)

#response = chain.invoke({"input": user_input})
response = agent_excutor.invoke({"input":user_input,"chat_history":history})




while True:
    user_input = input("You the User:")
    response = agent_excutor.invoke({"input":user_input,"chat_history":history})
    print(response["output"])
    history.append(HumanMessage(content = user_input))
    history.append(AIMessage(content = response["output"]))
