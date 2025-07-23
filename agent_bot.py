from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]

llm = ChatOpenAI(model="gpt-3.5-turbo")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    state["messages"].append(response)  # Add AI message to history
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})
