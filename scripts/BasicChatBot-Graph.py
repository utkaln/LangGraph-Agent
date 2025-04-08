# %%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# %% [markdown]
# ## Basic Chatbot
# - Define a graph and state that persists state of graph
# - When defining a graph, the first step is to define its State. The State includes the graph's schema and reducer functions that handle state updates. In our example, State is a TypedDict with one key: messages. The add_messages reducer function is used to append new messages to the list instead of overwriting it. Keys without a reducer annotation will overwrite previous values. Learn more about state, reducers, and related concepts in this guide.
# 
# ### Define State

# %%
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages

# Define class used for persistence
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)


# %% [markdown]
# ### Define Chatbot Node
# - The Chatbot node function accepts State as input and returns updated list of messages by adding the new response to the input message. This makes the agentic behavior persistent

# %%
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model=os.getenv('CLAUDE_LLM'))
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
def chatbot(state:State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)


# %% [markdown]
# ### Creating a Graph structure
# - LangGraph fundamentally works on the concept of adding Nodes and Edges
# - The start node is the LLM node or the chatbot in this case
# - The end node is defined as END node
# - The tools are added as other nodes
# - The decision edge is the case where the LLM decides to go down the tool route or if the final answer is there then call the End node, thus terminating the transaction

# %%
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Method to create starter message and invoke the graph
def stream_graph(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print(f"Assistant: {value["messages"][-1].content}")
        

# %% [markdown]
# ### User Input to invoke Chatbot

# %%
while True:
    try:
        user_input = input("User: ")
        if(user_input.lower() in ["exit", "quit", "q"]):
            print("Exiting the chatbot!")
            break
        stream_graph(user_input)
    except:
        user_input = "Assume the role of a helpful assistant. Tell me how do I interact with you to ask you questions?"
        stream_graph(user_input)
        break
    


