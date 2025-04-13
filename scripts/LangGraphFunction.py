# %%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages.tool import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from typing import Literal


# %%
system_prompt = """ You are an event manager planning a seminar about various software engineering topics. 
The user will ask you to inquire about topics of the seminar.
The user will ask you to register for certain sessions, you need to ask them the required information and register for the topics.
Always confirm with user before registering them for any topic.
Allow the user to modify or cancel their registration.
The user may inquire about their existing registrations, you need to provide them with the list of topics they are registered for.
If certain topic that is not available as session, then note down the feedback from the user and thank them. But do not register for any topic that you are not provided with.
Keep the chat limited to only the seminar topics and registration.
If any of the tools are not available, then you need to inform the user about it and tell that this will be added in the next version.
Do not engage into any emotional or intimate conversations with the user, politely decline if the user starts such a topic.
At the end of a successful registration offer the user to book nearby hotels and suggest food options.
"""

welcome_prompt = """Welcome to the Technology seminar of North Maryland Area. How can I assist you today? When you are done chatting, please write bye to end"""


# %%
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool




# %% [markdown]
# ### Define LLM node and other nodes for the State Graph

# %%
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
def chatbot(state: OrderState) -> OrderState:
    message_history = system_prompt + state["messages"]
    return{"messages": [llm.invoke(message_history)]}

# Define human node for user input
def human_node(state: OrderState) -> OrderState:
    user_input = input("User: " )
    if user_input.lower() == "bye":
        state["finished"] = True
        return state
    state["messages"].append(user_input)
    return state

# Define start node LLM with Welcome message
def chat_node_welcome(state: OrderState) -> OrderState:
    if state["messages"]:
        chat_ouput = llm.invoke(system_prompt + state["messages"])
    else:
        chat_ouput = AIMessage(welcome_prompt)
    state["messages"].append(chat_ouput)
    return state





# %% [markdown]
# ### Build State Graph

# %%
# Set up the graph
graph_builder = StateGraph(OrderState)
graph_builder.add_node("chatbot", chat_node_welcome)
graph_builder.add_node("human", human_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human")
graph_builder.add_conditional_edges("human",end_chat_check,{True: "chatbot", False: END})
chat_graph = graph_builder.compile()

# %%
# Optional for displaying the graph
from IPython.display import Image, display
Image(chat_graph.get_graph().draw_mermaid_png())

# %% [markdown]
# ### Invoke Model
# - Apply recursive limit so that the chat will end after these many calls

# %%
config = {"recursion_limit": 100}
state = chat_graph.invoke({"messages":[]}, config=config)
print(f"state : {state}")


