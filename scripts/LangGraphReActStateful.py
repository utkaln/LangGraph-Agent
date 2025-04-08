# %% [markdown]
# #### Make Agentic Session Stateful for the user
# - To achive session statefulness a checkpointer is provided by Langraph
# - A database is used to keep track of checkpointer
# - For quick demo, a in memory database is used
# - Pass the checkpointer to graph.compile step
# - Add a thread number assigned to user session to keep track of the conversation
# - Modify the LangGraph to Stream by changing the `invoke` call to `stream` call

# %%
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# %%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langgraph.graph import StateGraph, END 
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Define tool that is available to langraph that an action edge can find
tool = TavilySearchResults(max_results=3)
print(f"tool name -> {tool.name}")


# %% [markdown]
# #### Define A place holder for all the messages that is known as AgentState
# - This is a list of messages which keeps adding a new message everytime it is called

# %%
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# %% [markdown]
# #### Define the Agent Class that performs:
# 1. Call LLM in this example OpenAI
# 2. Check if Action is present
# 3. Take Action
# 
# Steps in the code 
# 1. The constructor function takes model name, available tools choices and system prompt
# 2. Start a LLM node
# 3. Then add an action node
# 4. Then define an action edge to link between LLM and action node. 
# 5. If no action decision made by LLM, then send to END node
# 6. Create an edge to loop back to LLM node from Action Node
# 7. Compile the graph and save it as class level attribute
# 8. Create a dictionary of tools sent as parameters and save as class level attribute
# 9. Save the tool name that sent as input as a class level attribute under the model

# %%
class Agent:
    def __init__(self,model,tools,checkpointer, system=""):
        # Save the system message as a class level attribute
        self.system = system

        # Initialize the state graph that will have one LLM node, One Tool node and one Action Edge
        graph = StateGraph(AgentState)
        # Start Node
        graph.add_node("llm",self.call_openai)
        
        # Action Node that is available as tool
        graph.add_node("action",self.action_node)

        # Decision Edge to decide to use action node
        # First parameter is the node from which this edge is coming from 
        # Second parameter is the function that let's langraph explore tools
        # Third parameter is available nodes after the decision either action node or END node
        graph.add_conditional_edges("llm",self.action_edge,{True: "action", False: END})

        # Create Another edge to loop back to LLM node from action node
        graph.add_edge("action","llm")

        # Define what node the graph should start, in this case the llm
        graph.set_entry_point("llm")

        # Once setup done compile the graph and Save the graph at the class level
        self.graph = graph.compile(checkpointer=checkpointer)

        # Create a dictionary of available tools sent to the constructor
        self.tools = {tool.name: tool for tool in tools}

        # Bind tools to model so that LLM can search for tools
        self.model = model.bind_tools(tools)


    # Define function for call llm node
    def call_openai(self, state: AgentState):
        # get the messages saved in the Agent state object
        messages = state["messages"]
        # If system message is not blank, append that to the beginning of the messages
        if self.system:
            system_message= [SystemMessage(content=self.system)]
            messages =  system_message + messages
        # Call the model with the messages, it should return response as a single message
        resp = self.model.invoke(messages)
        print(f"Response from LLM -> {resp}")
        # Return the response message as a list, that will be appended to the existing messages due to operator.add annotation at class level
        return {"messages": [resp]}
    
    # Define function for call action node
    def action_node(self, state: AgentState):
        # get the last message from the Agent State, since the last message is the response from LLM that suggests to use the tool
        # tool calls attribute is expected which has the name of the tool to be called
        referred_tools_list = state["messages"][-1].tool_calls
        results = []

        # Tool calls can be multiple tools, so iterate over them
        for tool in referred_tools_list:
            print(f"Tool to be called -> {tool}")
            # invoke tool call by finding the name and the arguments as suggested by LLM
            result = self.tools[tool["name"]].invoke(tool["args"])
            # Append the result to the results list
            results.append(ToolMessage(tool_call_id=tool["id"], name=tool["name"], content=str(result)))
            
        print(f"Finished tool call ...")
        # returns results and add to the messages list at class level
        return {"messages": results}
    
    # Define the actiton edge function that decides whether to look for tool or not
    # If the last message in the message list has tool_calls attribute, then return True, else False
    def action_edge(self, state: AgentState):
        result = state["messages"][-1]
        return  len(result.tool_calls) > 0
        
     

# %% [markdown]
# #### Define a chat model with system prompt

# %%
system_prompt = prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(model=os.getenv('OPENAI_LLM_GPT_4_O_mini'))
ai_agent = Agent(model,[tool],memory, system_prompt)

# %% [markdown]
# #### Invoke the langraph with user message as input

# %%
# Add a thread id to make the conversation persistent
thread_id = {"configurable": {"thread_id":"1"}}
while True:
    try:
        user_input = input("User: ")
        if(user_input.lower() in ["exit", "quit", "q"]):
            print("Exiting the chatbot!")
            break
        messages = [HumanMessage(content=user_input)]
        events = ai_agent.graph.stream({"messages": messages},thread_id,stream_mode="values")
        for event in events:
            event["messages"][-1].pretty_print()
    except:
        user_input = "Assume the role of a helpful assistant. Tell me how do I interact with you to ask you questions?"
        messages = [HumanMessage(content=user_input)]
        events = ai_agent.graph.stream({"messages": messages},thread_id,stream_mode="values")
        for event in events:
            event["messages"][-1].pretty_print()

