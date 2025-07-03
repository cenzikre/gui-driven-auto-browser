import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from util.agent_utils import execute_batch_actions, task_msg_template, system_msg, drop_image_string, has_image_string
from util.agent_utils import openai_image_payload_format, AgentState
from task_prompt import task_prompt

load_dotenv()

tools = [execute_batch_actions]
tools_node = ToolNode(tools)
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def assistant_node(state: AgentState) -> AgentState:
    return {"messages": [llm.invoke(state["messages"])]}

def attach_image_node(state: AgentState) -> AgentState:
    image = state.get("image")
    if image is None or image == "":
        return {"messages": []}
    
    text = "Here is the screenshot of the action you just took."
    human_msg = HumanMessage(content=openai_image_payload_format(text, image))
    return {"messages": [human_msg]}


g = StateGraph(AgentState)
g.add_node("assistant", assistant_node)
g.add_node("tools", tools_node)
g.add_node("attach_image", attach_image_node)
g.add_edge(START, "assistant")
g.add_conditional_edges("assistant", tools_condition, "tools")
g.add_edge("tools", "attach_image")
g.add_edge("attach_image", "assistant")

graph = g.compile()


async def main():
    task_msg = task_msg_template.format_messages(task_instructions=task_prompt)
    response = await graph.ainvoke({
        "messages": [system_msg, *task_msg], 
        "image": None, 
        "centers": None
    })
    for m in response["messages"]:
        if has_image_string(m):
            m = drop_image_string(m)
        m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())

