import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from util.agent_utils import call_action_endpoint, task_msg_template, system_msg, drop_image_string, has_image_string
from util.agent_utils import openai_image_payload_format, AgentState
from util.image_utils import encode_image_to_base64
from task_prompt import task_prompt

load_dotenv()

tools = [call_action_endpoint]
tools_node = ToolNode(tools)
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def assistant_node(state: AgentState) -> AgentState:
    return {"messages": [llm.invoke(state["messages"])]}

def attach_image_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    content = json.loads(last_message.content)
    if isinstance(last_message, ToolMessage) and "screenshot" in content:
        image_str = encode_image_to_base64(content["screenshot"])
        text = "Here is the screenshot of the action you just took."
        message = HumanMessage(content=openai_image_payload_format(text, image_str))
        if "centers" in content:
            return {"messages": [message], "image_path": content["screenshot"], "box_centers": content["centers"]}
        else:
            return {"messages": [message], "image_path": content["screenshot"]}
    return {"messages": []}


g = StateGraph(AgentState)
g.add_node("assistant", assistant_node)
g.add_node("tools", tools_node)
g.add_node("attach_image", attach_image_node)
g.add_edge(START, "assistant")
g.add_conditional_edges("assistant", tools_condition, "tools")
g.add_edge("tools", "attach_image")
g.add_edge("attach_image", "assistant")

graph = g.compile()


task_msg = task_msg_template.format_messages(task_instructions=task_prompt)

response = graph.invoke({"messages": [system_msg, *task_msg], "image_path": "", "box_centers": []})
for m in response["messages"]:
    if has_image_string(m):
        m = drop_image_string(m)
    m.pretty_print()

