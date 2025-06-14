import requests, base64, io
from pathlib import Path
from typing import Union, Annotated
from PIL import Image
from omni.util.utils import IconDetectRequest
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.graph import MessagesState


root = Path(__file__).parent.parent.parent
screenshot_dir = root / "play_wright" / "screenshots"
browser_api_url = "http://127.0.0.1:8000"
yolo_api_url = "http://127.0.0.1:8001"

system_msg = SystemMessage(content="""
You are a helpful assistant that help human with browser automation tasks. 
You are going to be given a task instructions related to web page interactions, e.g.
    - go to a specific website
    - search for some information or content
    - looking for specific items on a shopping website
    - login to account to summarize information, or make a payment
    - etc.

Through a set of tools provided to you, you can interact with a web browser to:
    - navigate to a specific url
    - see the browser viewport by taking screenshots
    - interactable elements in the screenshots will be annotated with bounding boxes and indexes, 
      and you can pick the index of the element you want to interact with by their indexes
                           
Then based on what you see, you can:
    - move the mouse to a specific interactive box
    - click the mouse at a specific interactive box
    - scroll the mouse horizontally and vertically
    - type text at a specific interactive box

1 second after executing an action, a auto screenshot of the browser viewport will be taken to try to capture the effect of your action.
You can also manually set wait time and take screenshot if the auto screenshot is showing loading web page.
Interactable elements in the screenshots will be annotated with bounding boxes and indexes.
Based on the information, you can plan your next move. 
Repeat the process until the task is completed.
""")

task_msg_template = ChatPromptTemplate.from_template("""
<task instructions>
{task_instructions}
</task instructions>

For every response, beside answering directly question, and generating tool calls, please provide the following information:
    Observation:
        - What you see on task instructions, or on the browser viewport
    Thought:
        - What you think about the observation, and how you plan for the following actions to complete the task
    Action:
        - What you want to do for the next step
""")


class AgentState(MessagesState):
    image_path: str = ""
    box_centers: list[list[float]] = []


def encode_image_to_base64(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    
def openai_image_payload_format(text: str, image_str: str) -> list[dict]:
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_str}"}}
    ]

def has_image_string(message: AnyMessage) -> bool:
    if not isinstance(message.content, list):
        return False
    return any(
        part.get("type") == "image_url" and part.get("image_url", {}).get("url", "").startswith("data:image/")
        for part in message.content
    )

def drop_image_string(message: AnyMessage) -> AnyMessage:
    if not isinstance(message.content, list):
        return message
    filtered = [part for part in message.content if part.get("type") != "image_url"]
    return message.model_copy(update={"content": filtered})

def annotate_image(image_str: str) -> Image.Image:
    payload = IconDetectRequest(source = image_str)
    response = requests.post(f"{yolo_api_url}/detect_icon", json=payload.model_dump())
    return response

def call_action_endpoint_function(endpoint_name: str, **kwargs) -> list[dict]:
    request = {**kwargs}
    response = requests.post(f"{browser_api_url}/{endpoint_name}", json=request)
    response_json = response.json()
    screenshot_path = screenshot_dir / response_json["screenshot"]
    response_json["screenshot"] = screenshot_path.as_posix()
    return response_json


@tool
def call_action_endpoint(endpoint_name: str, params: dict, state: Annotated[AgentState, InjectedState]) -> list[dict]:
    """
    Call an action endpoint by providing corresponding parameters to execute a browser action;
    A screenshot of the browser viewport will be taken after the action is executed, all interactive elements will be annotated 
    and returned to the agent.
    
    Args:
        endpoint_name (str): The name of the action endpoint to call
        params (dict): The parameters to pass to the action endpoint

            Available endpoints and their parameters:
            - take_screenshot: use to take a screenshot of the current browser viewport
                wait_time (float = 1.0): the time to wait before taking the screenshot

            - goto: use to navigate to a specific url
                url (str): the url to navigate to

            - move_mouse: use to move the mouse to a specific coordinate
                box_index (int): the index of the annotated box to move the mouse to
                step (int = 1): the number of steps to move the mouse

            - click_mouse: use to click the mouse at a specific coordinate
                box_index (int): the index of the annotated box to click the mouse on

            - scroll: use to scroll the mouse horizontally and vertically
                delta_x (float): the amount to scroll the mouse horizontally
                delta_y (float): the amount to scroll the mouse vertically
                
            - type_at: use to type text at a specific coordinate
                box_index (int): the index of the annotated box to type the text at
                text (str): the text content to type

        state (AgentState): the state of the agent, it will be injected to the tool, do not need to pass it in the tool call
        
    Returns (dict):
        Response paylaod from the browser app, including:
        - status (str): the description of the action
        - screenshot (str): the path to the annotated screenshot of the browser viewport after the action is executed
        - centers (list[list[float]]): the centers of the annotated boxes
    """

    if "box_index" in params:
        box_index = params["box_index"]
        request = {k: v for k, v in params.items() if k != "box_index"}
        if box_index < 0 or box_index >= len(state["box_centers"]):
            raise ValueError(f"Box index {box_index} is out of range, Please pick the box with valid index from the previous screenshot")
        request["x"] = state["box_centers"][box_index][0]
        request["y"] = state["box_centers"][box_index][1]
    else:
        request = params

    response = requests.post(f"{browser_api_url}/{endpoint_name}", json=request)
    response_json = response.json()
    screenshot_path = screenshot_dir / response_json["screenshot"]  

    annotated_repsonse = annotate_image(response_json["screenshot"])
    annotated_response_json = annotated_repsonse.json()

    centers = annotated_response_json['centers']
    annotated_image_str = annotated_response_json['image']
    annotated_image = Image.open(io.BytesIO(base64.b64decode(annotated_image_str)))
    annotated_screenshot_path = screenshot_dir / f"annotated_{response_json['screenshot']}"
    annotated_image.save(annotated_screenshot_path)

    response_json["screenshot"] = annotated_screenshot_path.as_posix()
    response_json["centers"] = centers
    return response_json







