import requests, base64
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate

root = Path(__file__).parent.parent.parent
screenshot_dir = root / "playwright" / "screenshots"
api_url = "http://127.0.0.1:8000"

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
                           
Then based on what you see, you can:
    - move the mouse to a specific coordinate
    - click the mouse at a specific coordinate
    - scroll the mouse horizontally and vertically
    - type text at a specific coordinate

1 second after executing an action, a auto screenshot of the browser viewport will be taken to try to capture the effect of your action.
You can also manually set wait time and take screenshot if the auto screenshot is showing loading web page.
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


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
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

def call_action_endpoint_function(endpoint_name: str, **kwargs) -> list[dict]:
    request = {**kwargs}
    response = requests.post(f"{api_url}/{endpoint_name}", json=request)
    response_json = response.json()
    screenshot_path = screenshot_dir / response_json["screenshot"]
    response_json["screenshot"] = screenshot_path.as_posix()
    return response_json


@tool
def call_action_endpoint(endpoint_name: str, params: dict) -> list[dict]:
    """
    Call an action endpoint by providing corresponding parameters to execute a browser action;
    A screenshot of the browser viewport will be taken after the action is executed, and returned to the agent
    
    Args:
        endpoint_name (str): The name of the action endpoint to call
        params (dict): The parameters to pass to the action endpoint

            Available endpoints and their parameters:
            - take_screenshot: use to take a screenshot of the current browser viewport
                wait_time (float = 1.0): the time to wait before taking the screenshot

            - goto: use to navigate to a specific url
                url (str): the url to navigate to

            - move_mouse: use to move the mouse to a specific coordinate
                x (float): the relative x coordinate (0-1) to move the mouse to
                y (float): the relative y coordinate (0-1) to move the mouse to
                step (int = 1): the number of steps to move the mouse

            - click_mouse: use to click the mouse at a specific coordinate
                x (float): the relative x coordinate (0-1) to click the mouse on
                y (float): the relative y coordinate (0-1) to click the mouse on

            - scroll: use to scroll the mouse horizontally and vertically
                delta_x (float): the amount to scroll the mouse horizontally
                delta_y (float): the amount to scroll the mouse vertically
                
            - type_at: use to type text at a specific coordinate
                x (float): the relative x coordinate (0-1) to type the text at
                y (float): the relative y coordinate (0-1) to type the text at
                text (str): the text content to type
        
    Returns (list[dict]):
        The payload to pass to the LLM, including:
        - status_str: the description of the action
        - image_str: the base64 encoded screenshot of the browser viewport after the action is executed
    """

    response = requests.post(f"{api_url}/{endpoint_name}", json=params)
    response_json = response.json()
    screenshot_path = screenshot_dir / response_json["screenshot"]  
    response_json["screenshot"] = screenshot_path.as_posix()
    return response_json







