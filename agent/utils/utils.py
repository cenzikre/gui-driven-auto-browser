import requests, base64
from langchain_core.tools import tool


api_url = "http://127.0.0.1:8000"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def openai_image_payload_format(text: str, image_str: str) -> list[dict]:
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_str}"}}
    ]


@tool
def call_action_endpoint(endpoint_name: str, **kwargs) -> list[dict]:
    """
    Call an action endpoint by providing corresponding parameters to execute a browser action;
    A screenshot of the browser viewport will be taken after the action is executed, and returned to the agent
    
    Args:
        endpoint_name (str): The name of the action endpoint to call
        **kwargs: The parameters to pass to the action endpoint

            Available endpoints and their parameters:
            - take_screenshot: use to take a screenshot of the current browser viewport
                wait_time (float = 1.0): the time to wait before taking the screenshot

            - go_to_url: use to navigate to a specific url
                url (str): the url to navigate to

            - move_mouse: use to move the mouse to a specific coordinate
                x (float): the x coordinate to move the mouse to
                y (float): the y coordinate to move the mouse to
                step (int = 1): the number of steps to move the mouse

            - click_mouse: use to click the mouse at a specific coordinate
                x (float): the x coordinate to click the mouse on
                y (float): the y coordinate to click the mouse on

            - scroll: use to scroll the mouse horizontally and vertically
                delta_x (float): the amount to scroll the mouse horizontally
                delta_y (float): the amount to scroll the mouse vertically
                
            - type_at: use to type text at a specific coordinate
                x (float): the x coordinate to type the text at
                y (float): the y coordinate to type the text at
                text (str): the text content to type
        
    Returns (list[dict]):
        The payload to pass to the LLM, including:
        - status_str: the description of the action
        - image_str: the base64 encoded screenshot of the browser viewport after the action is executed
    """
    request = {**kwargs}
    response = requests.post(f"{api_url}/{endpoint_name}", json=request)
    response_json = response.json()
    status_str = response_json["status"]
    image_str = encode_image_to_base64(response_json["screenshot"])
    return openai_image_payload_format(status_str, image_str)







