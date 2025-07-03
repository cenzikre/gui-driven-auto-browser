import requests, base64, io, httpx, json
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict
from PIL import Image
from util.api_models import IconDetectRequest
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, AnyMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command


root = Path(__file__).parent.parent
screenshot_dir = root / "screenshots"
browser_api_url = "http://127.0.0.1:8000"
yolo_api_url = "http://127.0.0.1:8001"

system_msg = SystemMessage(content="""
You are a helpful assistant that helps a human complete tasks by interacting with a real web browser.

You will receive instructions related to web interactions, such as:
- Navigating to a website
- Searching for content or information
- Filling out forms (e.g., login, checkout, payments)
- Clicking buttons or links
- Reviewing or summarizing page content

You can interact with the browser using a set of tools that allow you to:
- Navigate to a specific URL
- Move the mouse to an interactive element
- Click on an interactive element
- Scroll the page
- Type text into input fields

The browser will provide a screenshot of the current page, where **interactable elements** are annotated with bounding boxes and **indexed** for reference. 
You can use these indexes to specify where to type, click, or move the mouse.

### 💡 Action Planning Rules (Batch Execution Model):

- You must respond with a **batch of actions** based on what you see and the current task goal.
- You can include any number of **parallelizable actions**:
  - `type_at`: fill text into multiple input boxes simultaneously
- You may include **at most one non-parallelizable action**, and it must appear at the **end** of the batch:
  - Non-parallelizable actions: `goto`, `click_mouse`, `scroll`, `refresh_page`, `wait`

> ⚠️ Do **not** call the screenshot tool yourself. A screenshot will be taken **automatically after your batch of actions finishes.**

### 🔁 How to Act

1. Observe the screenshot and the list of interactable elements.
2. Decide your next move toward completing the task.
3. Plan a batch of actions:
   - Use `type_at` to fill all necessary fields in one batch.
   - Follow up with `click_mouse` or `goto` if needed, placing them at the end.
4. Wait for the new screenshot after your batch to determine your next step.
5. Repeat until the task is complete.
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

def image_reducer(old: list[AnyMessage], new: list[AnyMessage]) -> list[AnyMessage]:
    if old and has_image_string(old[-1]):
        old[-1] = drop_image_string(old[-1])
    return old + new

def image_reducer2(old: list[AnyMessage], new: list[AnyMessage]) -> list[AnyMessage]:
    if len(old) == 0:
        return old + new
    return [old[0]] + new

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


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], image_reducer]
    image: str | None = None
    centers: list[list[float]] | None = None


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


def organize_actions(actions: list[dict]) -> list[dict]:
    """
    Reorganize the actions into a list of parallelizable actions + 1 * non-parallelizable actions + 1 * take_screenshot action;
    Parallelizable actions: `type_at`
    Non-parallelizable actions: `goto`, `click_mouse`, `scroll`, `refresh_page`, `wait_for_loading`

    Args:
        actions (list[dict]): the list of actions to organize
    
    Returns:
        list[dict]: the list of actions to execute
    """
    parallelizable_action_list = ['type_at']
    non_parallelizable_action_count = 0
    parallelizable_actions = []
    non_parallelizable_actions = []

    for action in actions:
        if action["endpoint_name"] in parallelizable_action_list:
            parallelizable_actions.append(action)
        else:
            non_parallelizable_action_count += 1
            if non_parallelizable_action_count > 1:
                raise ValueError("Only one non-parallelizable action is allowed")
            non_parallelizable_actions.append(action)

    return parallelizable_actions + non_parallelizable_actions + [{"endpoint_name": "take_screenshot_stream"}]


async def get_api_response(api_url: str, endpoint: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        # Make POST request
        try:
            resp = await client.post(f"{api_url}/{endpoint}", json=payload)
        except httpx.RequestError as e:
            # network level error, cannot reach the server
            raise RuntimeError(f"Cound not reach the server: {e}") from e
        
        # Decode the response body
        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Invalid JSON in response: {e}")
        
        # Check if api returns handled error
        if data.get("status") == "error":
            error = data.get("error", {})
            raise RuntimeError(f"API error from {endpoint}: {error.get('type', 'Unknown')} - {error.get('message', 'No details')}")
        
        # Return valid response body
        return data['data']


@tool
async def execute_batch_actions(
    actions: list[dict], 
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Call this tool to sequentially execute the actions you provided;
    Action batch: N * parallelizable actions + 1 * non-parallelizable action (at the end);

    Parallelizable actions: `type_at`
    Non-parallelizable actions: `goto`, `click_mouse`, `scroll`, `refresh_page`, `wait_for_loading`
    
    Requested actions will be executed one by one and a screenshot will be taken at the end of all actions.
    The tool will automatically analyze it with an icon-detection model (YOLO), and store the detection results (the annotated
    image and box coordinates) in the agent's state for use in the next turn.

    If no screenshot is returned by any action in the batch, the agent's previous `image` and `centers` fields will be cleared 
    (set to `None`) to avoid carrying stale information forward.
    
    Args:
        actions (list[dict]): A list of actions to execute, each action is a dictionary with the following keys:
            - endpoint_name (str): the name of the browser API endpoint
            - params (dict): the parameters for that endpoint

                Available endpoints and their expected parameters include:

                - goto: navigate to a specific url
                    url (str): target URL

                - click_mouse: click the mouse at a specific annotated box
                    box_index (int): the index of the annotated box to click the mouse on

                - scroll: scroll the mouse horizontally and vertically
                    delta_x (float): scroll offset horizontally
                    delta_y (float): scroll offset vertically

                - refresh_page: refresh the current page (no parameters)

                - wait_for_loading: wait for the page to finish loading
                    wait_state (str, default "domcontentloaded"): event to wait for, can be "domcontentloaded", "load", "networkidle"
                    timeout (float, default 30000): maximum wait time in milliseconds

                - type_at: type text at a specific annotated box
                    box_index (int): the index of the annotated box to type the text at
                    text (str): the text content to type

            Example: [
                {
                    "endpoint_name": "type_at",
                    "params": {
                        "box_index": 0,
                        "text": "Hello, world!"
                    }
                },
                {
                    "endpoint_name": "click_mouse",
                    "params": {
                        "box_index": 1
                    }
                }
            ]

        state (AgentState): the state of the agent, it will be injected to the tool, do not need to pass it in the tool call
        tool_call_id (str): the id of the tool call, it will be injected to the tool, do not need to pass it in the tool call
    
    Returns:
        Command: contains an update with:
        - a new tool message summarizing the action responses
        - the most recent annotated screenshot and detected box centers, or None if no screenshot was taken
    """

    actions = organize_actions(actions)
    cur_centers = state.get("centers", [])

    log: dict[str, Any] = {}
    new_state: dict[str, Any] = {}
    saw_screenshot = False

    for i, action in enumerate(actions):
        ep, params = action["endpoint_name"], action.get("params", {})
        if ep in ["refresh_page"]:
            params = None

        # resolve box_index to x, y
        if "box_index" in params:
            idx = params.pop("box_index")
            if idx < 0 or idx >= len(cur_centers):
                raise ValueError(f"box_index {idx} out of range, please pick the box with valid index from the previous screenshot")
            params |= {"x": cur_centers[idx][0], "y": cur_centers[idx][1]}

        # call the browser api endpoint
        resp = await get_api_response(browser_api_url, ep, params)

        # if screenshot, run YOLO and cache big blobs only in state
        if 'image' in resp:
            saw_screenshot = True
            det = await get_api_response(
                yolo_api_url,
                "detect_icon",
                IconDetectRequest(source = resp['image']).model_dump()
            )

            new_state['centers'] = det['centers']
            new_state['image'] = det['image']
            resp.pop('image', None)

        # add human readable info to tool call log
        log[f"action_{i}"] = resp

    # if no screenshot, clear the image and centers from previous state
    if not saw_screenshot:
        new_state['image'] = None
        new_state['centers'] = None

    # wrap the whole log into a single tool message
    tool_msg = ToolMessage(
        content=json.dumps(log, indent=2, ensure_ascii=False),
        tool_call_id=tool_call_id
    )

    # return a command to merge 'new_state' and append 'tool_msg'
    return Command(
        update={
            **new_state,
            'messages': [tool_msg]
        }
    )






async def test_execute_batch_actions(
    actions: list[dict], 
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
 
    actions = organize_actions(actions)
    print(actions)
    cur_centers = state.get("centers", [])

    log: dict[str, Any] = {}
    new_state: dict[str, Any] = {}
    saw_screenshot = False

    for i, action in enumerate(actions):
        ep, params = action["endpoint_name"], action.get("params", {})
        if ep in ["refresh_page"]:
            params = None

        # resolve box_index to x, y
        if "box_index" in params:
            idx = params.pop("box_index")
            if idx < 0 or idx >= len(cur_centers):
                raise ValueError(f"box_index {idx} out of range, please pick the box with valid index from the previous screenshot")
            params |= {"x": cur_centers[idx][0], "y": cur_centers[idx][1]}

        # call the browser api endpoint
        print(ep, params)
        resp = await get_api_response(browser_api_url, ep, params)

        # if screenshot, run YOLO and cache big blobs only in state
        if 'image' in resp:
            saw_screenshot = True
            det = await get_api_response(
                yolo_api_url,
                "detect_icon",
                IconDetectRequest(source = resp['image']).model_dump()
            )

            new_state['centers'] = det['centers']
            new_state['image'] = det['image']
            resp.pop('image', None)

        # add human readable info to tool call log
        log[f"action_{i}"] = resp

    # if no screenshot, clear the image and centers from previous state
    if not saw_screenshot:
        new_state['image'] = None
        new_state['centers'] = None

    # wrap the whole log into a single tool message
    tool_msg = ToolMessage(
        content=json.dumps(log, indent=2, ensure_ascii=False),
        tool_call_id=tool_call_id
    )

    # return a command to merge 'new_state' and append 'tool_msg'

    update={
        **new_state,
        'messages': [str(tool_msg)]
    }

    with open('test/test_tool_update.json', 'w') as f:
        json.dump(update, f, indent=2)

