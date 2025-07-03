from util.agent_utils import test_execute_batch_actions
from PIL import Image
import json, asyncio, io, base64


with open('test/test_centers.json', 'r') as f:
    centers = json.load(f)

state = {
    "centers": centers
}

actions = [
    {
        "endpoint_name": "type_at",
        "params": {"text": "1234567890", "box_index": 10}
    },
    {
        "endpoint_name": "type_at",
        "params": {"text": "1234567890", "box_index": 9}
    },
]

asyncio.run(test_execute_batch_actions(actions, state, "test_tool_call_id"))

with open('test/test_tool_update.json', 'r') as f:
    update = json.load(f)

image_str = update['image']
image = Image.open(io.BytesIO(base64.b64decode(image_str)))
image.show()