import asyncio, httpx, base64, io, json
from PIL import Image


browser_api_url = "http://127.0.0.1:8000"
yolo_api_url = "http://127.0.0.1:8001"
test_website = "https://wipp.edmundsassoc.com/Wipp/?wippid=23"

goto_request = {
    "endpoint_name": "goto",
    "params": {
        "url": test_website
    }
}

screenshot_request = {
    "endpoint_name": "take_screenshot_stream",
}

click_request = {
    "endpoint_name": "click_mouse",
    "params": {
        "box_index": 17
    }
}

type_request = {
    "endpoint_name": "type_at",
    "params": {
        "text": "1234567890",
        "box_index": 10
    }
}

refresh_request = {
    "endpoint_name": "refresh_page",
}

scroll_request = {
    "endpoint_name": "scroll",
    "params": {
        "delta_x": 0,
        "delta_y": 100
    }
}

async def call_browser_api(request):
    async with httpx.AsyncClient() as client:
        ep = request["endpoint_name"]
        payload = request.get("params", {})
        resp = await client.post(browser_api_url + "/" + ep, json=payload)
        resp = resp.json()
        if resp["status"] == "success":
            return resp["data"]
        else:
            raise Exception(resp["error"]["message"])
        
async def call_yolo_api(request):
    async with httpx.AsyncClient() as client:
        ep = 'detect_icon'
        payload = request
        resp = await client.post(yolo_api_url + "/" + ep, json=payload)
        resp = resp.json()
        print(resp['status'])
        if resp["status"] == "success":
            return resp["data"]
        else:
            raise Exception(resp["error"]["message"])
        
async def get_screenshot():
    browser_resp = await call_browser_api(screenshot_request)
    print(browser_resp['message'])
    image_str = browser_resp['image']
    yolo_resp = await call_yolo_api({'source': image_str})
    image = base64.b64decode(yolo_resp['image'])
    image = Image.open(io.BytesIO(image))
    image.show()

    with open('test_centers.json', 'w') as f:
        json.dump(yolo_resp['centers'], f, indent=2)

    with open('test_image.png', 'wb') as f:
        f.write(base64.b64decode(yolo_resp['image']))

def resolve_box_index(request):
    if "box_index" in request["params"]:
        box_index = request["params"].pop("box_index")
        with open('test_centers.json', 'r') as f:
            centers = json.load(f)
        x, y = centers[box_index]
        request["params"]["x"] = x
        request["params"]["y"] = y
    return request

async def click():
    request = resolve_box_index(click_request)
    browser_resp = await call_browser_api(request)
    print(browser_resp['message'])

async def type():
    request = resolve_box_index(type_request)
    browser_resp = await call_browser_api(request)
    print(browser_resp['message'])

async def refresh():
    browser_resp = await call_browser_api(refresh_request)
    print(browser_resp['message'])

async def scroll():
    request = resolve_box_index(scroll_request)
    browser_resp = await call_browser_api(request)
    print(browser_resp['message'])

if __name__ == "__main__":
    # asyncio.run(call_browser_api(goto_request))
    # asyncio.run(get_screenshot())
    # asyncio.run(click())
    # asyncio.run(type())
    # asyncio.run(refresh())
    asyncio.run(scroll())
    