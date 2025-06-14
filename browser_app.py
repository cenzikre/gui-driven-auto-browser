from playwright.async_api import async_playwright
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import uvicorn, asyncio, os, shutil, re
from typing import Optional


app = FastAPI()

_browser = None
_context = None
_page = None
_session = None

_screenshot_dir = "play_wright/screenshots"
app.mount("/screenshots", StaticFiles(directory=_screenshot_dir), name="screenshots")
app.mount("/static", StaticFiles(directory="play_wright/static"), name="static")


# API request models

class GotoRequest(BaseModel):
    url: str

class MoveMouseRequest(BaseModel):
    x: float
    y: float
    step: Optional[int] = 1

class MovePathRequest(BaseModel):
    start_x: float  # normalized [0, 1]
    start_y: float
    end_x: float
    end_y: float
    steps: int = 20  # number of intermediate steps

class ClickRequest(BaseModel):
    x: float
    y: float

class GetScreenshotRequest(BaseModel):
    wait_time: float = 1.0

class ScrollRequest(BaseModel):
    delta_x: float
    delta_y: float

class TypeAtRequest(BaseModel):
    x: float
    y: float
    text: str


# helper functions

def extract_timestamp(filename: str) -> str:
    match = re.search(r"\d{8}_\d{6}", filename)
    return match.group() if match else ""


def get_viewport_size(page):
    viewport_size = page.viewport_size
    if viewport_size is None:
        raise ValueError("viewport size not available")
    w, h = viewport_size["width"], viewport_size["height"]
    return w, h


def get_screenshot_path(screenshot_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{screenshot_type}_{timestamp}.png"
    filepath = Path(_screenshot_dir) / filename
    return filepath.as_posix()


async def get_screenshot(screenshot_type: str = "default", wait_time: float = 1.0) -> str:
    global _page
    await asyncio.sleep(wait_time)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{screenshot_type}_{timestamp}.png"
    filepath = Path(_screenshot_dir) / filename

    await _page.screenshot(path=filepath, full_page=False)
    return filename


# API endpoints

@app.on_event("startup")
async def startup_browser():
    global _browser, _context, _page, _session
    _session = await async_playwright().start()
    _browser = await _session.chromium.launch(headless=False)
    _context = await _browser.new_context()
    _page = await _context.new_page()


@app.post("/goto")
async def goto(request: GotoRequest):
    global _page
    # print("goto", request.url)
    await _page.goto(request.url, wait_until="domcontentloaded")

    screenshot_path = await get_screenshot("goto")
    return {
        "status": "navigated to " + request.url, 
        "screenshot": screenshot_path
    }


@app.get("/screenshot")
async def take_screenshot():
    global _page

    os.makedirs(_screenshot_dir, exist_ok=True)

    filepath = get_screenshot_path("ts")
    await _page.screenshot(path=filepath, full_page=False)

    return {"status": "screenshot taken", "screenshot": filepath.as_posix()}


@app.post("/take_screenshot")
async def take_screenshot(request: GetScreenshotRequest):
    global _page
    screenshot_path = await get_screenshot(wait_time=request.wait_time)
    return {"status": "screenshot taken", "screenshot": screenshot_path}


@app.get("/screenshot_list")
async def screenshot_list():
    global _screenshot_dir

    if not os.path.exists(_screenshot_dir):
        return JSONResponse(content={"status": "Screenshots directory not found"}, status_code=404)

    screenshots = sorted(
        [f for f in os.listdir(_screenshot_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=extract_timestamp
    )
    return {"status": "screenshots listed", "screenshots": screenshots}


@app.get("/clear_screenshots")
async def clear_screenshots():
    global _screenshot_dir
    shutil.rmtree(_screenshot_dir)
    os.makedirs(_screenshot_dir, exist_ok=True)
    return {"status": "screenshots cleared"}


@app.post("/move_mouse")
async def move_mouse(request: MoveMouseRequest):
    global _page

    w, h = get_viewport_size(_page)
    x = request.x * w
    y = request.y * h

    await _page.mouse.move(x, y, steps=request.step)
    screenshot_path = await get_screenshot("move_mouse")
    return {
        "status": f"mouse moved to {x:.2f}, {y:.2f} with {request.step} steps",
        "screenshot": screenshot_path
    }


@app.post("/move_mouse_path")
async def move_mouse_path(request: MovePathRequest):
    global _page

    w, h = get_viewport_size(_page)
    start_x = request.start_x * w
    start_y = request.start_y * h
    end_x = request.end_x * w
    end_y = request.end_y * h

    await _page.mouse.move(start_x, start_y)
    await _page.mouse.move(end_x, end_y, steps=request.steps)

    screenshot_path = await get_screenshot("move_mouse_path")
    return {
        "status": f"Mouse moved from ({start_x:.2f},{start_y:.2f}) to ({end_x:.2f},{end_y:.2f})",
        "screenshot": screenshot_path
    }


@app.post("/click_mouse")
async def click_mouse(request: ClickRequest):
    global _page

    w, h = get_viewport_size(_page)
    x = request.x * w
    y = request.y * h

    # Prepare to listen for new page event
    new_page_future = _context.wait_for_event("page")

    # Perform the click
    await _page.mouse.click(x, y)

    try:
        new_page = await asyncio.wait_for(new_page_future, timeout=1)
        await new_page.wait_for_load_state("domcontentloaded")
        _page = new_page
        screenshot_path = await get_screenshot("click_mouse")
        return {
            "status": f"Mouse clicked at ({x:.2f},{y:.2f}) and navigated to new page",
            "screenshot": screenshot_path
        }
    except asyncio.TimeoutError:
        screenshot_path = await get_screenshot("click_mouse")
        return {
            "status": f"Mouse clicked at ({x:.2f},{y:.2f})",
            "screenshot": screenshot_path
        }
    

@app.post("/scroll")
async def scroll(request: ScrollRequest):
    global _page
    await _page.mouse.wheel(request.delta_x, request.delta_y)
    screenshot_path = await get_screenshot("scroll")
    return {
        "status": f"scrolled {request.delta_x} {request.delta_y}", 
        "screenshot": screenshot_path
    }


@app.post("/type_at")
async def type_at(request: TypeAtRequest):
    global _page

    w, h = get_viewport_size(_page)
    x = request.x * w
    y = request.y * h

    await _page.mouse.click(x, y)
    await _page.keyboard.type(request.text)
    screenshot_path = await get_screenshot("type_at")

    return {
        "status": f"typed {request.text} at {x:.2f}, {y:.2f}", 
        "screenshot": screenshot_path
    }

# @app.on_event("shutdown")
# async def shutdown_browser():
#     global _browser, _context, _page, _session
#     if _page:
#         await _page.close()
#     if _context:
#         await _context.close()
#     if _browser:
#         await _browser.close()
#     if _session:
#         await _session.stop()


# @app.on_event("shutdown")
# async def shutdown_browser():
#     global _browser, _context, _page, _session

#     await asyncio.gather(
#         _page.close() if _page and not _page.is_closed() else asyncio.sleep(0),
#         _context.close() if _context else asyncio.sleep(0),
#         _browser.close() if _browser else asyncio.sleep(0),
#         _session.stop() if _session else asyncio.sleep(0),
#         return_exceptions=True
#     )


@app.on_event("shutdown")
async def shutdown_browser():
    global _playwright, _browser, _context, _page
    try:
        if _page and not _page.is_closed():
            await _page.close()
    except Exception as e:
        print(f"Warning: page close failed: {e}")

    try:
        if _context:
            await _context.close()
    except Exception as e:
        print(f"Warning: context close failed: {e}")

    try:
        if _browser:
            await _browser.close()
    except Exception as e:
        print(f"Warning: browser close failed: {e}")

    try:
        if _session:
            await _session.stop()
    except Exception as e:
        print(f"Warning: playwright stop failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
