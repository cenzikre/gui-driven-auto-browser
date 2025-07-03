from playwright.async_api import async_playwright
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from pathlib import Path
from util.api_models import GotoRequest, MoveMouseRequest, MovePathRequest, ClickRequest, GetScreenshotRequest, ScrollRequest, TypeAtRequest, WaitForLoadingRequest, CommonResponse
from functools import wraps
import uvicorn, asyncio, os, shutil, re, base64


"""
Actions:

Parallelizable:
- type_at: type text at given coordinates

Non-parallelizable:
- goto: navigate to given url
- click_mouse: click at given coordinates
- scroll: scroll up/down the page
- refresh_page: refresh the page
- wait: wait for loading to complete

"""


# Initialize Browser Components

app = FastAPI()
browser_lock = asyncio.Lock()

_browser = None
_context = None
_page = None
_session = None

_screenshot_dir = "screenshots"
app.mount("/screenshots", StaticFiles(directory=_screenshot_dir), name="screenshots")
app.mount("/static", StaticFiles(directory="util/static"), name="static")


# helper functions

def with_browser_lock(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        async with browser_lock:
            return await func(*args, **kwargs)
    return wrapped


def with_error_handling(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return JSONResponse(
                content={
                    "status": "success",
                    "data": result,
                    "error": None,
                }
            )
        except Exception as e:
            return JSONResponse(
                content={
                    "status": "error",
                    "data": None,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                },
                status_code=500
            )
    return wrapper


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


# Controller Endpoints

@app.get("/screenshot")
async def take_screenshot():
    global _page
    os.makedirs(_screenshot_dir, exist_ok=True)
    filepath = get_screenshot_path("ts")
    await _page.screenshot(path=filepath, full_page=False)
    return {"message": "screenshot taken", "screenshot": filepath.as_posix()}


@app.post("/take_screenshot")
async def take_screenshot(request: GetScreenshotRequest):
    global _page
    screenshot_path = await get_screenshot(wait_time=request.wait_time)
    return {"message": "screenshot taken", "screenshot": screenshot_path}


@app.get("/screenshot_list")
async def screenshot_list():
    global _screenshot_dir

    if not os.path.exists(_screenshot_dir):
        return JSONResponse(content={"status": "Screenshots directory not found"}, status_code=404)

    screenshots = sorted(
        [f for f in os.listdir(_screenshot_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=extract_timestamp
    )
    return {"message": "screenshots listed", "screenshots": screenshots}


@app.get("/clear_screenshots")
async def clear_screenshots():
    global _screenshot_dir
    shutil.rmtree(_screenshot_dir)
    os.makedirs(_screenshot_dir, exist_ok=True)
    return {"message": "screenshots cleared"}


# Agent Endpoints

@app.post("/goto", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def goto(request: GotoRequest):
    global _page
    await _page.goto(request.url, wait_until="domcontentloaded")
    return {"message": "navigated to " + request.url}


@app.post("/take_screenshot_stream", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def take_screenshot_stream(request: GetScreenshotRequest):
    global _page
    await asyncio.sleep(request.wait_time)
    screenshot_bytes = await _page.screenshot(full_page=False)
    return {"message": "screenshot taken", "image": base64.b64encode(screenshot_bytes).decode("utf-8")}


@app.post("/move_mouse", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def move_mouse(request: MoveMouseRequest):
    global _page
    w, h = get_viewport_size(_page)
    x = request.x * w
    y = request.y * h

    await _page.mouse.move(x, y, steps=request.step)
    return {"message": f"mouse moved to {x:.2f}, {y:.2f} with {request.step} steps"}


@app.post("/move_mouse_path", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def move_mouse_path(request: MovePathRequest):
    global _page
    w, h = get_viewport_size(_page)
    start_x = request.start_x * w
    start_y = request.start_y * h
    end_x = request.end_x * w
    end_y = request.end_y * h

    await _page.mouse.move(start_x, start_y)
    await _page.mouse.move(end_x, end_y, steps=request.steps)
    return {"message": f"Mouse moved from ({start_x:.2f},{start_y:.2f}) to ({end_x:.2f},{end_y:.2f})"}


@app.post("/click_mouse", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def click_mouse(request: ClickRequest):
    global _page

    w, h = get_viewport_size(_page)
    x = request.x * w
    y = request.y * h

    new_page_future = _context.wait_for_event("page")
    await _page.mouse.click(x, y)

    try:
        new_page = await asyncio.wait_for(new_page_future, timeout=1)
        try:
            await new_page.wait_for_load_state("domcontentloaded")
            _page = new_page
            return {"message": f"Mouse clicked at ({x:.2f},{y:.2f}) and navigated to new page"}
        except asyncio.TimeoutError:
            return {"message": f"Mouse clicked at ({x:.2f},{y:.2f}), new page opened but failed to load within 30 seconds"}
    except asyncio.TimeoutError:
        return {"message": f"Mouse clicked at ({x:.2f},{y:.2f}), but no new page was navigated to"}
    

@app.post("/scroll", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def scroll(request: ScrollRequest):
    global _page
    await _page.mouse.wheel(request.delta_x, request.delta_y)
    return {"message": f"scrolled {request.delta_x} {request.delta_y}"}


@app.post("/type_at", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def type_at(request: TypeAtRequest):
    global _page

    w, h = get_viewport_size(_page)
    x = request.x * w
    y = request.y * h

    await _page.mouse.click(x, y)
    await _page.keyboard.type(request.text)
    return {"message": f"typed {request.text} at {x:.2f}, {y:.2f}"}


@app.post("/refresh_page", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def refresh_page():
    global _page
    await _page.reload(wait_until="domcontentloaded")
    return {"message": "Page successfully refreshed"}


@app.post("/wait_for_loading", response_model=CommonResponse)
@with_browser_lock
@with_error_handling
async def wait_for_loading(request: WaitForLoadingRequest):
    global _page
    await _page.wait_for_load_state(request.wait_state, timeout=request.timeout)
    return {"message": "Page finished loading"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)