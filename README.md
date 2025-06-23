# GUI-Driven Auto Browser

An AI agent framework for browser automation using Large Vision-Language Models (LVLMs) and computer vision. 
The system enables intelligent agents to interact with a web browser by detecting GUI components through vision models, planning actions via LVLMs, and executing them through structured APIs.

---

## Overview

This project integrates a YOLOv8-based object detector with a LangGraph-powered AI agent that can:

- Visually perceive browser UI via screenshots
- Identify actionable elements (buttons, inputs, etc.)
- Reason through tasks using a Large Vision-Language Model (LVLM)
- Execute web automation steps using callable browser actions

It serves as a foundation for vision-guided, multimodal browser automation agents.

---

## Key Technologies

- **YOLOv8**: Real-time GUI element detection
- **FastAPI**: Backend for exposing browser actions and visual context
- **LangGraph**: Orchestrates agent reasoning and state transitions
- **LVLMs (e.g., OpenAI GPT-4o)**: Understands visual scenes and generates action plans
- **Python + PIL + base64**: For image processing and communication

---

## Model Weights

This project uses a YOLOv8 model trained to detect browser GUI elements, based on Microsoft’s OmniParser detecting component.
The weights are not included, please access the model weights through Microsoft OmniParser.

---

## Example Flow

1. Agent requests a screenshot
2. YOLO detects UI elements
3. LVLM identifies target (e.g., "Login" button)
4. Agent calls action endpoint (move_mouse, goto, etc.)
5. Screenshot updates and loop continues

---
