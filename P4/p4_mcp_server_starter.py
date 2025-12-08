"""
Hybrid MCP Architecture for Voice-Controlled UI Automation

This module implements a hybrid architecture combining:
- mcp-agent: Provides workflow orchestration with guaranteed sequential execution
- FastMCP: Provides streamable HTTP transport on port 3000

The system enables voice-controlled screen interaction by:
1. Capturing screenshots of the current screen
2. Running OCR to detect UI elements and their positions
3. Mapping numbered labels to UI elements
4. Executing mouse clicks on target elements based on voice commands

"""

import os

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch

torch.set_default_device('cpu')
torch.set_num_threads(1)

# =============================================================================
# IMPORTS
# =============================================================================
from mcp.server.fastmcp import FastMCP
from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from datetime import timedelta

import tempfile
import json
import re
import logging
from PIL import ImageGrab
from paddleocr import PaddleOCR
from gtts import gTTS
from playsound3 import playsound
import threading
from typing import List
from pathlib import Path
import pyautogui


# =============================================================================
# TEXT-TO-SPEECH UTILITIES
# =============================================================================

def speak_text(text: str) -> None:
    """Speak text aloud using Google TTS (runs in background thread)."""
    def speak():
        try:
            print(f">>> TTS: Speaking: {text[:50]}...")
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"tts_{hash(text)}.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(audio_file)
            playsound(audio_file)
            try:
                os.remove(audio_file)
            except:
                pass
        except Exception as e:
            print(f">>> TTS Error: {e}")

    threading.Thread(target=speak, daemon=True).start()


# =============================================================================
# SCREEN DIMENSION DETECTION
# =============================================================================
screen_width, screen_height = pyautogui.size()
print(f"Screen size (pyautogui): {screen_width}x{screen_height}")

screenshot = ImageGrab.grab()
img_width, img_height = screenshot.size
print(f"Screenshot size: {img_width}x{img_height}")

scale_x = screen_width / img_width
scale_y = screen_height / img_height
print(f"Scaling factors: x={scale_x}, y={scale_y}")


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MCP SERVER AND AGENT INITIALIZATION
# =============================================================================
mcp_server = FastMCP("P4", port=3000)
mcp_agent_app = MCPApp(name="workflow-engine")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_env_file() -> dict:
    """Load environment variables from .env file."""
    env_path = Path(".env")
    env_vars = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


# =============================================================================
# OCR SINGLETON
# =============================================================================
_ocr_instance = None


def get_ocr() -> PaddleOCR:
    """Get or create singleton PaddleOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_instance


def extract_ascii_digits(text: str) -> str:
    """Extract only ASCII digits (0-9) from a text string."""
    return ''.join(c for c in text if c.isascii() and c.isdigit())


# =============================================================================
# WORKFLOW: VanillaWorkflow (PROVIDED AS REFERENCE)
# =============================================================================

@mcp_agent_app.workflow
class VanillaWorkflow(Workflow[dict]):
    """
    Basic workflow for screen capture and status announcement.
    USE THIS AS A REFERENCE for implementing your workflows.
    """
    
    path_to_screenshot = ''    

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def read_path_to_screenshot(self) -> dict:
        """Load screenshot path from environment configuration."""
        env_vars = load_env_file()
        self.path_to_screenshot = env_vars.get("PATH_TO_SCREENSHOT")        
        logger.info(f"screenshot path: {self.path_to_screenshot}")
        return env_vars

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def take_screenshot(self) -> str:
        """Capture and save a screenshot of the current screen."""
        current_screenshot_path = ''
        if self.path_to_screenshot:
            current_screenshot_path = os.path.join(
                self.path_to_screenshot, 
                f"screenshot_{id(self)}.png"
            )
            screenshot = ImageGrab.grab()
            screenshot.save(current_screenshot_path, 'PNG')
            logger.info(f"A new screenshot is captured at {current_screenshot_path}")                        
        else:
            logger.error("The .env is not loaded. Unable to save the screenshot")

        return current_screenshot_path

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def read_a_command_loudly(self, command: str) -> dict:
        """Announce a command or status message via text-to-speech."""
        speak_text(command)
        return {'command': command, 'status': 'completed'}
        
        
    @mcp_agent_app.workflow_run
    async def run(self, target: str) -> WorkflowResult[dict]:
        """Execute the complete vanilla workflow."""
        step1 = await self.read_path_to_screenshot()
        step2 = await self.take_screenshot()

        time.sleep(0.2)

        status = 'completed' if step2 else 'failure'
        step3 = await self.read_a_command_loudly(status)
        
        return WorkflowResult(value={
            'workflow': 'VanillaWorkflow',
            'target': 'None',
            'steps': [step1, step2, step3],
            'status': status,
            'message': f"The workflow is completed {'successfully' if step2 else 'unsuccessfully'}"
        })


@mcp_server.tool()
async def vanilla_workflow_tool(target=None) -> str:
    """MCP tool endpoint for executing the VanillaWorkflow."""
    try:
        workflow = VanillaWorkflow()
        result = await workflow.run(target)
        return json.dumps(result.value, indent=2)
    
    except Exception as e:
        logger.error(f"vanilla_workflow_tool Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


# =============================================================================
# WORKFLOW: ClickWorkFlow (TODO: IMPLEMENT THIS)
# =============================================================================

@mcp_agent_app.workflow
class ClickWorkFlow(Workflow[dict]):
    """
    Workflow for clicking on a specific UI element by name.
    """
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def read_metadata(self, path_to_json_file: str) -> dict:
        """Load OCR metadata from a JSON file."""
        # TODO: Implement this method
        logger.info(f"[ClickWorkFlow] Reading OCR metadata from: {path_to_json_file}")
        if not os.path.exists(path_to_json_file):
            raise FileNotFoundError(f"OCR metadata JSON not found: {path_to_json_file}")
        with open(path_to_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def find_target(self, ocr_data: dict, target: str) -> dict:
        """Find a target UI element in the OCR data by text matching."""
        # TODO: Implement this method
        logger.info(f"[ClickWorkFlow] Finding target '{target}' in OCR metadata")
        mappings = ocr_data.get("mappings", []) or []

        target_lower = target.lower()
        best = None

        for m in mappings:
            text = str(m.get("text", ""))
            if not text:
                continue
            if target_lower in text.lower():
                # Prefer the longest matching text as "best"
                if best is None or len(text) > len(best.get("text", "")):
                    best = m

        if best is None:
            logger.warning(f"[ClickWorkFlow] No match found for target '{target}'")
            return {
                "status": "not_found",
                "target": target,
                "mapping": None,
            }

        logger.info(
            f"[ClickWorkFlow] Found target '{target}' -> "
            f"number={best.get('number')} text='{best.get('text')}' "
            f"center=({best.get('center_x')},{best.get('center_y')})"
        )
        return {
            "status": "found",
            "target": target,
            "mapping": best,
        }
        
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def click_element(self, find_data: dict) -> dict:
        """Click on a UI element using mouse automation."""
        # TODO: Implement this method
        status = find_data.get("status")
        if status != "found":
            logger.warning("[ClickWorkFlow] click_element called with no found target")
            return {
                "status": "not_clicked",
                "reason": "target_not_found",
            }

        mapping = find_data.get("mapping") or {}
        cx = float(mapping.get("center_x", 0.0))
        cy = float(mapping.get("center_y", 0.0))

        # Map from screenshot coordinates to screen coordinates
        screen_x = int(cx * scale_x)
        screen_y = int(cy * scale_y)

        logger.info(
            f"[ClickWorkFlow] Clicking on target '{find_data.get('target')}' "
            f"at screen coords=({screen_x},{screen_y})"
        )

        try:
            pyautogui.moveTo(screen_x, screen_y, duration=0.2)
            pyautogui.click()
            return {
                "status": "clicked",
                "target": find_data.get("target"),
                "screen_x": screen_x,
                "screen_y": screen_y,
            }
        except Exception as e:
            logger.error(f"[ClickWorkFlow] Mouse click failed: {e}")
            return {
                "status": "error",
                "reason": str(e),
            }
        
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def read_a_command_loudly(self, command: str) -> dict:
        """Announce a command or status via text-to-speech."""
        # TODO: Implement this method
        logger.info(f"[ClickWorkFlow] TTS: {command}")
        speak_text(command)
        return {
            "status": "spoken",
            "text": command,
        }
        
        
    @mcp_agent_app.workflow_run
    async def run(self, target: str, path_to_json: str) -> WorkflowResult[dict]:
        """Execute the complete click workflow."""
        # TODO: Implement this method
        logger.info(
            f"[ClickWorkFlow] Starting workflow for target='{target}', "
            f"metadata='{path_to_json}'"
        )

        # Step 0: announce intent
        step0 = await self.read_a_command_loudly(f"Looking for {target}")

        # Step 1: read OCR metadata
        ocr_data = await self.read_metadata(path_to_json)

        # Step 2: find the target in metadata
        find_result = await self.find_target(ocr_data, target)

        if find_result.get("status") != "found":
            # Announce failure and stop
            await self.read_a_command_loudly(f"I cannot find {target}")
            return WorkflowResult(
                value={
                    "workflow": "ClickWorkFlow",
                    "target": target,
                    "status": "not_found",
                    "steps": [
                        {"step": "announce_start", "result": step0},
                        {"step": "find_target", "result": find_result},
                    ],
                    "message": f"Could not find target '{target}' in OCR metadata",
                }
            )

        # Step 3: click
        click_result = await self.click_element(find_result)

        # Step 4: announce success/failure
        if click_result.get("status") == "clicked":
            final_tts = f"Clicked on {target}"
        else:
            final_tts = f"Failed to click on {target}"
        step4 = await self.read_a_command_loudly(final_tts)

        status = click_result.get("status", "unknown")
        return WorkflowResult(
            value={
                "workflow": "ClickWorkFlow",
                "target": target,
                "status": status,
                "steps": [
                    {"step": "announce_start", "result": step0},
                    {"step": "find_target", "result": find_result},
                    {"step": "click_element", "result": click_result},
                    {"step": "announce_end", "result": step4},
                ],
                "message": f"Click workflow finished with status={status}",
            }
        )



@mcp_server.tool()
async def click_workflow_tool(target: str) -> str:
    """MCP tool endpoint for clicking on a UI element by name."""
    try:
        # TODO: Implement this tool
        env_vars = load_env_file()

        # Prefer a dedicated OCR_JSON path if provided; otherwise, fall back to screenshot dir.
        base_dir = (
            env_vars.get("PATH_TO_OCR_JSON")
            or env_vars.get("PATH_TO_SCREENSHOT")
            or "."
        )
        json_file = env_vars.get("OCR_JSON_FILE", "ocr_mappings.json")

        if not os.path.isabs(json_file):
            json_path = os.path.abspath(os.path.join(base_dir, json_file))
        else:
            json_path = json_file

        logger.info(
            f"[click_workflow_tool] Target='{target}', using metadata JSON='{json_path}'"
        )

        workflow = ClickWorkFlow()
        result = await workflow.run(target, json_path)
        return json.dumps(result.value, indent=2)
    
    except Exception as e:
        logger.error(f"click_workflow_tool Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


# =============================================================================
# WORKFLOW: CaptureScreenWithNumbers (TODO: IMPLEMENT THIS)
# =============================================================================

@mcp_agent_app.workflow
class CaptureScreenWithNumbers(Workflow[dict]):
    """
    Screen capture workflow with OCR and number mapping.
    """

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def prepare_screen(self, command: str) -> dict:
        """Announce a preparation command via TTS."""
        # TODO: Implement this method
        logger.info(f"[CaptureScreenWithNumbers] TTS: {command}")
        speak_text(command)
        return {
            "status": "spoken",
            "command": command,
        }

    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def analyze_screen(self) -> dict:
        """Capture screenshot and perform OCR with number-to-text mapping."""
        # TODO: Implement this method
        logger.info("[CaptureScreenWithNumbers] Starting screenshot + OCR analysis")
        env_vars = load_env_file()

        screenshot_dir = env_vars.get("PATH_TO_SCREENSHOT") or "."
        json_dir = env_vars.get("PATH_TO_OCR_JSON") or screenshot_dir

        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # Capture screenshot
        screenshot_path = os.path.join(screenshot_dir, "ocr_screenshot.png")
        img = ImageGrab.grab()
        img.save(screenshot_path)
        img_width, img_height = img.size
        logger.info(
            f"[CaptureScreenWithNumbers] Screenshot saved: {screenshot_path} "
            f"size={img_width}x{img_height}"
        )

        # Run OCR
        ocr = get_ocr()
        ocr_result = ocr.ocr(screenshot_path) # used to have cls=False
        # Diag -- newly added
        import numpy as np
        from pprint import pprint
        def numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [numpy_to_list(x) for x in obj]
            if isinstance(obj, dict):
                return {k: numpy_to_list(v) for k, v in obj.items()}
            return obj
        
        print("\n===== RAW OCR RESULT =====")
        # print(json.dumps(numpy_to_list(ocr_result), indent=2))
        pprint(ocr_result)
        print("==========================\n")

        mappings = []
        number_entries = []
        text_entries = []

        # Flatten OCR result and separate numeric / text entries
        for page in ocr_result or []:
            for det in page or []:
                if len(det) < 2:
                    continue
                # box, (txt, conf) = det
                try:
                    box = det[0]
                    txt, conf = det[1]
                except Exception:
                    continue
                
                if not txt:
                    continue

                # Compute center of bounding box
                try:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                except Exception:
                    continue

                digits = extract_ascii_digits(txt)
                stripped = txt
                for d in digits:
                    stripped = stripped.replace(d, "")
                stripped = stripped.strip()

                base_entry = {
                    "raw_text": txt,
                    "digits": digits,
                    "text": stripped if stripped else txt,
                    "center_x": cx,
                    "center_y": cy,
                }

                # Case 1: digits and label in same text (e.g., "1Settings")
                if digits and stripped:
                    mappings.append(
                        {
                            "number": digits,
                            "text": stripped,
                            "center_x": cx,
                            "center_y": cy,
                        }
                    )
                # Case 2: pure number (e.g., "1") – need to associate with nearest label
                elif digits and not stripped:
                    number_entries.append(base_entry)
                # Case 3: pure label, no leading digits
                else:
                    text_entries.append(base_entry)

        # Associate pure number entries with nearest text entry
        for n in number_entries:
            nx, ny = n["center_x"], n["center_y"]
            best_text = None
            best_dist = None
            for t in text_entries:
                tx, ty = t["center_x"], t["center_y"]
                dist = (nx - tx) ** 2 + (ny - ty) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_text = t
            if best_text is not None:
                mappings.append(
                    {
                        "number": n["digits"],
                        "text": best_text["text"],
                        "center_x": nx,
                        "center_y": ny,
                    }
                )

        # Deduplicate by number (keep first occurrence)
        unique_by_number = {}
        for m in mappings:
            num = m.get("number")
            if not num:
                continue
            if num not in unique_by_number:
                unique_by_number[num] = m

        final_mappings = list(unique_by_number.values())
        logger.info(
            f"[CaptureScreenWithNumbers] Total mappings created: {len(final_mappings)}"
        )

        metadata = {
            "image_path": screenshot_path,
            "image_width": img_width,
            "image_height": img_height,
            "mappings": final_mappings,
        }

        json_file = env_vars.get("OCR_JSON_FILE", "ocr_mappings.json")
        if not os.path.isabs(json_file):
            json_path = os.path.abspath(os.path.join(json_dir, json_file))
        else:
            json_path = json_file

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"[CaptureScreenWithNumbers] OCR metadata saved to {json_path}"
        )

        return {
            "status": "completed",
            "json_path": json_path,
            "mapping_count": len(final_mappings),
        }


    
    @mcp_agent_app.workflow_run
    async def run(self) -> WorkflowResult[dict]:
        """Execute the complete screen capture and OCR workflow."""
        # TODO: Implement this method
        logger.info("[CaptureScreenWithNumbers] Workflow run() started")

        # Step 1: tell OS VUI to start listening (user already has Voice Control / Voice Access on)
        step1 = await self.prepare_screen("start listening")
        time.sleep(4.0)

        # Step 2: tell OS to show numbered labels on the screen
        step2 = await self.prepare_screen("show numbers")
        time.sleep(4.0)

        # Step 3: capture + OCR + mapping
        step3 = await self.analyze_screen()

        status = step3.get("status", "completed")
        return WorkflowResult(
            value={
                "workflow": "CaptureScreenWithNumbers",
                "status": status,
                "json_path": step3.get("json_path"),
                "mapping_count": step3.get("mapping_count", 0),
                "steps": [
                    {"step": "start_listening", "result": step1},
                    {"step": "show_numbers", "result": step2},
                    {"step": "analyze_screen", "result": step3},
                ],
                "message": "Screen capture and OCR workflow completed",
            }
        )



@mcp_server.tool()
async def capture_screen_with_numbers_tool():
    """MCP tool endpoint for capturing screen with numbered labels and OCR."""
    try:        
        # TODO: Implement this tool
        workflow = CaptureScreenWithNumbers()
        result = await workflow.run()
        return json.dumps(result.value, indent=2)
    
    except Exception as e:
        logger.error(f"capture_screen_with_numbers_tool Error: {e}")
        return json.dumps({
            'status': 'error',
            'error': str(e)
        })


# =============================================================================
# DIRECT VOICE COMMAND TOOL (PROVIDED)
# =============================================================================

@mcp_server.tool()
async def echo_tool(command: str) -> str:
    """Direct voice command echo and TTS tool."""
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Hybrid Server: mcp-agent + FastMCP Streamable HTTP")
    logger.info("=" * 60)
    logger.info("URL: http://127.0.0.1:3000/mcp")
    logger.info("=" * 60)

    mcp_server.run(transport="streamable-http")