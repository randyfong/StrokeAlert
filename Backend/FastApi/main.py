import os
import uuid
import shutil
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any

# Import CrewAI integration
import sys
# Add parent directory to path to allow importing CrewAI modules if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CrewAi'))
from agent import create_stroke_detection_crew

app = FastAPI(title="Stroke Alert API")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "..", "Frontend")

# Mount frontend directory to serve index.html
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to store task results
tasks_results: Dict[str, Dict[str, Any]] = {}

TEMP_VIDEO_DIR = "/tmp/stroke_alert_videos"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    Accepts a video file upload, processes it synchronously using CrewAI, and returns detections.
    """
    is_video = file.content_type.startswith("video/") or \
               file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    
    if not is_video:
        raise HTTPException(status_code=400, detail=f"Uploaded file {file.filename} is not a video.")

    task_id = str(uuid.uuid4())
    video_filename = f"{task_id}_{file.filename}"
    video_path = os.path.join(TEMP_VIDEO_DIR, video_filename)

    # Save the file to local temp directory
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video file: {str(e)}")

    try:
        # Create and run the crew synchronously
        crew = create_stroke_detection_crew(video_path)
        result = crew.kickoff()
        
        # Parse the result as a list of events
        # The agent.py task description asks for a JSON list of events.
        # We try to parse the raw result string.
        events = []
        try:
            # result might be a CrewOutput object or a string
            result_str = str(result)
            # Find the starting [ and ending ] to extract JSON part if needed
            start_idx = result_str.find('[')
            end_idx = result_str.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                events = json.loads(result_str[start_idx:end_idx])
            else:
                # If no brackets found, try parsing the whole thing if it's JSON
                events = json.loads(result_str)
        except Exception:
            # Fallback if parsing fails - the agent might have returned raw text
            # In a real scenario, you'd want more robust parsing
            events = []

        response_data = {
            "task_id": task_id,
            "status": "completed",
            "events": events,
            "raw_result": str(result)
        }
        
        # Store for retrieval via ID if needed
        tasks_results[task_id] = response_data
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Retrieves the processing status and results for a given task_id.
    """
    if task_id not in tasks_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_results[task_id]

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
