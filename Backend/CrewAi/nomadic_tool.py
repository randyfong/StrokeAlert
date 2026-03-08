from nomadicml import NomadicML, AnalysisType
from nomadicml.video import CustomCategory
from crewai.tools import tool
import json
import os

class NomadicMLTool:
    @tool("analyze_stroke_video")
    def analyze_stroke_video(video_path: str) -> str:
        """
        Analyzes a video file using Nomadic ML to detect facial drooping, a sign of stroke.
        Returns a JSON list of detected events with label, timing, severity, confidence, and thumbnail URL.
        """
        NOMADICML_API_KEY = "sk_YxwW1JGpbum4u3IE0kf56FYs07cfGh92E5DkW6gdm8lunu2D"
        client = NomadicML(api_key=NOMADICML_API_KEY)
        
        # Upload and capture ID
        try:
            upload_result = client.upload(video_path)
            video_id = upload_result["video_id"]

            # Perform Targeted Analysis
            analysis = client.analyze(
                video_id,
                analysis_type=AnalysisType.ASK,
                custom_event="To find facial drooping related to stroke events. Only find moments of facial drooping",
                custom_category=CustomCategory.ROBOTICS,
                confidence="high"
            )

            # Format the specific output requested
            formatted_events = []
            for event in analysis.get('events', []):
                formatted_events.append({
                    "label": event.get("label"),
                    "event_timing": f"{event.get('t_start')} - {event.get('t_end')}",
                    "severity": event.get("severity"),
                    "confidence": event.get("confidence"),
                    "annotated_thumbnail_url": event.get("annotated_thumbnail_url")
                })
            
            return json.dumps(formatted_events, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
