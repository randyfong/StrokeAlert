from crewai import Agent, Task, Crew, Process
from nomadic_tool import NomadicMLTool

def create_stroke_detection_crew(video_path: str):
    # Define the Agent
    video_analyst = Agent(
        role='Stroke Detection Video Analyst',
        goal='Automatically process video files using the Nomadic ML library to identify medical emergencies, specifically focusing on "facial drooping" as a diagnostic indicator of a stroke.',
        backstory="""You are a highly specialized medical AI assistant designed for rapid triage. 
        You excel at integrating specialized computer vision SDKs to analyze video feeds for life-critical symptoms. 
        Your primary tool is the nomadicml Python library. 
        You prioritize high-confidence detections to ensure medical accuracy while maintaining a strict JSON output format for downstream emergency services.""",
        tools=[NomadicMLTool.analyze_stroke_video],
        verbose=True,
        allow_delegation=False
    )

    # Define the Task
    analysis_task = Task(
        description=f"Process the video file at '{video_path}' using the nomadicml tool to find facial drooping related to stroke events. Ensure the output is a clean JSON list of events with labels, timing, severity, confidence, and thumbnail URLs.",
        expected_output="A clean JSON list of events representing detected facial drooping with specific metadata.",
        agent=video_analyst
    )

    # Define the Crew
    stroke_crew = Crew(
        agents=[video_analyst],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True,
        trace=False  # Disabled to prevent blocking prompts
    )

    return stroke_crew
