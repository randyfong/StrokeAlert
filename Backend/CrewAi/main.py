from agent import create_stroke_detection_crew
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    print(f"Starting Stroke Detection Analysis for: {video_path}")
    
    crew = create_stroke_detection_crew(video_path)
    result = crew.kickoff()
    
    print("\n\n########################")
    print("## Final Result JSON ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    main()
