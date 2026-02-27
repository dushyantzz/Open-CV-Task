import cv2
import argparse
from detector import Detector
from tracker import SackCounter

def main(video_path, output_path=None):
    # Initialize components
    detector = Detector(model_path='yolo11n.pt')
    counter = SackCounter(line_y_ratio=0.6) # Adjust ratio based on video

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video... Press 'q' to stop.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect and Track
        results = detector.detect_and_track(frame)

        # 2. Process counts
        sacks = counter.process_tracks(results, frame)

        # 3. Draw UI
        frame = counter.draw_ui(frame, results)

        # 4. Save/Show
        if out:
            out.write(frame)
        
        cv2.imshow("Sack Counting & Laborer Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print("-" * 30)
    print("Final Report:")
    print(f"Total Sacks Counted: {sacks}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="c:/Users/Shubham/Desktop/Reference_Solution.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output_processed.mp4", help="Path to output video")
    args = parser.parse_args()
    
    main(args.video, args.output)
