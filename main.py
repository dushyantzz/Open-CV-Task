"""
Standalone CLI for bag counting on a single video.
Usage:
    python main.py --video "Problem Statement Scenario1.mp4"
    python main.py --video "Problem Statement Scenario2.mp4" --output out2.mp4
"""
import cv2
import argparse
from detector import Detector
from tracker import SackCounter


def main(video_path, output_path=None):
    detector = Detector(model_path='yolo11n.pt', conf=0.30, iou=0.50, img_size=640)
    counter = SackCounter(line_x_ratio=0.50)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {video_path}")
    print("Press 'q' to stop early.\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_and_track(frame)
        bags_in, bags_out = counter.process_tracks(results, frame)
        frame = counter.draw_overlay(frame, results)

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"Bags In: {bags_in}   Bags Out: {bags_out}",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if out:
            out.write(frame)

        cv2.imshow("Bag Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print("-" * 40)
    print("Final Report:")
    print(f"  Bags In  (Loading)   : {bags_in}")
    print(f"  Bags Out (Unloading) : {bags_out}")
    print(f"  Total Counted        : {bags_in + bags_out}")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str,
                        default="Problem Statement Scenario1.mp4",
                        help="Path to input video")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output video (optional)")
    args = parser.parse_args()
    main(args.video, args.output)
