# Sack Counting & Laborer Analysis AI

This project provides an automated solution for counting sacks and analyzing laborer activity in loading/unloading videos. Built for an internship assignment, it utilizes state-of-the-art computer vision models (YOLOv11) and custom tracking logic.

## Features
*   **Automatic Sack Counting**: Detects when a laborer carries a sack across a defined "Counting Line".
*   **Laborer Tracking**: Assigns unique IDs to distinct laborers.
*   **Distinct Laborer Count**: Estimates the total number of unique laborers involved in the process.
*   **Real-time Visualization**: Displays bounding boxes, tracking IDs, and a live counter on the video.
*   **Generated Report**: Outputs final counts to the console after processing.

## Tech Stack
*   **Python 3.12+**
*   **OpenCV**: For video processing and UI.
*   **Ultralytics YOLOv11**: For high-accuracy object detection and tracking.
*   **Numpy**: For numerical computations.

## Installation
1.  Clone the repository:
    ```bash
    git clone <repo-url>
    cd Sack-Counting-AI-Analysis
    ```
2.  Install dependencies:
    ```bash
    pip install ultralytics opencv-python numpy torch
    ```

## Usage
Run the main script providing the path to your video:
```bash
python main.py --video path/to/your/video.mp4 --output results.mp4
```

## Results
The system generates a processed video with overlays and provides a summary report:
*   Total Sacks Counted
*   Total Distinct Laborers

### Example Output:
```text
Final Report:
Total Sacks Counted: 6
Total Distinct Laborers: 53
```




