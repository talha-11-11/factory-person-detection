# factory-person-detection
# Factory Person Detection and Counting

## Overview
This project detects and counts persons entering a factory before and after attendance using computer vision. The project utilizes YOLOv5 for person detection in video frames.

## Instructions
1. Clone the repository.
2. Install the required libraries: `pip install -r requirements.txt`.
3. Run the person detection script: `python scripts/person_detection.py`.
4. Count the people in frames: `python scripts/count_people.py`.

## CI/CD Pipeline
This project uses GitHub Actions for Continuous Integration and Continuous Deployment. The workflow is triggered on pushes to the `main` branch and is scheduled to run daily.

## Requirements
- opencv-python
- torch
- pandas
- unittest
