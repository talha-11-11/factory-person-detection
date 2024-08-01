import cv2
import torch
from scripts.utils import draw_boxes, save_frame

MODEL_PATH = 'models/yolov5s.pt'

def detect_persons(video_path, output_dir):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]
        persons = detections[detections['name'] == 'person']

        frame = draw_boxes(frame, persons)
        save_frame(frame, output_dir, frame_id)
        frame_id += 1

    cap.release()

if __name__ == "__main__":
    detect_persons('data/input_videos/video1.mp4', 'data/output/detected_frames/')
