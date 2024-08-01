import cv2
import os

def draw_boxes(frame, detections):
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls, name = row
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def save_frame(frame, output_dir, frame_id):
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f'frame_{frame_id:05d}.jpg')
    cv2.imwrite(frame_path, frame)
