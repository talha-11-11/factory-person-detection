import os
import pandas as pd

def count_people_in_frames(frames_dir):
    frame_files = sorted(os.listdir(frames_dir))
    count_log = []

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        persons_detected = detect_persons_in_frame(frame)  # Custom function to detect persons
        count_log.append((frame_file, len(persons_detected)))

    count_log_df = pd.DataFrame(count_log, columns=['frame', 'count'])
    count_log_df.to_csv('data/output/count_log.csv', index=False)

def detect_persons_in_frame(frame):
    # Load model and detect persons in the frame
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt')
    results = model(frame)
    detections = results.pandas().xyxy[0]
    persons = detections[detections['name'] == 'person']
    return persons

if __name__ == "__main__":
    count_people_in_frames('data/output/detected_frames/')
