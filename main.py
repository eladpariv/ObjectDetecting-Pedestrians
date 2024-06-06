from ultralytics import YOLO, NAS
import cv2
from sort.sort import *


# For Detecting Single Image

model = YOLO('best-2.pt')

threshold = 0.6

image_path = 'test_image.jpg'
frame = cv2.imread(image_path)

mot_tracker = Sort()

track_ids = []
detections_ = []

colors = {

    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 255, 0),
    3: (0, 255, 255),
    4: (255, 0, 255)
        }

if frame is None:
    print(f"Error loading image {image_path}")
else:
    H, W, _ = frame.shape

    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            detections_.append([x1, y1, x2, y2, score])
            track_ids = mot_tracker.update(np.asarray(detections_))
    for j in range(len(track_ids)):
        x1, y1, x2, y2, index = track_ids[j]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[index % 5], 4)
        cv2.putText(frame, str(int(index)), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (colors[index % 5]), 3, cv2.LINE_AA)

    output_image_path = 'outputImage.jpg'
    cv2.imwrite(output_image_path, frame)




# For Detecting Videos

video_path_out = 'video_out.mp4'
cap = cv2.VideoCapture('test_video.mp4')
ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc('H', '2', '6', '4'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model = YOLO("yolov8n.pt")

threshold = 0.6
frame_nmr = -1

mot_tracker = Sort()

track_ids = []
detections_ = []

colors = {
    0: (255, 0, 0),
    1: (100, 0, 0),
    2: (0, 255, 0),
    3: (0, 100, 0),
    4: (0, 0, 255),
    5: (0, 0, 100),
    6: (0, 0, 0),
    7: (255, 255, 255),
    8: (255, 255, 0),
    9: (0, 255, 255),
    10: (255, 0, 255)
        }


while ret:
    frame_nmr += 1
    results = model(frame)[0]
    detections_ = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and class_id == 0:
            detections_.append([x1, y1, x2, y2, score])
    if detections_:
        track_ids = mot_tracker.update(np.asarray(detections_))

    for j in range(len(track_ids)):
        x1, y1, x2, y2, index = track_ids[j]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(int(index)), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
    
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
