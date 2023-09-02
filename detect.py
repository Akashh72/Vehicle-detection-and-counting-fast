import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from tracker import *
import time
import argparse
import os
import glob
import time


# Open video capture
parser = argparse.ArgumentParser(description="Vehicle Detection Script")

# Add an argument for the video file path
parser.add_argument("--path", type=str, required=True, help="Path to the video file")

# Parse the command-line arguments
args = parser.parse_args()

# Use the provided video file path
directory_path  = args.path
directory_path = str(directory_path)

file_list = glob.glob(os.path.join(directory_path, '*'))
num_files = len(file_list)
print(f"Number of files in the directory: {num_files}")
for file_path in file_list:
    # Your code to process each file goes here
    print(f"\n\n------------------------Processing file {file_path}------------------------\n\n ")

    start_time = time.time()

    my_file = open("classes.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    area_class_dictionary = dict()
    for _class in class_list:
        area_class_dictionary[_class] = set()

    # Initialize the tracker
    tracker = Tracker()
    area_coords = [(0, 210), (0, 225), (640, 225), (640, 210)]

    object_id_counter = 0
    object_id_mapping = {}  # Maps object index to its ID


    # Load pretrained model
    weights = 'finalmodel.pt'  # Replace with your model weights path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = attempt_load(weights).to(device)

    # Open video capture
    video_path = str(file_path)  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to specified imgsz
        frame = cv2.resize(frame, (640, 640))

        # Convert BGR to RGB
        frame_rgb = frame[..., ::-1]
        # frame_rgb = frame

        # Perform inference on the frame
        with torch.no_grad():
            frame_rgb_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(frame_rgb_tensor)

        # Apply NMS
        pred = non_max_suppression(pred,conf_thres=0.3, iou_thres=0.45, classes=None, agnostic=True, labels=(),max_det = 1000)

        # Render detection bounding boxes on the frame
        if len(pred) > 0:
            pred = pred[0]  # Assuming you're only interested in the first detection result
            objects_rect = []
            for det in pred:
                det = det.cpu().numpy()
                x1, y1, x2, y2, conf, cls = det
                objects_rect.append((x1, y1, x2, y2, cls))

            tracker_objects = tracker.update(objects_rect)

            for rect in tracker_objects:
                x1, y1, x2, y2, cls, obj_id = rect
                center_x = int((x1 + x2) // 2)
                center_y = int((y1 + y2) // 2)
                if cv2.pointPolygonTest(np.array(area_coords,np.int32), (center_x, center_y), False) >= 0:
                    area_class_dictionary[class_list[int(cls)]].add(obj_id)

    cap.release()
    for _class in class_list:
        count = len(area_class_dictionary.get(_class, []))
        print(f"{_class} = {count}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds\n")
