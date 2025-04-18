import numpy as np
import pandas
import cv2
from collections import deque

tracking_history = {}

def detect_objects(img, model):
    res = model(img)
    return res.pandas().xyxy[0]

def determine_position(xm, xM, ym, yM, w, h):
    xc = (xm + xM) / 2
    yc = (ym + yM) / 2
    center_xm = w / 3
    center_xM = 2 * w / 3
    center_ym = h / 3
    center_yM = 2 * h / 3

    if center_xm <= xc <= center_xM and center_ym <= yc <= center_yM:
        return "center"
    elif yc < h / 3:
        return "top left" if xc < w / 2 else "top right"
    elif yc < 2 * h / 3:
        return "mid left" if xc < w / 2 else "mid right"
    else:
        return "bottom left" if xc < w / 2 else "bottom right"

def smooth_detections(label, current_position, x_min, y_min, x_max, y_max):
    if label not in tracking_history:
        tracking_history[label] = deque(maxlen=5)

    tracking_history[label].append((current_position, x_min, y_min, x_max, y_max))
    avg_position = current_position
    avg_bbox = np.mean([item[1:] for item in tracking_history[label]], axis=0)

    return avg_position, int(avg_bbox[0]), int(avg_bbox[1]), int(avg_bbox[2]), int(avg_bbox[3])

def process_results(img, d, depth_map, detected_objects):
    h, w, _ = img.shape
    feedback = []
    threshold = 0.5

    for _, row in d.iterrows():
        confidence = row['confidence']
        if confidence < 0.5:
            continue

        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        lab = row['name']
        pos = determine_position(x_min, x_max, y_min, y_max, w, h)
        position, x_min, y_min, x_max, y_max = smooth_detections(lab, pos, x_min, y_min, x_max, y_max)
        object_key = (lab, pos)

        object_depth = depth_map[y_min:y_max, x_min:x_max]
        avg_depth = np.mean(object_depth)

        depth_cat = "near" if avg_depth < threshold else "far"

        if object_key not in detected_objects:
            feedback_sentence = f"{lab} is in {pos}. The object is {depth_cat}."
            feedback.append(feedback_sentence)
            detected_objects[object_key] = True

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"{lab}: {pos}, {depth_cat}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, feedback