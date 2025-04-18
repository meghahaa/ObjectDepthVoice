import cv2

import warnings
warnings.filterwarnings("ignore")

from codes.model_loader import load_models
from codes.detection import detect_objects, process_results
from codes.depth import estimate_depth
from codes.speech import speak_in_bgd


def main():
    '''
    Either webcam - device camera 
    or
    ipcam - stream from another device through IP Webcam app
    '''
    source = input("Select video source (webcam/ipcam): ").strip().lower()
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "ipcam":
        url = input("Enter IP camera stream URL: ").strip()
        cap = cv2.VideoCapture(url)
    else:
        print("Invalid option.")
        return

    # Load models
    detection_model, depth_model = load_models()

    # To Track previously detected objects
    detected_objects = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            depth_map = estimate_depth(frame, depth_model)
            detections = detect_objects(frame, detection_model)
            annotated_frame, feedback = process_results(frame, detections, depth_map, detected_objects)

            for sentence in feedback:
                speak_in_bgd(sentence)

            cv2.imshow("Object Detection", annotated_frame)
            cv2.imshow("Depth Map", depth_map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
