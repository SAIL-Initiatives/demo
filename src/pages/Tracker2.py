import cv2
import numpy as np

def run_local_tracker():
    # 1. Initialize the detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 2. Access local webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Detect Markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        # 4. Draw Results
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 5. Show the live window
        cv2.imshow('Local ArUco Tracker', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_local_tracker()
