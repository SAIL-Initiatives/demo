import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Tracker demo", layout="wide")
st.title( "AR Marker Tracking with Streamlit")
st.write( 'A Python-based marker tracking app built using OpenCV’s ArUco module with a Streamlit web interface.')

# -------------------------
# Initialize Data Logging
# -------------------------
log_data = []

# -------------------------
# Video Input
# -------------------------
# Use 0 for local webcam, or replace with your phone stream URL
video_source = st.text_input("Enter Video Source (0 for webcam, or IP stream URL):", "0")

if video_source.isdigit():
    video_source = int(video_source)

cap = cv2.VideoCapture(video_source)

# -------------------------
# ArUco Setup
# -------------------------
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

# Streamlit video display
frame_window = st.image([])

# -------------------------
# Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture video. Check source.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i, corner in enumerate(corners):
            int_corners = np.int0(corner)
            cv2.polylines(frame, int_corners, True, (0,255,0), 2)

            # Compute center
            cX = int(corner[0][:,0].mean())
            cY = int(corner[0][:,1].mean())
            cv2.putText(frame, f'ID: {ids[i][0]}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # Log the marker position with timestamp
            log_data.append({
                "timestamp": datetime.now(),
                "marker_id": int(ids[i][0]),
                "x": cX,
                "y": cY
            })

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------
# Save Logged Data
# -------------------------
if st.button("Save Marker Log"):
    df = pd.DataFrame(log_data)
    df.to_csv("marker_positions.csv", index=False)
    st.success("Marker positions saved to marker_positions.csv")




mkdn= '''

# About this demo

| Component                                              | Purpose                                                  |
| ------------------------------------------------------ | -------------------------------------------------------- |
| **Python**                                             | Programming language                                     |
| **OpenCV (`opencv-python` + `opencv-contrib-python`)** | Computer vision library; includes ArUco marker detection |
| **ArUco markers**                                      | Fiducial markers for robust detection/tracking           |
| **Streamlit**                                          | Web app interface to display video and logs              |
| **NumPy / Pandas**                                     | Data processing and logging                              |
'''

st.markdown( mkdn)
