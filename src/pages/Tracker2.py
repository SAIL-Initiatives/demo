import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab.output import eval_js
from base64 import b64decode, b64encode

import IPython
from IPython.display import display, Javascript, clear_output

import PIL.Image
import io
import html
import time



# 1. JavaScript to create the video element and the capture function
def start_webcam():
  js = Javascript('''
    async function startWebcam() {
      const div = document.createElement('div');
      const video = document.createElement('video');
      video.style.display = 'block';
      video.width = 640;
      video.height = 480;
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Attach capture function to window so Python can call it
      window.captureFrame = async function() {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.8);
      };
    }
    startWebcam();
  ''')
  display(js)

# 2. Helper to convert JS base64 string to OpenCV Mat
def js_to_image(js_reply):
  image_bytes = b64decode(js_reply.split(',')[1])
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  img = cv2.imdecode(jpg_as_np, flags=1)
  return img



# 3. Main Loop
def run_tracker():
  # Initialize ArUco Detector (Using 6x6 markers)
  aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  parameters = cv2.aruco.DetectorParameters()
  detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

  start_webcam()
  print("Webcam initialized. Waiting for stream...")
  time.sleep(2) # Give the camera a moment to warm up

  try:
    while True:
      # Call the JavaScript function we attached to 'window'
      js_reply = eval_js('window.captureFrame()')
      if not js_reply:
        break

      # Process frame
      frame = js_to_image(js_reply)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
      # Detect markers
      corners, ids, rejected = detector.detectMarkers(gray)

      # Draw results
      if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Draw a small circle in the center of the first marker found
        for corner in corners:
            center = np.mean(corner[0], axis=0).astype(int)
            cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)

      # Update display in Colab
      clear_output(wait=True)
      cv2_imshow(frame)
      
  except Exception as e:
    print(f"Stream closed or Error: {e}")

# Execute
import time
run_tracker()
