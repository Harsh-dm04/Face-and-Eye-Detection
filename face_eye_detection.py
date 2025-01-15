# import cv2

# # Load the Haar Cascade file for object detection (e.g., face detection)
# cascade_path = "haarcascade_frontalface_default.xml"  # Replace with your Haar cascade file
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# # Load the image
# img = cv2.imread("demo_img.jpeg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect objects (e.g., faces)
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Draw rectangles around detected objects
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# # Display the result
# cv2.imshow("Detected Objects", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

cap = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml") 

# # if not cap.isOpened():
# #     print("Error opening video stream or file")
# #     exit()

while True:
    ret, frame = cap.read()

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 3,minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0, 0), 5)
        roi_gray=gray[ y:y + h, x:x + w]
        roi_color=frame[ y:y + h, x:x + w]
        eyes=eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255, 0),5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()