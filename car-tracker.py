'''
Sources
    car detection: https://www.geeksforgeeks.org/opencv-python-program-vehicle-detection-video-frame/
    camera input: https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
'''

# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV

# from gtts import gTTS
# import os
import cv2
import pyttsx3

ttsEngine = pyttsx3.init()  # object creation

language = 'en'

videoFile = 'video.avi'  # change to 0 for camera input

# capture frames from a video
cap = cv2.VideoCapture(videoFile)

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')

frameCtr = 0

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(cars):
        frameCtr += 1
    else:
        frameCtr = 0
    if frameCtr >= 8:
        cv2.putText(frames, 'Found a Car', (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),2)
        ttsEngine.say("Do not cross")
        ttsEngine.runAndWait()
        # ttsEngine.stop()
    else:
        cv2.putText(frames, 'Safe to Cross', (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),2)
        ttsEngine.say("Safe to cross")
        ttsEngine.runAndWait()
        # ttsEngine.stop()

    # Display frames in a window
    cv2.imshow('video2', frames)
    # ttsEngine.stop()

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
