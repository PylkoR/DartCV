import numpy as np
import cv2 as cv
from cvzone.ColorModule import ColorFinder

# Load the video
cap = cv.VideoCapture("pics/dart_normal.mp4")
frameCounter = 0

# Get video properties
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps, "size:", width, "x", height)

# Get first frame
ret, frame = cap.read()
cropped = frame[200:1280, :]

cv.namedWindow("Frame 1", cv.WINDOW_NORMAL)
cv.imshow("Frame 1", frame)
cv.resizeWindow("Frame 1", 540, 960)

cv.namedWindow("Frame 1 - cropped", cv.WINDOW_NORMAL)
cv.imshow("Frame 1 - cropped", cropped)
cv.imwrite("pics/frame1.png", cropped)
cv.resizeWindow("Frame 1 - cropped", 540, 540)
cv.waitKey(0)

# score fields recognition
while True:
    frameCounter += 1
    if frameCounter == cap.get(cv.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'): # Naciśnięcie 'q' zamyka okno
        break
    
    ret, frame = cap.read()
    frame = frame[200:1280, :]
    cv.namedWindow("Video", cv.WINDOW_NORMAL)
    cv.imshow("Video", frame)
    cv.resizeWindow("Video", 540, 540)
    cv.waitKey(1)

# end
cap.release()