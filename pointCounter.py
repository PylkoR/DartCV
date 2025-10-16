import numpy as np
import cv2 as cv

# Load the video
cap = cv.VideoCapture("pics/throws_cut.mp4")

# Get video properties
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps, "size:", width, "x", height)

# Get first frame
ret, frame = cap.read()
cropped = frame[300:1740, :]

cv.namedWindow("Frame 1", cv.WINDOW_NORMAL)
cv.imshow("Frame 1", frame)
cv.resizeWindow("Frame 1", 536, 960)

cv.namedWindow("Frame 1 - cropped", cv.WINDOW_NORMAL)
cv.imshow("Frame 1 - cropped", cropped)
cv.resizeWindow("Frame 1 - cropped", 536, 700)
cv.imwrite("pics/frame1.png", cropped)
cv.waitKey(0)

# score fields recognition



# end
cap.release()