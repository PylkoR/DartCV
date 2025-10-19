import numpy as np
import cv2 as cv
from cvzone.ColorModule import ColorFinder

# Load the video
cap = cv.VideoCapture("pics/throws_cut.mp4")
frameCounter = 0
colorFinder = ColorFinder(True)

# Get video properties
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps, "size:", width, "x", height)
hsvNet = {'hmin': 0, 'smin': 24, 'vmin': 114, 'hmax': 73, 'smax': 144, 'vmax': 177}
hsvGreen = {'hmin': 33, 'smin': 70, 'vmin': 60, 'hmax': 130, 'smax': 255, 'vmax': 255}
hsvRed = {'hmin': 0, 'smin': 93, 'vmin': 126, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Get first frame
ret, frame = cap.read()
cropped = frame[300:1740, :]

# cv.namedWindow("Frame 1", cv.WINDOW_NORMAL)
# cv.imshow("Frame 1", frame)
# cv.resizeWindow("Frame 1", 536, 960)

# cv.namedWindow("Frame 1 - cropped", cv.WINDOW_NORMAL)
# cv.imshow("Frame 1 - cropped", cropped)
# cv.resizeWindow("Frame 1 - cropped", 536, 700)
# cv.imwrite("pics/frame1.png", cropped)
# cv.waitKey(0)

# score fields recognition
while True:
    frameCounter += 1
    if frameCounter == cap.get(cv.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    frame = frame[300:1740, :]
    cv.namedWindow("Video", cv.WINDOW_NORMAL)
    cv.imshow("Video", frame)
    cv.resizeWindow("Video", 536, 700)

    img = cv.imread("pics/frame1.png")
    dartboard, mask = colorFinder.update(img)
    cv.imshow("Dartboard", dartboard)
    cv.waitKey(1)


# end
cap.release()