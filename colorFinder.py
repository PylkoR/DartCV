import numpy as np
import cv2 as cv
from cvzone.ColorModule import ColorFinder

frameCounter = 0
colorFinder = ColorFinder(True)

hsvNet_wide = {'hmin': 0, 'smin': 24, 'vmin': 114, 'hmax': 73, 'smax': 144, 'vmax': 177}
hsvGreen_wide = {'hmin': 33, 'smin': 70, 'vmin': 60, 'hmax': 130, 'smax': 255, 'vmax': 255}
hsvRed_wide = {'hmin': 0, 'smin': 93, 'vmin': 126, 'hmax': 14, 'smax': 255, 'vmax': 255}

hsvNet = {'hmin': 0, 'smin': 0, 'vmin': 141, 'hmax': 55, 'smax': 130, 'vmax': 255}
hsvGreen = {'hmin': 39, 'smin': 108, 'vmin': 64, 'hmax': 96, 'smax': 255, 'vmax': 255}
hsvRed = {'hmin': 0, 'smin': 133, 'vmin': 108, 'hmax': 13, 'smax': 255, 'vmax': 255}

# score fields recognition
while True:

    img = cv.imread("pics/frame1.png")
    dartboard, mask = colorFinder.update(img)
    cv.imshow("Dartboard", dartboard)
    # waitKey returns the pressed key (or -1). Break the loop on 'q'
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()