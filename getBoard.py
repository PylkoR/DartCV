import cv2 as cv
import numpy as np

# --- Twoje zakresy HSV z colorFinder ---
hsvGreen_wide = {'hmin': 33, 'smin': 70, 'vmin': 60, 'hmax': 130, 'smax': 255, 'vmax': 255}
hsvRed_wide   = {'hmin': 0,  'smin': 93, 'vmin': 126, 'hmax': 14,  'smax': 255, 'vmax': 255}

hsvGreen = {'hmin': 39, 'smin': 108, 'vmin': 64, 'hmax': 96, 'smax': 255, 'vmax': 255}
hsvRed = {'hmin': 0, 'smin': 133, 'vmin': 108, 'hmax': 13, 'smax': 255, 'vmax': 255}

IMG = r"pics/frame1.png"  # <- ścieżka do Twojej klatki
# --- 1) Wczytanie obrazu i konwersja do HSV ---
img = cv.imread(IMG)

SCALE = 0.5  # zmniejsz jeśli duży obraz
if SCALE != 1.0:
    img = cv.resize(img, (0, 0), fx=SCALE, fy=SCALE)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# --- 2) Maski ---
# zielony
lower_green = np.array([hsvGreen['hmin'], hsvGreen['smin'], hsvGreen['vmin']])
upper_green = np.array([hsvGreen['hmax'], hsvGreen['smax'], hsvGreen['vmax']])
mask_green = cv.inRange(hsv, lower_green, upper_green)

# czerwony – tylko jeden zakres (u Ciebie 0–14)
lower_red = np.array([hsvRed['hmin'], hsvRed['smin'], hsvRed['vmin']])
upper_red = np.array([hsvRed['hmax'], hsvRed['smax'], hsvRed['vmax']])
mask_red = cv.inRange(hsv, lower_red, upper_red)

# --- 3) Czyszczenie masek (morfologia) ---
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask_red_clean = cv.morphologyEx(mask_red, cv.MORPH_OPEN, kernel, iterations=1)
mask_green_clean = cv.morphologyEx(mask_green, cv.MORPH_OPEN, kernel, iterations=1)

mask_combined = cv.bitwise_or(mask_red_clean, mask_green_clean)

# --- 5) Pokazanie efektów ---
cv.imshow("original", img)
#cv.imshow("mask_red", mask_red_clean)
#cv.imshow("mask_green", mask_green_clean)
cv.imshow("mask_combined", mask_combined)
# --- 6) Zapisz do pliku (łatwiej porównać pikselowo) ---
cv.imwrite("pics/out_mask_combined.png", mask_combined)

# --- wypełnij małe szczeliny między polami (zamykanie morfologiczne) ---
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask_closed = cv.morphologyEx(mask_combined, cv.MORPH_CLOSE, kernel, iterations=2)

cv.imshow("mask_closed", mask_closed)

# elipsa
cnts, _ = cv.findContours(mask_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
c = max(cnts, key=cv.contourArea)
ellipse = cv.fitEllipse(c)

# # --- narysuj elipsę na obrazie podglądowym ---
# preview = cv.cvtColor(mask_combined, cv.COLOR_GRAY2BGR)
# cv.ellipse(preview, ellipse, (0, 255, 0), 2)
# cv.imshow("ellipse", preview)
# cv.waitKey(0)

# ellipse: ((cx,cy),(W,H),angle)
(cx, cy), (W, H), ang = ellipse

# upewnij się, że W to oś DUŻA, H to oś MAŁA
if H > W:
    W, H = H, W
    ang += 90.0

A, B = W/2.0, H/2.0                      # półosie
rad = np.deg2rad(ang)
v = np.array([np.cos(rad), np.sin(rad)]) # kierunek osi dużej
u = np.array([-np.sin(rad), np.cos(rad)])# kierunek osi małej

# cztery punkty kardynalne
R = (cx - A*v[0], cy - A*v[1])  # right
L = (cx + A*v[0], cy + A*v[1])  # left
D = (cx - B*u[0], cy - B*u[1])  # down
U = (cx + B*u[0], cy + B*u[1])  # up

# podgląd
vis = cv.cvtColor(mask_closed, cv.COLOR_GRAY2BGR)
cv.ellipse(vis, ellipse, (0,255,0), 2)
for label, (x,y) in {"L":L, "R":R, "U":U, "D":D}.items():
    p = (int(round(x)), int(round(y)))
    cv.circle(vis, p, 6, (255,0,255), -1)
    cv.putText(vis, label, (p[0]+6, p[1]-6),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

cv.imshow("ellipse + L/R/U/D", vis)

# 1) Zbierz punkty źródłowe (dokładnie w tej kolejności!)
src = np.float32([L, R, U, D])

# 2) Zdefiniuj punkty docelowe (okrąg w kwadracie 1000x1000)
dst = np.float32([
    [100, 500],  # LEFT
    [900, 500],  # RIGHT
    [500, 100],  # UP
    [500, 900],  # DOWN
])

# 3) Homografia + warp
M = cv.getPerspectiveTransform(src, dst)
rectified = cv.warpPerspective(img, M, (960, 960))

# 4) Podgląd / zapis
cv.imshow("rectified", rectified)
cv.imwrite("pics/rectified_dartboard.png", rectified)


cv.waitKey(0)

