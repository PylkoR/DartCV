import cv2 as cv
import numpy as np

IMG = r"pics/frame1.png"   # <-- podmień na swoją klatkę
SCALE = 0.5                # zmniejsz obraz, żeby liczyło szybciej (0.5 = połowa rozdzielczości)

img0 = cv.imread(IMG)
if img0 is None:
    raise SystemExit(f"❌ Nie mogę wczytać obrazu: {IMG}")

# --- skalowanie dla wydajności ---
if SCALE != 1.0:
    img = cv.resize(img0, (0, 0), fx=SCALE, fy=SCALE)
else:
    img = img0.copy()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)

found = None

def draw_result():
    vis = img.copy()
    if found is not None:
        x, y, r = found
        cv.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv.circle(vis, (x, y), 4, (0, 0, 255), -1)
        cv.putText(vis, f"center=({x},{y}) r={r}", (30, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.imshow("circle finder", vis)

print("""
[instrukcja]
r - znajdź okrąg (HoughCircles)
m - kliknij centrum, potem punkt na obwodzie (tryb ręczny)
s - zapisz wynik do dart_outer_circle.npz
q - wyjdź
""")

manual_pts = []

def on_mouse(event, x, y, flags, param):
    global manual_pts
    if event == cv.EVENT_LBUTTONDOWN:
        manual_pts.append((x, y))

cv.namedWindow("circle finder", cv.WINDOW_NORMAL)
cv.resizeWindow("circle finder", 900, 900)
cv.setMouseCallback("circle finder", on_mouse)

draw_result()

while True:
    key = cv.waitKey(50) & 0xFF
    if key == ord('q'):
        break

    # 🔹 automatyczne wyszukiwanie okręgu po 'r'
    if key == ord('r'):
        print("⏳ Szukam okręgu...")
        circles = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT, dp=1.2, minDist=300,
            param1=100, param2=40, minRadius=200, maxRadius=800
        )
        if circles is not None:
            c = np.uint16(np.around(circles[0][0]))
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            found = (x, y, r)
            print(f"✅ Znaleziono okrąg: center=({x}, {y}), r={r}")
        else:
            print("⚠️  Nie znaleziono okręgu – spróbuj ręcznie (m)")
        draw_result()

    # 🔹 ręczne kliknięcie: centrum + punkt na obwodzie
    if key == ord('m'):
        manual_pts = []
        print("Kliknij centrum, potem punkt na obwodzie (2 kliknięcia)...")
        while len(manual_pts) < 2:
            cv.waitKey(10)
        (cx, cy), (px, py) = manual_pts
        r = int(np.hypot(px - cx, py - cy))
        found = (cx, cy, r)
        print(f"✅ Ręcznie: center=({cx}, {cy}), r={r}")
        draw_result()

    # 🔹 zapis wyniku po 's'
    if key == ord('s') and found is not None:
        x, y, r = found
        # przeskalowanie do oryginalnej rozdzielczości
        if SCALE != 1.0:
            x, y, r = x / SCALE, y / SCALE, r / SCALE
        np.savez("dart_outer_circle.npz", cx=x, cy=y, R=r)
        print(f"💾 Zapisano dart_outer_circle.npz: center=({x:.1f}, {y:.1f}), R={r:.1f}")
