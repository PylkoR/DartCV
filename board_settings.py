import cv2
import numpy as np
import math
import os
import json

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "dart_normal.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ustawienia obrazu
CANVAS_SIZE = (960, 960)     # Rozmiar okna wyjściowego
BOARD_SIZE = (660, 660)      # Rozmiar samej tarczy w pikselach
VIEW_SCALE = 0.6              # Skala widoku (mniejsza = więcej otoczenia)

# Zmienne globalne
perspective_points = []
grid_points = []
grid_step = 0

# --- Funkcje pomocnicze ---

def order_points(pts):
    """Sortuje punkty w kolejności: TL, TR, BR, BL, aby uniknąć odbić."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Lewy-Góra
    rect[2] = pts[np.argmax(s)] # Prawy-Dół
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Prawy-Góra
    rect[3] = pts[np.argmax(diff)] # Lewy-Dół
    return rect

def select_perspective_points(event, x, y, flags, param):
    """Klikanie 4 rogów do perspektywy."""
    global perspective_points
    if event == cv2.EVENT_LBUTTONDOWN and len(perspective_points) < 4:
        perspective_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        # Rysuj linie łączące dla podglądu
        if len(perspective_points) > 1:
            cv2.line(param, tuple(perspective_points[-2]), tuple(perspective_points[-1]), (0, 255, 0), 1)
        cv2.imshow("Krok 1: Zaznacz 4 rogi", param)

def select_grid_points(event, x, y, flags, param):
    """Klikanie środka i promieni."""
    global grid_points, grid_step
    img_display = param
    if event == cv2.EVENT_LBUTTONDOWN and grid_step < 3:
        grid_points.append((x, y))
        grid_step += 1
        colors = [(0, 255, 0), (0, 0, 255), (0, 0, 255)] # Zielony środek, czerwone krawędzie
        cv2.circle(img_display, (x, y), 5, colors[grid_step-1], -1)
        if grid_step > 1:
            cv2.line(img_display, grid_points[0], (x,y), (255, 255, 0), 1)
        cv2.imshow("Krok 2: Siatka", img_display)

# --- Główny proces ---

# 1. Wczytanie klatki
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret: exit("Błąd wideo")

cropped_img = frame[200:1280, :] # Przycięcie
img_display = cropped_img.copy()

# 2. Pobieranie 4 punktów (Perspektywa)
cv2.namedWindow("Krok 1: Zaznacz 4 rogi")
cv2.setMouseCallback("Krok 1: Zaznacz 4 rogi", select_perspective_points, img_display)
print("--- KROK 1: Zaznacz 4 punkty na obwodzie tarczy ---")
cv2.imshow("Krok 1: Zaznacz 4 rogi", img_display)

while len(perspective_points) < 4:
    if cv2.waitKey(10) == 27: exit()
cv2.destroyAllWindows()

# 3. Transformacja (Prostowanie + Zoom Out)
pts_src = np.array(perspective_points, dtype="float32")
pts_src = order_points(pts_src) # Automatyczne sortowanie punktów

# Obliczam rozmiar docelowy z uwzględnieniem skali (żeby widzieć otoczenie)
dst_w = int(CANVAS_SIZE[0] * VIEW_SCALE)
dst_h = int(CANVAS_SIZE[1] * VIEW_SCALE)

# Marginesy, żeby wycentrować pomniejszoną tarczę
pad_x = (CANVAS_SIZE[0] - dst_w) / 2
pad_y = (CANVAS_SIZE[1] - dst_h) / 2

# Punkty docelowe (TL, TR, BR, BL) - ściśnięte do środka
pts_dst = np.float32([
    [pad_x, pad_y],                 # TL
    [pad_x + dst_w, pad_y],         # TR
    [pad_x + dst_w, pad_y + dst_h], # BR
    [pad_x, pad_y + dst_h]          # BL
])

matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
rectified_img = cv2.warpPerspective(cropped_img, matrix, CANVAS_SIZE)

grid_display = rectified_img.copy()

# 4. Pobieranie punktów siatki (Środek + Promienie)
cv2.namedWindow("Krok 2: Siatka")
cv2.setMouseCallback("Krok 2: Siatka", select_grid_points, grid_display)
print("\n--- KROK 2: Kliknij środek (bullseye) i 2x krawędź (double) ---")
cv2.imshow("Krok 2: Siatka", grid_display)

while grid_step < 3:
    if cv2.waitKey(10) == 27: exit()
cv2.destroyAllWindows()

# 5. Obliczenia i rysowanie
center = grid_points[0]
r1 = math.dist(center, grid_points[1])
r2 = math.dist(center, grid_points[2])
avg_radius = (r1 + r2) / 2.0

print(f"\n--- WYNIK ---")
print(f"Środek: {center}, Promień: {avg_radius:.1f}")

final_img = rectified_img.copy()

# Rysowanie okręgów
factors = [0.035, 0.094, 0.58, 0.63, 0.94, 1.0]
for f in factors:
    cv2.circle(final_img, center, int(avg_radius * f), (0, 255, 255), 2)

# Rysowanie linii sektorów
start_r = avg_radius * 0.094
for i in range(20):
    rad = math.radians(i * 18 - 9)
    # Zwiększamy mnożnik końca linii (np. * 1.5), żeby wychodziły poza pole punktowe
    p1 = (int(center[0] + start_r * math.cos(rad)), int(center[1] - start_r * math.sin(rad)))
    p2 = (int(center[0] + avg_radius * 1.1 * math.cos(rad)), int(center[1] - avg_radius * 1.1 * math.sin(rad)))
    cv2.line(final_img, p1, p2, (255, 0, 255), 2)

cv2.imshow("Wynik", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zapis
cv2.imwrite(os.path.join(OUTPUT_DIR, "board_rectified.png"), rectified_img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "board_with_grid.png"), final_img)

data = {"center": center, "radius": avg_radius, "perspective_matrix": matrix.tolist()}
with open(os.path.join(OUTPUT_DIR, "calibration_data.json"), 'w') as f:
    json.dump(data, f, indent=4)