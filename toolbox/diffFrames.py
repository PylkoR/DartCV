import cv2
import numpy as np
import os
import json

# --- Konfiguracja Ścieżek ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Wyjście poziom wyżej
VIDEO_PATH = os.path.join(PROJECT_ROOT, "pics", "dart_normal.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "throws")
OUTPUT_DIR_PIC = os.path.join(OUTPUT_DIR, "pics")
OUTPUT_DIR_MASK = os.path.join(OUTPUT_DIR, "masks")
JSON_PATH = os.path.join(PROJECT_ROOT, "output", "calibration_data.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PIC, exist_ok=True)
os.makedirs(OUTPUT_DIR_MASK, exist_ok=True)

# --- Parametry ---
LIMIT_TIME_MS = 60000    # Limit 60 sekund
MIN_AREA_DART = 1000     # Min pixeli dla lotki
MAX_AREA_DART = 6000     # Max pixeli (powyżej to gracz)
MIN_FRAMES_STABLE = 15   # Czas stabilizacji
MOTION_THRESHOLD = 35    # Czułość różnicy
CANVAS_SIZE = (960, 960) # Rozmiar wyprostowanego obrazu

# --- Wczytanie Macierzy Transformacji ---
if not os.path.exists(JSON_PATH):
    print("Błąd: Brak pliku calibration_data.json. Uruchom najpierw kalibrację.")
    exit()

with open(JSON_PATH, 'r') as f:
    data = json.load(f)
    matrix = np.array(data["perspective_matrix"])

# --- Inicjalizacja Wideo ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# Pobranie tła (pierwsza klatka)
ret, first_frame = cap.read()
if not ret: exit()

# Crop musi być taki sam jak przy kalibracji!
bg_raw = first_frame[200:1280, :] 
gray_bg = cv2.cvtColor(bg_raw, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, (7, 7), 0)

prev_gray = gray_bg.copy()
current_bg_gray = gray_bg.copy() # Tło referencyjne (zaktualizowane o wbite lotki)

frames_stable_counter = 0
throw_counter = 0

print(f"Przetwarzanie pierwszych {LIMIT_TIME_MS/1000}s nagrania...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Preprocessing
    raw_cropped = frame[200:1280, :]
    gray = cv2.cvtColor(raw_cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Wykrywanie ruchu (Frame vs Previous Frame)
    frame_delta = cv2.absdiff(prev_gray, gray)
    thresh_delta = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_area = cv2.countNonZero(thresh_delta)
    prev_gray = gray.copy()

    # 3. Logika Stabilizacji
    if motion_area < 50:
        frames_stable_counter += 1
    else:
        frames_stable_counter = 0

    # 4. Analiza po ustabilizowaniu
    if frames_stable_counter == MIN_FRAMES_STABLE:
        # Różnica względem tła gry (current_bg)
        diff_bg = cv2.absdiff(current_bg_gray, gray)
        _, mask_bg = cv2.threshold(diff_bg, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Czyszczenie szumu
        kernel = np.ones((4, 4), np.uint8)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel, iterations=1)
        change_area = cv2.countNonZero(mask_bg)

        if MIN_AREA_DART < change_area < MAX_AREA_DART:
            # ---> WYKRYTO RZUT <---
            throw_counter += 1
            print(f"Rzut #{throw_counter} (Area: {change_area})")
            
            # Prostowanie obrazu i maski przed zapisem
            rectified_img = cv2.warpPerspective(raw_cropped, matrix, CANVAS_SIZE)
            rectified_mask = cv2.warpPerspective(mask_bg, matrix, CANVAS_SIZE)

            # Zapis
            cv2.imwrite(os.path.join(OUTPUT_DIR_PIC, f"throw_{throw_counter}_rect.png"), rectified_img)
            cv2.imwrite(os.path.join(OUTPUT_DIR_MASK, f"throw_{throw_counter}_mask.png"), rectified_mask)
            
            # Aktualizacja tła (dodajemy nową lotkę do "pustej" tarczy)
            current_bg_gray = gray.copy()

        elif change_area >= MAX_AREA_DART:
            # ---> RESET (GRACZ PRZY TARCZY) <---
            print(f"Reset tła (Gracz przy tarczy). Area: {change_area}")
            current_bg_gray = gray.copy()

    # (Opcjonalnie) Podgląd na żywo - zakomentuj dla szybkości
    cv2.imshow("Live", raw_cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()