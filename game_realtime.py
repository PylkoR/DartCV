# SKRYPT DO DETEKCJI RZUTÓW REALTIME
import cv2
import numpy as np
import os
import json
import math

# --- KONFIGURACJA SYSTEMU ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PROJECT_ROOT, "dart_normal.mp4")
JSON_PATH = os.path.join(PROJECT_ROOT, "calibration_data.json")

# 1. Parametry Obrazu i Filtrów (Jakość obrazu)
BLUR_KERNEL_SIZE = (5, 5)       # Rozmycie obrazu (wygładzanie szumów kamery)
MORPH_GLUE_SIZE = (3, 3)        # "Klej" (Close) - łączy rozerwane elementy rzutki
MORPH_CLEAN_SIZE = (2, 2)       # "Czyszczenie" (Open) - usuwa drobny piasek/szum

# 2. Parametry Detekcji Ruchu (Czułość)
MOTION_THRESHOLD = 30           # Próg różnicy kolorów (0-255). Mniej = czulej.
STABILITY_PIXEL_LIMIT = 35      # Ile pikseli może drgać, żeby uznać obraz za stabilny
MIN_FRAMES_STABLE = 15          # Ile klatek musi być stabilnie, żeby zaliczyć rzut

# 3. Parametry Logiki (Rozmiary obiektów)
MIN_AREA_DART = 1000            # Minimalna wielkość rzutki (px)
MAX_AREA_DART = 7000            # Maksymalna wielkość rzutki (px)
MAX_PLAYER_AREA = 7000          # Powyżej tego uznajemy, że to człowiek (Reset)
MIN_NOISE_AREA = 900            # Filtr końcowy: ignoruj obiekty mniejsze niż to (px)

# 4. Parametry Gry
CANVAS_SIZE = (960, 960) 
SCORE_MAP = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]


# --- WCZYTANIE KALIBRACJI TARCZY ---
if not os.path.exists(JSON_PATH):
    print("Błąd: Brak pliku calibration_data.json!")
    exit()

with open(JSON_PATH, 'r') as f:
    calib_data = json.load(f)
    MATRIX = np.array(calib_data["perspective_matrix"])
    CENTER = tuple(calib_data["center"])
    RADIUS = calib_data["radius"]

print(f"--- START SYSTEMU ---")
print(f"Promień tarczy: {RADIUS:.1f}px")
print(f"Filtry: Blur={BLUR_KERNEL_SIZE}, Glue={MORPH_GLUE_SIZE}, Clean={MORPH_CLEAN_SIZE}")
print(f"Logika: MinDart={MIN_AREA_DART}, NoiseFilter={MIN_NOISE_AREA}")

# --- FUNKCJE POMOCNICZE ---
def transform_point(point, matrix):
    """Przelicza punkt z kamery na wyprostowaną tarczę."""
    pts = np.array([[[point[0], point[1]]]], dtype='float32')
    rect_pts = cv2.perspectiveTransform(pts, matrix)
    return tuple(map(int, rect_pts[0][0]))

def calculate_score(point, center, radius):
    """Oblicza wynik"""
    x, y = point
    dx = x - center[0]
    dy = y - center[1]
    dist = math.sqrt(dx**2 + dy**2)
    
    theta = math.degrees(math.atan2(-dy, dx))
    if theta < 0: theta += 360

    r_norm = dist / radius
    multiplier = 0
    label = "Miss"
    
    if r_norm <= 0.035: return 50, "BULLSEYE"
    elif r_norm <= 0.094: return 25, "BULL"
    elif 0.58 <= r_norm <= 0.63: multiplier = 3; label = "T"
    elif 0.94 <= r_norm <= 1.0: multiplier = 2; label = "D"
    elif r_norm < 0.94: multiplier = 1; label = ""
    else: return 0, "OUT"

    sector_idx = int(((theta + 9) % 360) / 18)
    val = SCORE_MAP[sector_idx]
    return val * multiplier, f"{label}{val}"

def find_tip_in_cloud(mask):
    """Znajduje najniższy punkt (Tip) w chmurze konturów."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_pts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50: 
            valid_pts.append(cnt)
            
    if not valid_pts: return None
    
    all_points = np.vstack(valid_pts)
    # Szukamy punktu z największym Y
    tip = tuple(all_points[all_points[:, :, 1].argmax()][0])
    return tip

# --- PĘTLA GŁÓWNA ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret: exit()

# Inicjalizacja tła
raw_crop = first_frame[200:1280, :]
gray_bg = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, BLUR_KERNEL_SIZE, 0)

prev_gray = gray_bg.copy()
current_bg_gray = gray_bg.copy()

frames_stable = 0
game_history = [] 
total_score = 0
round_score = 0
last_action_text = "Czekam na rzut..."

WINDOW_NAME = "Dart Live System"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 960)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Obraz na żywo
    raw_crop = frame[200:1280, :] 
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)

    # 2. Stabilizacja
    delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    if cv2.countNonZero(thresh) < STABILITY_PIXEL_LIMIT:
        frames_stable += 1
    else:
        frames_stable = 0
    prev_gray = gray.copy()

    # 3. Logika (tylko gdy stabilnie)
    if frames_stable == MIN_FRAMES_STABLE:
        diff = cv2.absdiff(current_bg_gray, gray)
        _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # --- ZAAWANSOWANE PRZETWARZANIE MASKI ---
        
        # A. Morfologia - Użycie zdefiniowanych kerneli
        kernel_glue = np.ones(MORPH_GLUE_SIZE, np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_glue)
        
        kernel_clean = np.ones(MORPH_CLEAN_SIZE, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        # B. Inteligentny Filtr Szumów
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_NOISE_AREA:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        
        mask = clean_mask 
        
        # --- KONIEC PRZETWARZANIA ---
        
        change_area = cv2.countNonZero(mask)

        # Logika Gry
        # A. RZUT
        if MIN_AREA_DART < change_area < MAX_AREA_DART:
            raw_tip = find_tip_in_cloud(mask)
            
            if raw_tip:
                rect_tip = transform_point(raw_tip, MATRIX)
                pts, label = calculate_score(rect_tip, CENTER, RADIUS)
                
                game_history.append((raw_tip, pts, label))
                
                total_score += pts
                round_score += pts
                last_action_text = f"Trafienie: {label} ({pts})"
                print(f"Rzut. Wynik: {last_action_text}")
                
                current_bg_gray = gray.copy()

        # B. RESET
        elif change_area >= MAX_PLAYER_AREA:
            print(f"Reset rundy. Area: {change_area}")
            last_action_text = "Wyjmowanie lotek..."
            current_bg_gray = gray.copy()
            round_score = 0
            game_history.clear()

    # --- WIZUALIZACJA ---
    display_img = raw_crop.copy() 

    # Rysowanie historii
    for (px, py), p_val, p_lbl in game_history:
        cv2.circle(display_img, (px, py), 9, (0, 0, 0), -1) 
        cv2.circle(display_img, (px, py), 7, (0, 0, 255), -1) 
        cv2.circle(display_img, (px, py), 2, (0, 255, 255), -1) 

    # Interfejs
    overlay = display_img.copy()
    cv2.rectangle(overlay, (0, 0), (1080, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)

    cv2.putText(display_img, f"TOTAL: {total_score}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(display_img, f"RUNDA: {round_score}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.putText(display_img, last_action_text, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, display_img)
    
    if cv2.waitKey(10) == 27: break

cap.release()
cv2.destroyAllWindows()