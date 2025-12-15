import cv2
import numpy as np
import os
import json
import math

# --- KONFIGURACJA ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PROJECT_ROOT, "pics", "dart_normal.mp4")
JSON_PATH = os.path.join(PROJECT_ROOT, "output", "calibration_data.json")

# Parametry detekcji
MIN_AREA_DART = 800      
MAX_AREA_DART = 8000     
MAX_PLAYER_AREA = 12000  
MIN_FRAMES_STABLE = 15   
MOTION_THRESHOLD = 30    
CANVAS_SIZE = (960, 960) # Rozmiar wirtualnej tarczy do obliczeń

SCORE_MAP = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

# --- WCZYTANIE DANYCH ---
if not os.path.exists(JSON_PATH):
    print("Błąd: Brak pliku calibration_data.json!")
    exit()

with open(JSON_PATH, 'r') as f:
    calib_data = json.load(f)
    MATRIX = np.array(calib_data["perspective_matrix"])
    CENTER = tuple(calib_data["center"])
    RADIUS = calib_data["radius"]

print(f"Start systemu. Promień tarczy (wewn. obliczenia): {RADIUS:.1f}px")

# --- FUNKCJE ---

def transform_point(point, matrix):
    """Przelicza punkt z kamery na wirtualną, wyprostowaną tarczę."""
    pts = np.array([[[point[0], point[1]]]], dtype='float32')
    rect_pts = cv2.perspectiveTransform(pts, matrix)
    return tuple(map(int, rect_pts[0][0]))

def calculate_score(point, center, radius):
    """Oblicza wynik (matematyka na wyprostowanych współrzędnych)."""
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
    tip = tuple(all_points[all_points[:, :, 1].argmax()][0])
    return tip

# --- MAIN ---

cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret: exit()

# Inicjalizacja tła
raw_crop = first_frame[200:1280, :]
gray_bg = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, (7, 7), 0)

prev_gray = gray_bg.copy()
current_bg_gray = gray_bg.copy()

frames_stable = 0
# Historia przechowuje teraz: (punkt_RAW, punkty, etykieta)
game_history = [] 
total_score = 0
round_score = 0
last_action_text = "Czekam na rzut..."

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Obraz na żywo
    raw_crop = frame[200:1280, :] # To będziemy wyświetlać
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Stabilizacja
    delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(thresh) < 50:
        frames_stable += 1
    else:
        frames_stable = 0
    prev_gray = gray.copy()

    # 3. Logika (tylko gdy stabilnie)
    if frames_stable == MIN_FRAMES_STABLE:
        diff = cv2.absdiff(current_bg_gray, gray)
        _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Chmura punktów (Klejenie + Czyszczenie)
        kernel_glue = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_glue)
        kernel_clean = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        change_area = cv2.countNonZero(mask)

        # A. RZUT
        if MIN_AREA_DART < change_area < MAX_AREA_DART:
            raw_tip = find_tip_in_cloud(mask) # Punkt w układzie KAMERY
            
            if raw_tip:
                # Transformacja TYLKO do obliczeń
                rect_tip = transform_point(raw_tip, MATRIX)
                pts, label = calculate_score(rect_tip, CENTER, RADIUS)
                
                # Zapisujemy RAW_TIP do historii, żeby rysować na kamerze
                game_history.append((raw_tip, pts, label))
                
                total_score += pts
                round_score += pts
                last_action_text = f"Trafienie: {label} ({pts})"
                print(f"Rzut! {last_action_text}")
                
                current_bg_gray = gray.copy()

        # B. RESET
        elif change_area >= MAX_PLAYER_AREA:
            print("Reset rundy.")
            last_action_text = "Wyjmowanie lotek..."
            current_bg_gray = gray.copy()
            round_score = 0
            game_history.clear()

    # --- WIZUALIZACJA NA LIVE FEEDZIE ---
    display_img = raw_crop.copy() # Bierzemy aktualną klatkę wideo

    # Rysowanie historii (punkty "przyklejone" do obrazu)
    for (px, py), p_val, p_lbl in game_history:
        # Cień kropki dla lepszej widoczności
        cv2.circle(display_img, (px, py), 9, (0, 0, 0), -1) 
        cv2.circle(display_img, (px, py), 7, (0, 0, 255), -1) # Czerwona kropka
        cv2.circle(display_img, (px, py), 2, (0, 255, 255), -1) # Żółty środek

    # Interfejs (Półprzezroczysty pasek na górze)
    overlay = display_img.copy()
    cv2.rectangle(overlay, (0, 0), (1080, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)

    # Napisy
    cv2.putText(display_img, f"TOTAL: {total_score}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(display_img, f"RUNDA: {round_score}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    
    # Status na dole
    cv2.putText(display_img, last_action_text, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Dart Live System", display_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()