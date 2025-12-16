import cv2
import numpy as np
import math
import os
import json

# ==========================================
# --- 1. KONFIGURACJA GŁÓWNA ---
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(PROJECT_ROOT, "dart_game_video.mp4")
JSON_PATH = os.path.join(PROJECT_ROOT, "calibration_data.json")
OUTPUT_DIR = PROJECT_ROOT

# A. Parametry Obrazu i Filtrów
BLUR_KERNEL_SIZE = (5, 5)
MORPH_GLUE_SIZE = (3, 3)
MORPH_CLEAN_SIZE = (2, 2)

# B. Parametry Detekcji Ruchu
MOTION_THRESHOLD = 30
STABILITY_PIXEL_LIMIT = 35
MIN_FRAMES_STABLE = 15

# C. Parametry Logiki (Rozmiary obiektów)
MIN_AREA_DART = 1000
MAX_AREA_DART = 7000
MAX_PLAYER_AREA = 7000
MIN_NOISE_AREA = 900

# D. Ustawienia Tarczy i Gry
CANVAS_SIZE = (960, 960)     # Rozmiar wirtualnej tarczy
VIEW_SCALE = 0.6             # Skala widoku przy kalibracji (0.6 = widać otoczenie)
SCORE_MAP = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

# Zmienne globalne dla obsługi myszy (kalibracja)
g_perspective_points = []
g_grid_points = []
g_grid_step = 0

# ==========================================
# --- 2. FUNKCJE POMOCNICZE ---
# ==========================================

def order_points(pts):
    """Sortuje punkty do perspektywy: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def transform_point(point, matrix):
    """Przelicza punkt z kamery na wirtualną tarczę."""
    pts = np.array([[[point[0], point[1]]]], dtype='float32')
    rect_pts = cv2.perspectiveTransform(pts, matrix)
    return tuple(map(int, rect_pts[0][0]))

def calculate_score(point, center, radius):
    """Logika punktacji darta."""
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
    """Znajduje grot (najniższy punkt) na wyczyszczonej masce."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_pts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50: 
            valid_pts.append(cnt)
    
    if not valid_pts: return None
    all_points = np.vstack(valid_pts)
    tip = tuple(all_points[all_points[:, :, 1].argmax()][0])
    return tip

# ==========================================
# --- 3. MODUŁ KALIBRACJI ---
# ==========================================

def mouse_perspective(event, x, y, flags, param):
    global g_perspective_points
    if event == cv2.EVENT_LBUTTONDOWN and len(g_perspective_points) < 4:
        g_perspective_points.append([x, y])

def mouse_grid(event, x, y, flags, param):
    global g_grid_points, g_grid_step
    if event == cv2.EVENT_LBUTTONDOWN and g_grid_step < 3:
        g_grid_points.append((x, y))
        g_grid_step += 1

def run_calibration():
    print("\n--- ROZPOCZYNAM KALIBRACJĘ ---")
    global g_perspective_points, g_grid_points, g_grid_step
    
    # Reset zmiennych
    g_perspective_points = []
    g_grid_points = []
    g_grid_step = 0

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Błąd: Nie można pobrać klatki do kalibracji.")
        return False

    # Przycięcie (takie jak w grze)
    cropped_img = frame[200:1280, :] 
    img_display = cropped_img.copy()

    # KROK 1: 4 Rogi
    cv2.namedWindow("Kalibracja: Rogi")
    cv2.setMouseCallback("Kalibracja: Rogi", mouse_perspective)
    print("KROK 1: Kliknij 4 punkty na obwodzie tarczy (ESC aby anulować).")

    while len(g_perspective_points) < 4:
        temp_img = img_display.copy()
        for pt in g_perspective_points:
            cv2.circle(temp_img, tuple(pt), 5, (0, 255, 0), -1)
        if len(g_perspective_points) > 1:
            # Rysuj linię między ostatnimi punktami
            cv2.line(temp_img, tuple(g_perspective_points[-2]), tuple(g_perspective_points[-1]), (0, 255, 0), 1)
        
        cv2.imshow("Kalibracja: Rogi", temp_img)
        if cv2.waitKey(10) == 27: # ESC
            cv2.destroyAllWindows()
            return False
    cv2.destroyWindow("Kalibracja: Rogi")

    # Przetwarzanie Perspektywy
    pts_src = order_points(np.array(g_perspective_points, dtype="float32"))
    
    dst_w = int(CANVAS_SIZE[0] * VIEW_SCALE)
    dst_h = int(CANVAS_SIZE[1] * VIEW_SCALE)
    pad_x = (CANVAS_SIZE[0] - dst_w) / 2
    pad_y = (CANVAS_SIZE[1] - dst_h) / 2

    pts_dst = np.float32([
        [pad_x, pad_y],
        [pad_x + dst_w, pad_y],
        [pad_x + dst_w, pad_y + dst_h],
        [pad_x, pad_y + dst_h]
    ])

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    rectified_img = cv2.warpPerspective(cropped_img, matrix, CANVAS_SIZE)

    # KROK 2: Siatka
    grid_display = rectified_img.copy()
    cv2.namedWindow("Kalibracja: Srodek")
    cv2.setMouseCallback("Kalibracja: Srodek", mouse_grid)
    print("KROK 2: Kliknij ŚRODEK (Bullseye), potem 2x KRAWĘDŹ (Double Ring).")

    while g_grid_step < 3:
        temp_grid = grid_display.copy()
        colors = [(0, 255, 0), (0, 0, 255), (0, 0, 255)]
        
        for i, pt in enumerate(g_grid_points):
            cv2.circle(temp_grid, pt, 5, colors[i], -1)
            if i > 0: # Rysuj linię od środka do promienia
                cv2.line(temp_grid, g_grid_points[0], pt, (255, 255, 0), 1)

        cv2.imshow("Kalibracja: Srodek", temp_grid)
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            return False
    cv2.destroyWindow("Kalibracja: Srodek")

    # Obliczenia finałowe
    center = g_grid_points[0]
    r1 = math.dist(center, g_grid_points[1])
    r2 = math.dist(center, g_grid_points[2])
    avg_radius = (r1 + r2) / 2.0

    # Zapis
    data = {"center": center, "radius": avg_radius, "perspective_matrix": matrix.tolist()}
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Kalibracja zakończona! Dane zapisano w {JSON_PATH}")
    return True

# ==========================================
# --- 4. MODUŁ GRY (REALTIME) ---
# ==========================================

def run_game():
    print("\n--- URUCHAMIAM GRĘ ---")
    
    # Wczytanie konfiguracji
    if not os.path.exists(JSON_PATH):
        print("Brak pliku kalibracji! Uruchom kalibrację najpierw.")
        return

    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
        matrix = np.array(calib_data["perspective_matrix"])
        center = tuple(calib_data["center"])
        radius = calib_data["radius"]

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, first_frame = cap.read()
    if not ret: 
        print("Nie można otworzyć wideo/kamery.")
        return

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

    cv2.namedWindow("Dart System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dart System", 960, 960)

    print("Gra aktywna. Naciśnij 'q' lub 'ESC', aby wyjść do menu.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            # Pętla wideo dla plików
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        raw_crop = frame[200:1280, :] 
        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)

        # Stabilizacja
        delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        if cv2.countNonZero(thresh) < STABILITY_PIXEL_LIMIT:
            frames_stable += 1
        else:
            frames_stable = 0
        prev_gray = gray.copy()

        # Logika
        if frames_stable == MIN_FRAMES_STABLE:
            diff = cv2.absdiff(current_bg_gray, gray)
            _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Morfologia
            kernel_glue = np.ones(MORPH_GLUE_SIZE, np.uint8) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_glue)
            kernel_clean = np.ones(MORPH_CLEAN_SIZE, np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
            
            # Filtr Szumów
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(mask)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > MIN_NOISE_AREA:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            
            mask = clean_mask 
            change_area = cv2.countNonZero(mask)

            # --- ZDARZENIA GRY ---
            # 1. Rzut
            if MIN_AREA_DART < change_area < MAX_AREA_DART:
                raw_tip = find_tip_in_cloud(mask)
                if raw_tip:
                    rect_tip = transform_point(raw_tip, matrix)
                    pts, label = calculate_score(rect_tip, center, radius)
                    
                    game_history.append((raw_tip, pts, label))
                    total_score += pts
                    round_score += pts
                    last_action_text = f"Trafienie: {label} ({pts})"
                    print(f"> RZUT: {last_action_text} [Area: {change_area}]")
                    
                    current_bg_gray = gray.copy()

            # 2. Reset / Gracz
            elif change_area >= MAX_PLAYER_AREA:
                print(f"> RESET RUNDY (Wykryto gracza/ruch - Area: {change_area})")
                last_action_text = "Wyjmowanie lotek..."
                current_bg_gray = gray.copy()
                round_score = 0
                game_history.clear()

        # Wizualizacja
        display_img = raw_crop.copy() 
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

        cv2.imshow("Dart System", display_img)
        
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# --- MENU GŁÓWNE ---
# ==========================================

if __name__ == "__main__":
    while True:
        print("\n" + "="*30)
        print("   SYSTEM DART - MENU GŁÓWNE   ")
        print("="*30)
        
        has_calib = os.path.exists(JSON_PATH)
        
        if not has_calib:
            print("(!) Brak pliku kalibracji. Wymagana konfiguracja.")
            choice = 'c'
        else:
            print(f"[s] START GRY")
            print(f"[c] KALIBRACJA (Ustawienie tarczy)")
            print(f"[q] WYJŚCIE")
            choice = input("\nWybierz opcję: ").lower().strip()

        if choice == 'c':
            success = run_calibration()
            if success:
                print("Kalibracja udana. Możesz grać.")
            else:
                print("Anulowano kalibrację.")
        
        elif choice == 's':
            if has_calib:
                run_game()
            else:
                print("Najpierw musisz skalibrować system!")
        
        elif choice == 'q':
            print("Do widzenia!")
            break
        
        else:
            print("Nieznana opcja.")