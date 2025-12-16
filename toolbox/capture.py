import cv2
import numpy as np
import os

# --- KONFIGURACJA ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PROJECT_ROOT, "pics", "dart_normal.mp4") 
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "captured_throws")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Parametry (IDENTYCZNE JAK W TWOIM DZIAŁAJĄCYM KODZIE)
MIN_AREA_DART = 700      
MAX_AREA_DART = 7000   
MAX_PLAYER_AREA = 7000  
MIN_FRAMES_STABLE = 15   
MOTION_THRESHOLD = 45    
MIN_NOISE_AREA = 600      

# --- INICJALIZACJA ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret: exit()

raw_crop = first_frame[200:1280, :]
gray_bg = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, (7, 7), 0)

prev_gray = gray_bg.copy()
current_bg_gray = gray_bg.copy()
# Dodatkowo: musimy pamiętać kolorowe tło, żeby je zapisać na dysk
current_bg_color = raw_crop.copy() 

frames_stable = 0
shots_captured = 0
MAX_SHOTS = 4

print(f"--- TRYB PRZECHWYTYWANIA (LOGIKA UŻYTKOWNIKA) ---")
print(f"Zapiszę {MAX_SHOTS} rzuty do folderu: {OUTPUT_FOLDER}")

while shots_captured < MAX_SHOTS:
    ret, frame = cap.read()
    if not ret: break

    # 1. Przetwarzanie
    raw_crop = frame[200:1280, :]
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Stabilizacja
    delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    if cv2.countNonZero(thresh) < 35:
        frames_stable += 1
    else:
        frames_stable = 0
    prev_gray = gray.copy()

    # 3. Logika detekcji
    if frames_stable == MIN_FRAMES_STABLE:
        
        diff = cv2.absdiff(current_bg_gray, gray)
        _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # --- ETAP 1: MORFOLOGIA (Twoja) ---
        kernel_glue = np.ones((12, 12), np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_glue)
        
        kernel_clean = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        # --- ETAP 2: INTELIGENTNY FILTR (Twój) ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_NOISE_AREA:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        
        mask = clean_mask
        change_area = cv2.countNonZero(mask)

        # --- WYNIKI I ZAPIS ---
        if MIN_AREA_DART < change_area < MAX_AREA_DART:
            print(f" -> ZAPISANO RZUT NR {shots_captured+1} (Area: {change_area})")
            
            # ZAPISUJEMY ZDJĘCIA
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"bg_{shots_captured}.png"), current_bg_color)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"throw_{shots_captured}.png"), raw_crop)
            
            shots_captured += 1
            
            # Aktualizacja tła
            current_bg_gray = gray.copy()
            current_bg_color = raw_crop.copy()
            
            # Reset stabilizacji (żeby nie zapisać tego samego rzutu 2 razy pod rząd)
            frames_stable = 0

        elif change_area >= MAX_PLAYER_AREA:
            print(f" -> RESET (Gracz/Zbieranie - Area: {change_area})")
            current_bg_gray = gray.copy()
            current_bg_color = raw_crop.copy()
            frames_stable = 0

    # Wizualizacja
    cv2.putText(raw_crop, f"Saved: {shots_captured}/{MAX_SHOTS}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Capture", raw_crop)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Gotowe. Teraz odpal 2_tuning_lab.py")