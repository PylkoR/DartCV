import cv2
import numpy as np
import os

# --- KONFIGURACJA ---
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
VIDEO_PATH = os.path.join(PROJECT_ROOT, "pics", "dart_normal.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Parametry detekcji (Twoje sprawdzone + próg szumu)
MIN_AREA_DART = 1000      
MAX_AREA_DART = 7000   
MAX_PLAYER_AREA = 7000  
MIN_FRAMES_STABLE = 15   
MOTION_THRESHOLD = 30    # Wróciłem do 45 (mniej szumu na starcie)

# NOWE: Wszystko co przetrwa morfologię, ale jest mniejsze niż to, wylatuje.
MIN_NOISE_AREA = 900       

# --- INICJALIZACJA ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret: exit()

raw_crop = first_frame[200:1280, :]
gray_bg = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, (5, 5), 0)

prev_gray = gray_bg.copy()
current_bg_gray = gray_bg.copy()
frames_stable = 0
last_valid_mask = np.zeros_like(gray_bg)

print("--- TRYB HYBRYDOWY (Twoja morfologia + Inteligentny filtr) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Przetwarzanie
    raw_crop = frame[200:1280, :]
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

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
        
        # --- ETAP 1: MORFOLOGIA (Twoja - sprawdzona) ---
        # Close (sklejanie) - ale mniejszym ziarnem, żeby nie robić siatki
        kernel_glue = np.ones((3, 3), np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_glue)
        
        # Open (czyszczenie) - to "gumkuje" drobny piasek, zanim się poskleja
        kernel_clean = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        # --- ETAP 2: INTELIGENTNY FILTR (Dla pewności) ---
        # Teraz, gdy morfologia zrobiła swoje, sprawdzamy co zostało.
        # Jeśli zostały 2 plamy: jedna wielka (rzutka) i jedna mała (błąd), 
        # to ten kod usunie tę małą.
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Zostawiamy tylko to, co jest większe niż szum
            if area > MIN_NOISE_AREA:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        
        mask = clean_mask
        change_area = cv2.countNonZero(mask)

        # --- WYNIKI ---
        if change_area > 50:
            last_valid_mask = mask.copy()
            
            cv2.putText(last_valid_mask, f"AREA: {change_area}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 3)

            print(f"[POMIAR] AREA: {change_area}", end="")
            
            if MIN_AREA_DART < change_area < MAX_AREA_DART:
                print(" -> RZUTKA")
                current_bg_gray = gray.copy()
                cv2.putText(last_valid_mask, "RZUTKA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150), 2)
                
            elif change_area >= MAX_PLAYER_AREA:
                print(" -> RESET")
                current_bg_gray = gray.copy()
                cv2.putText(last_valid_mask, "RESET", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150), 2)
            else:
                print(" -> ??? (Za male na rzutke)")
                cv2.putText(last_valid_mask, "???", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150), 2)

    # --- WIZUALIZACJA ---
    cv2.putText(raw_crop, f"Stable: {frames_stable}/{MIN_FRAMES_STABLE}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("OSTATNIA MASKA", last_valid_mask)
    cv2.imshow("Podglad Kamery", raw_crop)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask.png"), last_valid_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "photo.png"), raw_crop)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()