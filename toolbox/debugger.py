import cv2
import numpy as np
import os

# --- KONFIGURACJA ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Ustaw 0 dla kamery USB, lub ścieżkę do pliku wideo
VIDEO_PATH = os.path.join(PROJECT_ROOT, "pics", "dart_normal.mp4") 

# Parametry do dostrojenia:
MIN_AREA_DART = 800      
MAX_AREA_DART = 8000   
MAX_PLAYER_AREA = 8000  
MIN_FRAMES_STABLE = 15   
MOTION_THRESHOLD = 45

# --- INICJALIZACJA ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    print("Błąd: Nie można otworzyć wideo/kamery.")
    exit()

# Przycinamy i przygotowujemy tło
raw_crop = first_frame[200:1280, :]
gray_bg = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
gray_bg = cv2.GaussianBlur(gray_bg, (7, 7), 0)

prev_gray = gray_bg.copy()
current_bg_gray = gray_bg.copy()
frames_stable = 0

# Zmienna przechowująca OSTATNIĄ widoczną maskę (żeby nie znikała)
last_valid_mask = np.zeros_like(gray_bg)

print("--- TRYB KALIBRACJI (Z PAMIĘCIĄ OBRAZU) ---")
print(f"Parametry: Min={MIN_AREA_DART}, Max={MAX_AREA_DART}, Player={MAX_PLAYER_AREA}")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Przetwarzanie
    raw_crop = frame[200:1280, :]
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Stabilizacja
    delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    status_color = (0, 0, 255) # Czerwony (ruch)
    
    if cv2.countNonZero(thresh) < 35:
        frames_stable += 1
    else:
        frames_stable = 0
    
    prev_gray = gray.copy()

    # 3. Logika detekcji (Moment pomiaru)
    if frames_stable == MIN_FRAMES_STABLE:
        status_color = (0, 255, 0) # Zielony (stabilnie)
        
        diff = cv2.absdiff(current_bg_gray, gray)
        _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Morfologia
        kernel_glue = np.ones((12, 12), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_glue)
        kernel_clean = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
        
        change_area = cv2.countNonZero(mask)

        # JEŚLI COŚ WYKRYTO -> ZAPISUJEMY MASKĘ DO ZMIENNEJ GLOBALNEJ PĘTLI
        if change_area > 50:
            last_valid_mask = mask.copy() # <--- TUTAJ JEST ZMIANA
            
            # Wypisywanie na maskę wartości AREA, żebyś nie musiał patrzeć na konsolę
            cv2.putText(last_valid_mask, f"AREA: {change_area}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 3)

            print(f"[POMIAR] AREA: {change_area}", end="")
            
            if MIN_AREA_DART < change_area < MAX_AREA_DART:
                print(" -> RZUTKA")
                current_bg_gray = gray.copy()
                cv2.putText(last_valid_mask, "TYP: RZUTKA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150), 2)
                
            elif change_area >= MAX_PLAYER_AREA:
                print(" -> RESET")
                current_bg_gray = gray.copy()
                cv2.putText(last_valid_mask, "TYP: RESET", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150), 2)
            else:
                print(" -> ??? (Dostosuj parametry)")
                cv2.putText(last_valid_mask, "TYP: ???", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150), 2)

    # --- WIZUALIZACJA ---
    
    cv2.putText(raw_crop, f"Stable: {frames_stable}/{MIN_FRAMES_STABLE}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Wyświetlamy ostatnią zapamiętaną maskę zamiast aktualnej (która może być czarna)
    cv2.imshow("OSTATNIA WYKRYTA MASKA", last_valid_mask)
    cv2.imshow("Podglad Kamery", raw_crop)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1) 

cap.release()
cv2.destroyAllWindows()