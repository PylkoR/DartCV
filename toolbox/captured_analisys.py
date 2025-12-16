import cv2
import numpy as np
import os

# --- KONFIGURACJA ---
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "captured_throws")

if not os.path.exists(INPUT_FOLDER):
    print("Błąd: Brak folderu captured_throws. Uruchom najpierw skrypt przechwytujący.")
    exit()

# Wczytanie par zdjęć
samples = []
for i in range(4):
    p_bg = os.path.join(INPUT_FOLDER, f"bg_{i}.png")
    p_throw = os.path.join(INPUT_FOLDER, f"throw_{i}.png")
    
    if os.path.exists(p_bg) and os.path.exists(p_throw):
        bg = cv2.imread(p_bg)
        throw = cv2.imread(p_throw)
        samples.append((bg, throw))

if not samples:
    print("Brak zdjęć w folderze.")
    exit()

print(f"Wczytano {len(samples)} rzutów. Instrukcja:")
print("- SPACJA: Zmień rzut")
print("- Q: Zakończ i wypisz parametry")

WINDOW = "Dart Lab v2"
cv2.namedWindow(WINDOW)

def nothing(x): pass

# --- SUWAKI ---
# Domyślne wartości (takie jak w Twoim obecnym kodzie)
cv2.createTrackbar("Blur Kernel", WINDOW, 7, 31, nothing)       # NOWOŚĆ
cv2.createTrackbar("Threshold", WINDOW, 45, 255, nothing)       
cv2.createTrackbar("Glue Kernel", WINDOW, 12, 30, nothing)      
cv2.createTrackbar("Clean Kernel", WINDOW, 5, 20, nothing)      
cv2.createTrackbar("Min Noise Area", WINDOW, 600, 2000, nothing) 

current_sample_idx = 0

# Zmienne do przechowywania ostatnich ustawień
final_blur = 7
final_thresh = 45
final_glue = 12
final_clean = 5
final_noise = 600

while True:
    bg_color, throw_color = samples[current_sample_idx]
    
    # 1. Pobranie ustawień z suwaków
    blur_val = cv2.getTrackbarPos("Blur Kernel", WINDOW)
    th_val = cv2.getTrackbarPos("Threshold", WINDOW)
    glue_size = cv2.getTrackbarPos("Glue Kernel", WINDOW)
    clean_size = cv2.getTrackbarPos("Clean Kernel", WINDOW)
    min_noise = cv2.getTrackbarPos("Min Noise Area", WINDOW)

    # Walidacja (nieparzyste i > 0)
    if blur_val < 1: blur_val = 1
    if blur_val % 2 == 0: blur_val += 1
    
    if glue_size < 1: glue_size = 1
    if glue_size % 2 == 0: glue_size += 1
    
    if clean_size < 1: clean_size = 1
    if clean_size % 2 == 0: clean_size += 1

    # Aktualizacja zmiennych globalnych (do wypisania na koniec)
    final_blur = blur_val
    final_thresh = th_val
    final_glue = glue_size
    final_clean = clean_size
    final_noise = min_noise

    # 2. Przetwarzanie wstępne (Gray + Blur z suwaka)
    bg_gray = cv2.cvtColor(bg_color, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (blur_val, blur_val), 0) # Użycie suwaka
    
    throw_gray = cv2.cvtColor(throw_color, cv2.COLOR_BGR2GRAY)
    throw_gray = cv2.GaussianBlur(throw_gray, (blur_val, blur_val), 0) # Użycie suwaka

    # 3. Logika detekcji
    diff = cv2.absdiff(bg_gray, throw_gray)
    _, mask = cv2.threshold(diff, th_val, 255, cv2.THRESH_BINARY)
    
    # Morfologia
    k_glue = np.ones((glue_size, glue_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_glue)
    
    k_clean = np.ones((clean_size, clean_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_clean)
    
    # Filtr Szumów i Wizualizacja
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    
    output_img = throw_color.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > min_noise:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            
            # Kolorowanie: Zielony = OK, Niebieski = Duży Reset, Żółty = Prawie rzutka
            color = (0, 255, 0)
            if area > 7000: color = (255, 0, 0) # Reset
            
            cv2.drawContours(output_img, [cnt], -1, color, 2)
            cv2.putText(output_img, f"{int(area)}", tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Szary = Odrzucony szum
            cv2.drawContours(output_img, [cnt], -1, (128, 128, 128), 1)

    # 4. Wyświetlanie (Maska obok Obrazu)
    mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    
    # Dodanie napisów na obrazie
    cv2.putText(mask_bgr, f"Maska (Blur: {blur_val})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(output_img, f"Wynik (Thresh: {th_val})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    combined = np.hstack([output_img, mask_bgr])
    
    # Skalowanie okna
    h, w = combined.shape[:2]
    combined_resized = cv2.resize(combined, (int(w*0.5), int(h*0.5)))
    
    cv2.imshow(WINDOW, combined_resized)

    key = cv2.waitKey(30)
    if key & 0xFF == ord('q'): break
    elif key & 0xFF == ord(' '): 
        current_sample_idx = (current_sample_idx + 1) % len(samples)

cv2.destroyAllWindows()

# --- PODSUMOWANIE NA KONSOLI ---
print("\n" + "="*40)
print("   TWOJE OPTYMALNE PARAMETRY   ")
print("="*40)
print(f"# Skopiuj to do sekcji KONFIGURACJA lub FUNKCJI:")
print(f"BLUR_KERNEL_SIZE = ({final_blur}, {final_blur})")
print(f"MOTION_THRESHOLD = {final_thresh}")
print(f"KERNEL_GLUE_SIZE = ({final_glue}, {final_glue})")
print(f"KERNEL_CLEAN_SIZE = ({final_clean}, {final_clean})")
print(f"MIN_NOISE_AREA = {final_noise}")
print("="*40 + "\n")