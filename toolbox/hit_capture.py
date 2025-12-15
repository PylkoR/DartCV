import cv2
import numpy as np
import os
import glob

# --- Konfiguracja ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
THROWS_DIR = os.path.join(PROJECT_ROOT, "output", "throws")
PIC_DIR = os.path.join(THROWS_DIR, "pics")
MASK_DIR = os.path.join(THROWS_DIR, "masks")
DEBUG_DIR = os.path.join(THROWS_DIR, "debug_cloud_logic")

os.makedirs(DEBUG_DIR, exist_ok=True)

mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))

if not mask_files:
    print("Brak masek.")
    exit()

print(f"Analiza {len(mask_files)} rzutów metodą 'Chmura Punktów'...")

for mask_path in mask_files:
    filename = os.path.basename(mask_path)
    # Zakładam, że plik zdjęcia nazywa się analogicznie (rect/mask)
    # Dostosuj replace jeśli masz inne nazewnictwo plików
    pic_filename = filename.replace("_mask", "_rect") 
    pic_path = os.path.join(PIC_DIR, pic_filename)
    
    if not os.path.exists(pic_path):
        continue

    # Wczytaj
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img_viz = cv2.imread(pic_path)

    # 1. Znajdź WSZYSTKIE kontury
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    
    # 2. Filtrowanie śmieci
    for cnt in contours:
        # Bierzemy wszystko co ma więcej niż 50px (piórko, grot, kawałki)
        if cv2.contourArea(cnt) > 50:
            valid_contours.append(cnt)
            # Rysujemy na niebiesko wszystkie uwzględnione kawałki
            cv2.drawContours(img_viz, [cnt], -1, (255, 0, 0), 1)

    if not valid_contours:
        print(f"Pusta maska: {filename}")
        continue

    # 3. ZŁĄCZENIE PUNKTÓW (Kluczowa zmiana)
    # Tworzymy jedną wielką tablicę punktów ze wszystkich kawałków
    all_points = np.vstack(valid_contours)
    
    # 4. Znalezienie punktu wbicia (Heurystyka: Najniższy punkt - Max Y)
    # extBot = punkt o największej współrzędnej Y w całej chmurze
    c = all_points
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Alternatywa: Najwyższy punkt (Min Y) - odkomentuj jeśli kamera jest pod tarczą
    # extTop = tuple(c[c[:, :, 1].argmin()][0]) 

    # --- Wizualizacja ---
    
    # Rysujemy Bounding Box wokół CAŁOŚCI (wszystkich kawałków)
    x, y, w, h = cv2.boundingRect(all_points)
    cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 255), 1)

    # Rysujemy wykryty punkt wbicia (ZIELONA KROPA)
    cv2.circle(img_viz, extBot, 8, (0, 255, 0), -1)
    cv2.circle(img_viz, extBot, 2, (0, 0, 0), -1) # Środek
    
    cv2.putText(img_viz, "TIP (Max Y)", (extBot[0] - 40, extBot[1] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Zapis
    save_path = os.path.join(DEBUG_DIR, f"cloud_{filename.replace('.png', '.jpg')}")
    cv2.imwrite(save_path, img_viz)

print(f"Gotowe. Sprawdź folder: {DEBUG_DIR}")