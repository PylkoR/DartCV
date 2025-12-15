import cv2
import numpy as np

# transformacja perspektywiczna obrazu tarczy

# --- Konfiguracja ---
IMG_PATH = "output/frame1.png"
OUTPUT_PATH = "output/board_rectified.png"

# Rozmiar całego obrazu wyjściowego (nasze "płótno")
CANVAS_SIZE = (960, 960)

# Rozmiar, jaki ma mieć sama tarcza na obrazie wyjściowym.
# Musi być mniejszy niż CANVAS_SIZE, aby zmieścił się margines.
TARGET_BOARD_SIZE = (660, 660)

# 1. Wklej tutaj swoje PRECYZYJNE punkty kliknięte na tarczy
pts_src_precise = np.float32([[191.0, 160.0], [886.0, 156.0], [141.0, 836.0], [927.0, 828.0]])
pts_src_precise = np.float32([[248.0, 227.0], [816.0, 217.0], [223.0, 766.0], [846.0, 759.0]])

# --- Główna część skryptu ---

# 2. Oblicz margines (offset), aby wycentrować tarczę na płótnie
canvas_w, canvas_h = CANVAS_SIZE
target_w, target_h = TARGET_BOARD_SIZE

offset_x = (canvas_w - target_w) / 2
offset_y = (canvas_h - target_h) / 2

# 3. Zdefiniuj punkty docelowe dla PRECYZYJNYCH kliknięć.
# Będą to narożniki naszej docelowej tarczy, przesunięte o margines.
pts_dst_precise = np.float32([
    [offset_x, offset_y],                               # Lewy-górny
    [offset_x + target_w, offset_y],                    # Prawy-górny
    [offset_x, offset_y + target_h],                    # Lewy-dolny
    [offset_x + target_w, offset_y + target_h]          # Prawy-dolny
])

# 4. Wczytaj obraz
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {IMG_PATH}")
else:
    # 5. Oblicz POPRAWNĄ macierz transformacji perspektywicznej
    matrix = cv2.getPerspectiveTransform(pts_src_precise, pts_dst_precise)

    # 6. Zastosuj transformację, renderując wynik na dużym płótnie
    corrected_img = cv2.warpPerspective(img, matrix, CANVAS_SIZE)

    # 7. Zapisz i wyświetl wyniki
    cv2.imwrite(OUTPUT_PATH, corrected_img)
    print(f"Skorygowany obraz został zapisany jako: {OUTPUT_PATH}")

    # (Opcjonalnie) Narysuj docelową ramkę na wynikowym obrazie
    cv2.rectangle(corrected_img, (int(offset_x), int(offset_y)), 
                  (int(offset_x + target_w), int(offset_y + target_h)), 
                  (0, 255, 255), 2)

    cv2.imshow("Oryginalny Obraz", img)
    cv2.imshow("Poprawnie Skorygowany Obraz", corrected_img)
    
    print("\nNaciśnij dowolny klawisz, aby zamknąć okna.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()