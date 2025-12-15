import cv2
import numpy as np

# Zbiera punkty potrzebne do tranformacji perspektywiczenej

# --- Konfiguracja ---
IMG_PATH = "output/frame1.png"
points = [] # Lista do przechowywania współrzędnych punktów

# --- Funkcja callback dla zdarzeń myszy ---
def select_point(event, x, y, flags, param):
    """
    Funkcja obsługująca kliknięcia myszy.
    Dodaje punkt po kliknięciu lewym przyciskiem myszy.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        # Dodaj współrzędne do listy
        points.append([x, y])
        print(f"Dodano punkt nr {len(points)}: ({x}, {y})")
        
        # Narysuj kółko w miejscu kliknięcia dla wizualizacji
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Zaznacz 4 punkty", img)

# --- Główna część skryptu ---

# 1. Wczytaj obraz
img = cv2.imread(IMG_PATH)

if img is None:
    print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {IMG_PATH}")
else:
    # Kopia obrazu, aby oryginalny pozostał nienaruszony
    img_clone = img.copy()
    
    # 2. Stwórz okno i ustaw obsługę myszy
    cv2.namedWindow("Zaznacz 4 punkty")
    cv2.setMouseCallback("Zaznacz 4 punkty", select_point)

    print("Zaznacz 4 punkty na tarczy w następującej kolejności:")
    print("1. lewy-górny, 2. prawy-górny, 3. lewy-dolny, 4. prawy-dolny")
    print("Po zaznaczeniu 4 punktów, okno zamknie się automatycznie.")
    
    # 3. Wyświetl obraz i czekaj na akcję użytkownika
    cv2.imshow("Zaznacz 4 punkty", img)
    
    # Czekaj, aż użytkownik zaznaczy 4 punkty
    while len(points) < 4:
        cv2.waitKey(1) # Czekaj 1ms, aby okno mogło się odświeżyć

    # 4. Zamknij okno po zebraniu punktów
    cv2.destroyAllWindows()

    # 5. Wypisz finalną listę punktów w formacie gotowym do użycia
    if len(points) == 4:
        pts_src = np.array(points, dtype=np.float32)
        print("\n--- Zakończono ---")
        print(f"pts_src = np.float32({pts_src.tolist()})")