import cv2
import numpy as np
import math

# --- Konfiguracja Siatki (Musisz uzupełnić!) ---
# Wklej tutaj wartości center i board_radius uzyskane z poprzedniego skryptu kalibracji
# CALIBRATED_CENTER = (474, 487)
# CALIBRATED_BOARD_RADIUS = 408.55
CALIBRATED_CENTER = (479, 481)
CALIBRATED_BOARD_RADIUS = 440.04

# --- Ścieżki i zmienne globalne ---
IMG_PATH = "pics/board_corrected.png" # Ścieżka do Twojego wyprostowanego obrazu tarczy
points = [] # Lista do przechowywania wszystkich zaznaczonych trafień

# Słownik mapujący sektory kątowe na punkty bazowe
SCORE_MAP = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

# --- Funkcja do obliczania wyniku (przeniesiona i dostosowana) ---
def get_score(x, y, center, radius):
    """
    Oblicza wynik rzutki na podstawie współrzędnych (x, y) i parametrów siatki.
    """
    dx = x - center[0]
    dy = y - center[1]
    
    r = math.sqrt(dx**2 + dy**2)
    theta = math.degrees(math.atan2(-dy, dx))
    if theta < 0:
        theta += 360

    multiplier = 0
    
    # Progi promieni (wartości procentowe promienia tarczy)
    # Są to te same wartości, które używamy do rysowania pierścieni
    bullseye_inner_r_perc = 0.035
    bullseye_outer_r_perc = 0.094
    triple_inner_r_perc = 0.57
    triple_outer_r_perc = 0.62
    double_inner_r_perc = 0.94
    double_outer_r_perc = 1.0 # Faktyczna krawędź tarczy

    if r <= radius * bullseye_inner_r_perc:
        return 50
    elif r <= radius * bullseye_outer_r_perc:
        return 25
    elif r >= radius * triple_inner_r_perc and r <= radius * triple_outer_r_perc:
        multiplier = 3
    elif r >= radius * double_inner_r_perc and r <= radius * double_outer_r_perc:
        multiplier = 2
    elif r < radius * double_inner_r_perc: # Pola pojedyncze (pomiędzy bullseye a triple, i triple a double)
        multiplier = 1
    else: # Poza tarczą
        return 0

    sector_index = int(((theta + 9) % 360) / 18)
    base_score = SCORE_MAP[sector_index]
    
    return base_score * multiplier

# --- Funkcja callback dla zdarzeń myszy w demo ---
def click_to_score(event, x, y, flags, param):
    global points, current_score, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Zaznaczono trafienie w: ({x}, {y})")

        # Oblicz wynik
        current_score = get_score(x, y, CALIBRATED_CENTER, CALIBRATED_BOARD_RADIUS)
        print(f"Wynik za to trafienie: {current_score}")

        # Odśwież obraz z nowym trafieniem i wynikiem
        img_display = original_img.copy() # Reset do oryginalnego obrazu (bez poprzednich rysunków)
        draw_all_hits_and_scores() # Narysuj wszystkie trafienia i aktualny wynik

# --- Funkcja do rysowania wszystkich trafień i wyniku ---
def draw_all_hits_and_scores():
    global img_display, current_score

    # Narysuj wszystkie poprzednie trafienia
    for p in points:
        cv2.circle(img_display, p, 7, (0, 0, 255), -1) # Czerwona kropka dla trafienia

    # Wyświetl aktualny wynik
    if current_score is not None:
        cv2.putText(img_display, f"Ostatni rzut: {current_score} pkt", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Demo Darta - Kliknij, aby rzucic", img_display)

# --- Główna część skryptu ---

original_img = cv2.imread(IMG_PATH)
if original_img is None:
    print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {IMG_PATH}")
else:
    img_display = original_img.copy()
    current_score = None # Przechowuje wynik ostatniego rzutu

    cv2.namedWindow("Demo Darta - Kliknij, aby rzucic")
    cv2.setMouseCallback("Demo Darta - Kliknij, aby rzucic", click_to_score)
    
    print("--- Demo Gry w Darta ---")
    print("Klikaj myszką na tarczę, aby symulować trafienia.")
    print("Naciśnij 'q', aby zakończyć grę.")

    draw_all_hits_and_scores() # Wyświetl początkowy obraz

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Naciśnięcie 'q' zamyka okno
            break
          
    cv2.destroyAllWindows()
    print("Dziękuję za grę!")