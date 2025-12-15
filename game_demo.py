import cv2
import numpy as np
import math
import os
import json

# --- Konfiguracja ścieżek ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMG_PATH = os.path.join(OUTPUT_DIR, "board_rectified.png")
JSON_PATH = os.path.join(OUTPUT_DIR, "calibration_data.json")

# --- Wczytanie danych kalibracyjnych ---
if not os.path.exists(JSON_PATH):
    print(f"Błąd: Brak pliku kalibracji {JSON_PATH}")
    exit()

with open(JSON_PATH, 'r') as f:
    data = json.load(f)
    CALIB_CENTER = tuple(data["center"]) # Konwersja na krotkę
    CALIB_RADIUS = data["radius"]

print(f"Wczytano kalibrację: Środek={CALIB_CENTER}, Promień={CALIB_RADIUS:.2f}")

# --- Zmienne globalne ---
points = []
score_history = []
img_display = None
original_img = None

# Mapowanie sektorów (kąt -> punkty)
SCORE_MAP = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

# --- Logika gry ---

def get_score(x, y, center, radius):
    """Oblicza punkty na podstawie współrzędnych biegunowych."""
    dx = x - center[0]
    dy = y - center[1]
    
    r = math.sqrt(dx**2 + dy**2)
    
    # Kąt w stopniach (0-360), przesunięty o 9 stopni dla wyrównania sektorów
    theta = math.degrees(math.atan2(-dy, dx))
    if theta < 0: theta += 360

    # Definicja progów (procent promienia)
    r_norm = r / radius
    multiplier = 0
    
    if r_norm <= 0.035: return 50     # Bullseye
    if r_norm <= 0.094: return 25     # Single Bull
    
    # Sektory punktowe
    if 0.58 <= r_norm <= 0.63: multiplier = 3   # Triple
    elif 0.94 <= r_norm <= 1.0: multiplier = 2  # Double
    elif r_norm < 0.94: multiplier = 1          # Single
    else: return 0                              # Miss

    sector_idx = int(((theta + 9) % 360) / 18)
    return SCORE_MAP[sector_idx] * multiplier

def draw_interface():
    """Rysuje trafienia i interfejs."""
    global img_display
    img_display = original_img.copy()

    # Rysowanie trafień
    for p in points:
        cv2.circle(img_display, p, 5, (0, 0, 255), -1)

    # Wyświetlanie ostatniego wyniku
    if score_history:
        last_score = score_history[-1]
        cv2.putText(img_display, f"Wynik: {last_score}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Symulacja Gry", img_display)

def mouse_callback(event, x, y, flags, param):
    """Obsługa kliknięć."""
    if event == cv2.EVENT_LBUTTONDOWN:
        score = get_score(x, y, CALIB_CENTER, CALIB_RADIUS)
        points.append((x, y))
        score_history.append(score)
        
        print(f"Trafienie: ({x}, {y}) -> Punkty: {score}")
        draw_interface()

# --- Main ---

if not os.path.exists(IMG_PATH):
    print(f"Błąd: Nie znaleziono obrazu {IMG_PATH}")
    exit()

original_img = cv2.imread(IMG_PATH)
img_display = original_img.copy()

cv2.namedWindow("Symulacja Gry")
cv2.setMouseCallback("Symulacja Gry", mouse_callback)

print("--- Start Symulacji ---")
print("Kliknij na tarcze. 'q' aby wyjsc.")

draw_interface()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()