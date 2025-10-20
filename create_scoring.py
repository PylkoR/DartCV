import cv2
import numpy as np
import math

# --- Zmienne globalne do przechowywania kliknięć ---
points = []
step = 0

# --- Funkcja callback dla zdarzeń myszy ---
def select_points_for_grid(event, x, y, flags, param):
    global points, step
    
    # Dodaj punkt po kliknięciu lewym przyciskiem myszy
    if event == cv2.EVENT_LBUTTONDOWN and step < 3:
        points.append((x, y))
        step += 1
        
        # Wizualizacja kliknięć
        if step == 1:
            print("Krok 1/3: Zaznaczono środek. Teraz zaznacz pierwszy punkt na krawędzi pola 'double'.")
            cv2.circle(img_clone, (x, y), 5, (0, 255, 0), -1) # Zielony środek
            cv2.imshow("Kalibracja siatki", img_clone)
        elif step == 2:
            print("Krok 2/3: Zaznaczono pierwszą krawędź. Teraz zaznacz drugi, inny punkt na krawędzi pola 'double'.")
            cv2.circle(img_clone, (x, y), 5, (0, 0, 255), -1) # Czerwona krawędź
            cv2.line(img_clone, points[0], points[1], (255, 255, 0), 2) # Linia pierwszego promienia
            cv2.imshow("Kalibracja siatki", img_clone)
        elif step == 3:
            print("Krok 3/3: Zaznaczono drugą krawędź. Obliczanie siatki...")
            cv2.circle(img_clone, (x, y), 5, (0, 0, 255), -1) # Czerwona krawędź
            cv2.line(img_clone, points[0], points[2], (255, 255, 0), 2) # Linia drugiego promienia
            cv2.imshow("Kalibracja siatki", img_clone)

# --- Główna część skryptu ---

# Ścieżka do Twojego wyprostowanego obrazu tarczy
IMG_PATH = "pics/board_corrected.png"  

img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {IMG_PATH}")
else:
    img_clone = img.copy()
    cv2.namedWindow("Kalibracja siatki")
    cv2.setMouseCallback("Kalibracja siatki", select_points_for_grid)
    
    print("Krok 1/3: Kliknij w sam środek tarczy (bullseye).")
    cv2.imshow("Kalibracja siatki", img_clone)

    # Czekaj, aż użytkownik zaznaczy 3 punkty
    while step < 3:
        cv2.waitKey(20)

    # --- Po zaznaczeniu punktów, oblicz i narysuj siatkę ---
    
    # 1. Pobierz punkty
    center = points[0]
    edge_point1 = points[1]
    edge_point2 = points[2]
    
    # 2. Oblicz dwa promienie
    radius1 = math.sqrt((edge_point1[0] - center[0])**2 + (edge_point1[1] - center[1])**2)
    radius2 = math.sqrt((edge_point2[0] - center[0])**2 + (edge_point2[1] - center[1])**2)
    
    # 3. Oblicz średni promień
    board_radius = (radius1 + radius2) / 2.0
    
    print(f"\nKalibracja zakończona!")
    print(f"Ustawiony środek: {center}")
    print(f"Promień 1: {radius1:.2f}, Promień 2: {radius2:.2f}")
    print(f"Uśredniony promień tarczy: {board_radius:.2f} pikseli")

    # 4. Rysuj pierścienie na obrazie na podstawie średniego promienia
    cv2.circle(img, center, int(board_radius * 0.035), (196, 6, 34), 2)
    cv2.circle(img, center, int(board_radius * 0.094), (196, 6, 34), 2)
    cv2.circle(img, center, int(board_radius * 0.58), (0, 255, 255), 2)
    cv2.circle(img, center, int(board_radius * 0.63), (0, 255, 255), 2)
    cv2.circle(img, center, int(board_radius * 0.94), (0, 255, 255), 2)
    cv2.circle(img, center, int(board_radius), (0, 255, 255), 2)

    # 5. Rysuj linie sektorów - ZMIANA WPROWADZONA TUTAJ
    # Definiujemy promień startowy (krawędź "single bull")
    start_radius = board_radius * 0.094
    for i in range(20):
        angle = i * 18 - 9 
        rad_angle = math.radians(angle)
        
        # Oblicz punkt startowy na wewnętrznym okręgu
        x_start = int(center[0] + start_radius * math.cos(rad_angle))
        y_start = int(center[1] - start_radius * math.sin(rad_angle))
        
        # Oblicz punkt końcowy na zewnętrznym okręgu
        x_end = int(center[0] + board_radius * math.cos(rad_angle))
        y_end = int(center[1] - board_radius * math.sin(rad_angle))
        
        # Narysuj linię między nowymi punktami
        cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)

    # 6. Wyświetl finalny obraz
    cv2.imshow("Tarcza z dopasowaną siatką (uśrednioną)", img)
    print("\nGotowe. Naciśnij dowolny klawisz, aby zamknąć.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()