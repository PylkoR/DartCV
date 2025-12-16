import cv2
import os

# --- KONFIGURACJA ---
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
INPUT_FILE = os.path.join(PROJECT_ROOT, "pics", "dart_normal.mp4") # Twój plik wejściowy
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "dart_game_video.mp4") # Gdzie zapisać wycinek

# Czas w sekundach (np. od 2 minuty 10 sekundy do 2 minuty 30 sekundy)
START_SEC = 0   # Początek wycinka (sekundy)
END_SEC = 30     # Koniec wycinka (sekundy)

# --- SKRYPT ---
def cut_video_fragment():
    if not os.path.exists(INPUT_FILE):
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE}")
        return

    cap = cv2.VideoCapture(INPUT_FILE)
    
    # Pobranie parametrów wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Otwarto wideo: {duration:.2f}s, {width}x{height}, {fps:.2f} FPS")

    # Przeliczenie czasu na klatki
    start_frame = int(START_SEC * fps)
    end_frame = int(END_SEC * fps)

    if start_frame >= total_frames or start_frame > end_frame:
        print("Błąd: Nieprawidłowy zakres czasu.")
        return

    # Ustawienie kodera wideo (mp4v jest uniwersalny dla .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))

    # Przeskoczenie do momentu startu
    print(f"Przewijanie do {START_SEC} sekundy...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    print("Zapisywanie fragmentu...")

    while ret := cap.read()[0]:
        if current_frame > end_frame:
            break
        
        frame = cap.read()[1]
        out.write(frame)
        
        current_frame += 1
        
        # Prosty pasek postępu co 100 klatek
        if current_frame % 100 == 0:
            percent = ((current_frame - start_frame) / (end_frame - start_frame)) * 100
            print(f"Postęp: {percent:.1f}%")

    cap.release()
    out.release()
    print(f"Gotowe! Zapisano jako: {OUTPUT_FILE}")

if __name__ == "__main__":
    cut_video_fragment()