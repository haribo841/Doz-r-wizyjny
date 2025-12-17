import numpy as np
import cv2

# Open Video
cap = cv2.VideoCapture(r'J:\My Drive\StudiaDokumenty\Mgr\2semestr\Dozór wizyjny\Laboratorium\Lab1 - Modelowanie tła\highway.mp4')

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć pliku wideo. Sprawdź ścieżkę i uprawnienia.")

# znajdź ile jest klatek w filmie
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# wybieramy max do 100, ale nie przekraczamy total_frames-1
last = min(100, max(0, total_frames - 1))

# Prostsze tworzenie listy klatek (0..last)
frameIds = list(range(0, last + 1))

# Alokujemy listę na rzeczywiste obrazy (bez None)
frames = []

for fid in frameIds:
    # przesuwamy wskaźnik do konkretnej klatki
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()

    if not ret or frame is None:
        # informacja diagnostyczna i pominięcie tej klatki
        print(f"Warning: nie udało się wczytać klatki {fid}, pomijam.")
        continue

    frames.append(frame)

cap.release()  # zwolnij wcześniej, bo dalej nie potrzebujemy VideoCapture

if len(frames) == 0:
    raise RuntimeError("Brak poprawnie wczytanych klatek — nic do uśredniania/medianowania.")

# Konwersja listy klatek do tablicy numpy: (N, H, W, C)
stack = np.stack(frames, axis=0)

# --- ŚREDNIA ---
# Obliczenia na typie zmiennoprzecinkowym
mean_float = np.mean(stack.astype(np.float32), axis=0)

# Zaokrąglenie i konwersja do uint8
meanFrame = np.rint(mean_float).clip(0, 255).astype(np.uint8)

# --- MEDIANA ---
# Mediana liczona po osi czasu (axis=0)
median_float = np.median(stack, axis=0)

# Zaokrąglenie i konwersja do uint8
medianFrame = np.rint(median_float).clip(0, 255).astype(np.uint8)


# Wyświetlanie / zapis
cv2.imshow('median frame', medianFrame)
cv2.imwrite('medianFrame.jpg', medianFrame, [cv2.IMWRITE_JPEG_QUALITY, 100])

cv2.imshow('mean frame', meanFrame)
cv2.imwrite('meanFrame.jpg', meanFrame, [cv2.IMWRITE_JPEG_QUALITY, 100])

cv2.waitKey(0)
cv2.destroyAllWindows()
