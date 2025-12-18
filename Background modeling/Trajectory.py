import cv2
import numpy as np
import math

# 1. KONFIGURACJA
video_source = 'pixabay1.mp4'
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
    exit()

# 2. INICJALIZACJA MODELU TŁA
# Zgodnie z instrukcją używamy MOG2 z detekcją cieni 
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Zmienne do śledzenia obiektów (aby rysować osobne linie dla każdego auta)
# Słownik: {id_obiektu: [lista_punktów_trajektorii]}
object_paths = {}
# Słownik: {id_obiektu: (B, G, R)} - kolory
object_colors = {}
next_object_id = 0

# Puste płótno do rysowania trajektorii (aby nie znikały w nowej klatce)
trajectory_overlay = None

print("Przetwarzanie wideo... Naciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inicjalizacja płótna trajektorii przy pierwszej klatce
    if trajectory_overlay is None:
        trajectory_overlay = np.zeros_like(frame)

    # 3. PRZETWARZANIE OBRAZU I MASKA
    # Wyodrębnienie maski
    fgMask = backSub.apply(frame)
    
    # Progowanie (Threshold) - usunięcie cieni (wartości szare)
    _, fgMask_no_shadows = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    
    # Operacje morfologiczne - usunięcie szumów 
    fgMask_no_shadows = cv2.morphologyEx(fgMask_no_shadows, cv2.MORPH_OPEN, kernel)
    fgMask_no_shadows = cv2.morphologyEx(fgMask_no_shadows, cv2.MORPH_CLOSE, kernel)

    # 4. DETEKCJA KONTURÓW I ŚRODKÓW CIĘŻKOŚCI
    # Znajdowanie konturów
    contours, _ = cv2.findContours(fgMask_no_shadows, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_frame_centroids = []

    for contour in contours:
        # Filtracja małych obszarów (szumu)
        if cv2.contourArea(contour) > 500:
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                # Obliczenie środka masy (CX, CY)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                current_frame_centroids.append((cX, cY))
                
                # Rysowanie kropki w aktualnym położeniu
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

    # 5. ŚLEDZENIE (PRZYPISYWANIE PUNKTÓW DO OBIEKTÓW)
    # Prosty algorytm dystansowy, aby spełnić wymóg "różne kolory dla różnych obiektów"
    # Jeśli nowy punkt jest blisko ostatniego punktu znanej ścieżki -> dopisz go.
    
    active_ids = [] # Lista ID wykrytych w tej klatce
    
    for cX, cY in current_frame_centroids:
        matched_id = None
        min_dist = 50.0 # Maksymalna odległość w pikselach między klatkami
        
        # Szukamy, czy ten punkt pasuje do istniejącej ścieżki
        for obj_id, points in object_paths.items():
            if not points: continue
            last_point = points[-1]
            dist = math.hypot(cX - last_point[0], cY - last_point[1])
            
            if dist < min_dist:
                min_dist = dist
                matched_id = obj_id
        
        if matched_id is not None:
            # Znaleziono pasujący obiekt - aktualizujemy ścieżkę
            object_paths[matched_id].append((cX, cY))
            active_ids.append(matched_id)
        else:
            # Nie znaleziono - tworzymy nowy obiekt
            object_paths[next_object_id] = [(cX, cY)]
            # Losowy kolor dla nowego obiektu 
            object_colors[next_object_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            active_ids.append(next_object_id)
            next_object_id += 1

    # 6. RYSOWANIE TRAJEKTORII
    # Rysujemy linie na warstwie 'trajectory_overlay'
    for obj_id in active_ids:
        points = object_paths[obj_id]
        if len(points) > 1:
            # Rysowanie linii łączącej ostatnie dwa punkty
            # Używamy koloru przypisanego do obiektu
            cv2.line(trajectory_overlay, points[-2], points[-1], object_colors[obj_id], 2)

    # Łączenie obrazu z kamery z narysowanymi trajektoriami
    final_image = cv2.add(frame, trajectory_overlay)

    cv2.imshow('Frame with Trajectories', final_image)

    # Zapis klatka po klatce (opcjonalnie, aby widzieć postęp) lub wyjście
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 7. ZAPIS I CZYSZCZENIE
# Zapisanie ostatecznej mapy trajektorii 
# Łączymy overlay z ostatnią klatką lub czarnym tłem, w zależności od preferencji.
# Tutaj zapisujemy na tle ostatniej klatki wideo.
cv2.imwrite('final_trajectory_map.png', final_image)
print("Zapisano obraz: final_trajectory_map.png")

cap.release()
cv2.destroyAllWindows()