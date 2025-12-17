import cv2
import numpy as np

# 1. KONFIGURACJA I INICJALIZACJA
# Ścieżka do pliku wideo
video_source = 'highway.mp4'
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print('Nie można otworzyć pliku wideo')
    exit(0)

# Inicjalizacja algorytmu usuwania tła (MOG2) - zgodnie z wskazówką 
# detectShadows=True pozwala lepiej wyodrębnić same obiekty, ignorując ich cienie
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Pobranie wymiarów klatki wideo
ret, frame = cap.read()
if not ret:
    print("Błąd odczytu pierwszej klatki")
    exit(0)
height, width = frame.shape[:2]

# Inicjalizacja macierzy akumulacyjnej dla mapy ciepła (float32 dla precyzji)
# Wymiary takie same jak klatka wideo, wypełniona zerami
accum_image = np.zeros((height, width), dtype=np.float32)

print("Rozpoczynanie przetwarzania... Naciśnij 'q', aby przerwać.")

# 2. GŁÓWNA PĘTLA PRZETWARZANIA
while True:
    ret, frame = cap.read()
    
    if not ret:
        break # Koniec wideo

    # Krok 1: Wyodrębnienie maski pierwszego planu (ruchomych obiektów) 
    fgMask = backSub.apply(frame)

    # Krok 2: Usuwanie szumów (operacje morfologiczne) 
    # Używamy binaryzacji, aby pozbyć się cieni (które MOG2 oznacza szarym kolorem)
    _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
    
    # Opcjonalnie: Operacja otwarcia (erozja + dylatacja) usuwa drobne kropki (szum)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    # Krok 3: Aktualizacja mapy ciepła
    # Dodajemy wartość tam, gdzie wykryto ruch.
    # Konwersja maski na float (0.0 lub 1.0) i dodanie do akumulatora
    accum_image += fgMask.astype(np.float32) / 255.0

    # Krok 4: Wizualizacja w czasie rzeczywistym
    # Normalizacja wartości akumulatora do zakresu 0-255, aby wyświetlić jako obraz
    heatmap_norm = cv2.normalize(accum_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_norm = heatmap_norm.astype(np.uint8)

    # Nałożenie mapy kolorów (COLORMAP_JET to klasyczna mapa: niebieski=zimny, czerwony=gorący) 
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Opcjonalnie: Nakładanie mapy ciepła na oryginalny obraz (dla lepszego kontekstu)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Wyświetlanie okien
    cv2.imshow('Oryginalny obraz', frame)
    cv2.imshow('Maska ruchu (MOG2 + Morfologia)', fgMask)
    cv2.imshow('Mapa Ciepla (Czas rzeczywisty)', heatmap_color)
    cv2.imshow('Nakladka (Overlay)', overlay)

    # Obsługa wyjścia klawiszem 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# 3. ZAPIS I CZYSZCZENIE
# Zapisanie ostatecznej mapy ciepła jako plik PNG 
cv2.imwrite('heatmap_final.png', heatmap_color)
cv2.imwrite('heatmap_overlay_final.png', overlay) # Dodatkowy zapis z widokiem drogi

print("Przetwarzanie zakończone. Mapa ciepła zapisana jako 'heatmap_final.png'.")

cap.release()
cv2.destroyAllWindows()