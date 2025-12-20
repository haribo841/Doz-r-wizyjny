import cv2
import numpy as np
import imutils
from collections import OrderedDict
import math

# Klasa do śledzenia obiektów
class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # Inicjalizacja następnego unikalnego ID obiektu
        self.nextObjectID = 0
        # Słownik przechowujący aktualnie śledzone obiekty
        self.objects = OrderedDict()
        # Liczba kolejnych klatek, w których obiekt nie został wykryty
        self.disappeared = OrderedDict()
        # Maksymalna liczba klatek, w których obiekt może nie zostać wykryty
        self.maxDisappeared = maxDisappeared
        # Maksymalna odległość pomiędzy centroidami, aby uznać je za ten sam obiekt
        self.maxDistance = maxDistance

    def register(self, centroid):
        # Rejestracja nowego obiektu
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID +=1

    def deregister(self, objectID):
        # Usunięcie obiektu ze śledzenia
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        # Jeśli nie ma wykrytych obiektów
        if len(inputCentroids) == 0:
            # Zwiększ licznik zniknięć dla każdego śledzonego obiektu
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] +=1
                # Jeśli obiekt przekroczył maksymalną liczbę zniknięć, usuń go
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Jeśli nie śledzimy żadnych obiektów, zarejestruj wszystkie centroidy
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Lista istniejących ID obiektów i ich centroidów
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Oblicz macierz odległości pomiędzy starymi a nowymi centroidami
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(inputCentroids), axis=2)

            # Znajdź minimalne wartości w macierzy odległości
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Używane do śledzenia już przetworzonych wierszy i kolumn
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # Jeśli wiersz lub kolumna została już użyta, pomiń
                if row in usedRows or col in usedCols:
                    continue

                # Jeśli odległość jest większa niż maksymalna, pomiń
                if D[row, col] > self.maxDistance:
                    continue

                # W przeciwnym razie zaktualizuj centroid obiektu
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # Ustalenie niewykorzystanych wierszy i kolumn
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Jeśli liczba centroidów starych jest większa lub równa liczbie nowych
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] +=1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Inicjalizacja liczników
count_in = 0
count_out = 0
area_count = 0  # Dodany licznik dla obszaru

# Definiowanie współrzędnych linii zliczania
line_coord = [(100, 200), (500, 200)]  # Przykładowe współrzędne

# Definiowanie współrzędnych obszaru zliczania (prostokąt)
area_pts = np.array([[300, 150], [500, 150], [500, 350], [300, 350]])

# Inicjalizacja substraktora tła
backSub = cv2.createBackgroundSubtractorMOG2()

# Inicjalizacja wideo
cap = cv2.VideoCapture('people3.mp4')

# Inicjalizacja śledzenia
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    H, W = frame.shape[:2]

    # Odejmowanie tła
    fgMask = backSub.apply(frame)

    # Usunięcie cieni
    _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)

    # Operacje morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Znajdowanie konturów
    contours, _ = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    for c in contours:
        # Ignorowanie małych konturów
        if cv2.contourArea(c) < 500:
            continue

        # Obliczanie prostokąta otaczającego
        (x, y, w, h) = cv2.boundingRect(c)
        rects.append((x, y, x + w, y + h))

    # Aktualizacja śledzenia obiektów
    inputCentroids = []
    for (startX, startY, endX, endY) in rects:
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroids.append((cX, cY))

    objects = ct.update(inputCentroids)

    # Pętla przez śledzone obiekty
    for (objectID, centroid) in objects.items():
        # Jeśli obiekt nie jest śledzony, zainicjuj go
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = {"centroids": [centroid], "counted_line": False, "counted_area": False}
        else:
            to["centroids"].append(centroid)
            # Sprawdzenie, czy obiekt został już policzony przy linii
            if not to["counted_line"]:
                # Sprawdzenie kierunku ruchu
                y = [c[1] for c in to["centroids"]]
                direction = centroid[1] - np.mean(y)

                # Sprawdzenie, czy obiekt przekroczył linię
                if centroid[1] < line_coord[0][1] and direction < 0:
                    count_in +=1
                    to["counted_line"] = True
                elif centroid[1] > line_coord[0][1] and direction > 0:
                    count_out +=1
                    to["counted_line"] = True

            # Sprawdzenie, czy obiekt został już policzony w obszarze
            if not to["counted_area"]:
                if cv2.pointPolygonTest(area_pts, (centroid[0], centroid[1]), False) >= 0:
                    area_count += 1
                    to["counted_area"] = True

        # Zapisanie obiektu
        trackableObjects[objectID] = to

        # Rysowanie centroidu i ID obiektu
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Rysowanie linii zliczania
    cv2.line(frame, line_coord[0], line_coord[1], (255, 0, 0), 2)

    # Rysowanie obszaru zliczania
    cv2.polylines(frame, [area_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Wyświetlanie liczników
    cv2.putText(frame, f'Count In: {count_in}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Count Out: {count_out}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Area Count: {area_count}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()