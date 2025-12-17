import cv2 as cv
import numpy as np

# --- Wybierz algorytm i jego parametry ---
use_mog2 = True  # False -> użyje KNN

if use_mog2:
    # przykład ustawień: history, varThreshold, detectShadows
    backSub = cv.createBackgroundSubtractorMOG2(history=120, varThreshold=50, detectShadows=True)
else:
    # KNN: history, dist2Threshold, detectShadows
    backSub = cv.createBackgroundSubtractorKNN(history=120, dist2Threshold=400.0, detectShadows=True)

# ścieżka do pliku wideo
cap = cv.VideoCapture(r'J:\My Drive\StudiaDokumenty\Mgr\2semestr\Dozór wizyjny\Laboratorium\Lab1 - Modelowanie tła\highway.mp4')
if not cap.isOpened():
    print('Unable to open file')
    exit(0)

# struktury morfologiczne
kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # opcjonalnie: redukcja szumu w obrazie wejściowym (pomaga przy bardzo zaszumionych nagraniach)
    # frame_blur = cv.GaussianBlur(frame, (5,5), 0)
    # fgMask = backSub.apply(frame_blur)
    fgMask = backSub.apply(frame, learningRate=-1)  # learningRate=-1 -> automatyczny

    # --- Mask post-processing ---
    # 1) jeśli detectShadows=True to cienie zazwyczaj mają wartość 127 w masce (MOG2/KNN)
    #    -> jeżeli chcemy je odrzucić, zamieniamy 127 na 0 albo progowanie
    fgMask[fgMask == 127] = 0

    # 2) usunięcie drobnych zakłóceń: median + morfologia
    fgMask = cv.medianBlur(fgMask, 5)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel_small, iterations=1)  # usuwa szum
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel_close, iterations=1)  # łączy obszary

    # 3) progowanie (pewność, że mamy binarną maskę 0/255)
    _, fgBin = cv.threshold(fgMask, 127, 255, cv.THRESH_BINARY)

    # --- (opcjonalnie) znajdź kontury i rysuj bounding boxy ---
    contours, _ = cv.findContours(fgBin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = 500  # filtr na minimalny rozmiar obiektu (dostosuj)
    vis = frame.copy()
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # wyświetlanie
    cv.imshow('Frame', vis)
    cv.imshow('FG Mask (raw)', fgMask)
    cv.imshow('FG Mask (binary)', fgBin)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
