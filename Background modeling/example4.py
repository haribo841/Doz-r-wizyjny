import cv2
import numpy as np

def read_and_prepare(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    # Jeśli obraz ma 4 kanały (BGRA), usuń kanał alfa
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # Konwersja do szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

current_gray = read_and_prepare('currentFrame.jpg')
background_gray = read_and_prepare('medianFrame.jpg')

# Opcjonalne rozmycie, by zmniejszyć szum (kernel dobieraj eksperymentalnie)
blur_ksize = (5, 5)
cur_blur = cv2.GaussianBlur(current_gray, blur_ksize, 0)
bg_blur  = cv2.GaussianBlur(background_gray, blur_ksize, 0)

# Różnica bezwzględna
difference = cv2.absdiff(cur_blur, bg_blur)

# Możesz spróbować automatycznego progu (Otsu) lub stałego progu.
# 1) Stały próg (np. 25)
thresh_val = 25
_, th = cv2.threshold(difference, thresh_val, 255, cv2.THRESH_BINARY)

# 2) Alternatywnie: Otsu (dobiera próg automatycznie)
# _, th = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3) Jeżeli oświetlenie jest niejednorodne, można użyć adaptiveThreshold:
# th = cv2.adaptiveThreshold(difference, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv2.THRESH_BINARY, blockSize=11, C=2)

# Morfologia: usuń drobne szumy i wypełnij dziury
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)   # usuwa małe kropki
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)  # łączy obszary

# Opcjonalnie: odrzuć bardzo małe obiekty konturami
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 100  # dopasuj minimalny rozmiar obiektu w pikselach
mask = np.zeros_like(th)
for c in contours:
    if cv2.contourArea(c) >= min_area:
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

# Wyświetlanie
cv2.imshow("Current frame", cv2.imread('currentFrame.jpg'))
cv2.imshow("Background Model", cv2.imread('medianFrame.jpg'))
cv2.imshow("Raw difference", difference)
cv2.imshow("Thresholded (cleaned)", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
