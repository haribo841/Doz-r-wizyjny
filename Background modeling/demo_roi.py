import cv2
import numpy as np
import os

# Utworzenie folderu na wyniki
output_dir = "roi_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Wczytanie obrazu
image_path = "car.jpg"  # Podmień na ścieżkę do swojego obrazu
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")


# ================= ROI na podstawie progowania =================
# Konwersja obrazu na skalę szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Progowanie za pomocą metody Otsu
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY) # Progowanie globalne ze stałym progiem
cv2.imwrite(os.path.join(output_dir, "thresholded_image.jpg"), thresh)

# Wyznaczenie ROI jako największego obszaru w obrazie binarnym
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



largest_contour = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest_contour)

# Rysowanie bounding box na oryginalnym obrazie
image_thresh_bbox = image.copy()
cv2.rectangle(image_thresh_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir, "roi_thresh_bbox.jpg"), image_thresh_bbox)

roi_thresh = image[y:y+h, x:x+w]
cv2.imwrite(os.path.join(output_dir, "roi_threshold.jpg"), roi_thresh)

# ================= ROI na podstawie konturów =================
# Wykrywanie krawędzi za pomocą operatora Canny'ego
# edges = cv2.Canny(gray, 50, 150)
edges = cv2.Canny(gray, 30, 100)# Zmniejszone progi - większa czułość
cv2.imwrite(os.path.join(output_dir, "edges_canny.jpg"), edges)

# Detekcja konturów
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Rysowanie konturów na obrazie
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir, "contours.jpg"), image_contours)

# Wyznaczenie ROI dla największego konturu
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Rysowanie bounding box na oryginalnym obrazie
image_contours_bbox = image.copy()
cv2.rectangle(image_contours_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir, "roi_contours_bbox.jpg"), image_contours_bbox)

roi_contours = image[y:y+h, x:x+w]
cv2.imwrite(os.path.join(output_dir, "roi_contours.jpg"), roi_contours)

#

# ================= Podsumowanie =================
print("Zadanie ukończone. Wyniki zapisano w folderze 'roi_results'.")