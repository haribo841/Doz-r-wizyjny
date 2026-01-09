import cv2
import numpy as np
import os
from ultralytics import YOLO

# Utworzenie folderu na wyniki
output_dir = "roi_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Wczytanie obrazu
image_path = "car.png"  # Podmień na ścieżkę do swojego obrazu
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

# Zapisanie oryginalnego obrazu
cv2.imwrite(os.path.join(output_dir, "original_image.jpg"), image)





# ================= ROI przy użyciu modelu głębokiego uczenia (YOLO) =================
def yolo_detection(image_path, output_dir, model_path="yolov8x.pt"):
    from ultralytics import YOLO

    # Wczytanie modelu YOLO
    model = YOLO(model_path)

    # Detekcja obiektów
    results = model(image_path)

    # Iteracja przez wyniki detekcji
    for i, result in enumerate(results):
        # Tworzenie obrazu z naniesionymi ramkami
        annotated_image = result.plot()
        
        # Zapisanie obrazu
        output_file = os.path.join(output_dir, f"annotated_image_{i+1}.jpg")
        cv2.imwrite(output_file, annotated_image)
        print(f"Zapisano: {output_file}")

yolo_detection(image_path, output_dir)

# ================= Podsumowanie =================
print("Zadanie ukończone. Wyniki zapisano w folderze 'roi_results'.")