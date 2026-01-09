import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import csv

# Wymuszenie backendu bez GUI
matplotlib.use('Agg')

# Funkcja do analizy ruchu w regionach obrazu
def analyze_regions_with_metrics(flow, grid_size=(4, 4)):
    """
    Analizuje intensywność i kierunki ruchu w podziałach regionów obrazu.
    :param flow: Przepływ optyczny (wektorowy).
    :param grid_size: Liczba regionów (rząd x kolumny).
    :return: Metryki dla regionów: intensywność, dominujący kierunek, rozkład kierunków.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    h, w = magnitude.shape
    region_h, region_w = h // grid_size[0], w // grid_size[1]
    
    metrics = []
    
    for i in range(grid_size[0]):
        row_metrics = []
        for j in range(grid_size[1]):
            region_magnitude = magnitude[i * region_h:(i + 1) * region_h, j * region_w:(j + 1) * region_w]
            region_angle = angle[i * region_h:(i + 1) * region_h, j * region_w:(j + 1) * region_w]
            
            # Średnia intensywność
            avg_intensity = np.mean(region_magnitude)
            
            # Dominujący kierunek (moda w histogramie kątów)
            hist, bins = np.histogram(region_angle, bins=np.linspace(0, 360, 9))  # 8 przedziałów co 45°
            dominant_direction = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
            
            # Rozkład liczby wektorów w przedziałach kierunków
            distribution = hist / hist.sum()  # Procentowy rozkład
            
            row_metrics.append((avg_intensity, dominant_direction, distribution))
        metrics.append(row_metrics)
    
    return metrics

# Ścieżka do pliku wideo
video_path = "Highway.mp4"  # Podaj tutaj ścieżkę do pliku wideo
output_folder = "optical_flow_analysis_metrics"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
    exit(1)

ret, prev_frame = cap.read()
if not ret:
    print("Nie udało się odczytać pierwszej klatki.")
    cap.release()
    exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_idx = 0

# Plik CSV do zapisu wyników
csv_file = os.path.join(output_folder, "region_metrics.csv")
csv_header = ["Frame", "Region_Row", "Region_Col", "Avg_Intensity", "Dominant_Direction", "Direction_Distribution"]

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Wyznaczanie przepływu optycznego za pomocą metody Farnebäcka
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Analiza ruchu w regionach obrazu
        region_metrics = analyze_regions_with_metrics(flow, grid_size=(4, 4))

        for i, row_metrics in enumerate(region_metrics):
            for j, (avg_intensity, dominant_direction, distribution) in enumerate(row_metrics):
                writer.writerow([frame_idx, i, j, avg_intensity, dominant_direction] + list(distribution))

        prev_gray = gray
        frame_idx += 1

cap.release()

# Wizualizacja intensywności w regionach dla ostatniej klatki
if region_metrics:
    avg_intensity_map = np.array([[m[0] for m in row] for row in region_metrics])
    plt.imshow(avg_intensity_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Średnia intensywność ruchu')
    plt.title(f"Intensywność ruchu w regionach (klatka {frame_idx - 1})")
    plt.savefig(os.path.join(output_folder, f"frame_{frame_idx - 1}_intensity_map.png"))
    plt.close()

print(f"Przetwarzanie zakończone. Wyniki zapisano w folderze: {output_folder}")
print(f"Metryki zapisano w pliku CSV: {csv_file}")