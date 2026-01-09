import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

# Wymuszenie backendu bez GUI
matplotlib.use('Agg')

# Funkcja do wizualizacji wektorów przepływu optycznego
def draw_optical_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)
    x, y = x.flatten(), y.flatten()  # Spłaszczenie współrzędnych
    fx, fy = flow[y, x, 0], flow[y, x, 1]  # Pobranie wartości przepływu dla współrzędnych
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return vis

# Funkcja do wizualizacji przepływu optycznego jako mapy ciepła
def visualize_flow_as_heatmap(flow):
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

# Funkcja do analizy ruchu w regionach obrazu
def analyze_regions(flow, grid_size=(4, 4)):
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    h, w = magnitude.shape
    region_h, region_w = h // grid_size[0], w // grid_size[1]
    region_intensities = np.zeros(grid_size)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            region = magnitude[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            region_intensities[i, j] = np.mean(region)
    
    return region_intensities

# Ścieżka do pliku wideo
video_path = "Highway.mp4"  # Podaj tutaj ścieżkę do pliku wideo
output_folder = "optical_flow_analysis"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
    exit(1)

# Ustawienia dla wyjściowych wideo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_vectors_path = os.path.join(output_folder, "flow_vectors.avi")
output_heatmap_path = os.path.join(output_folder, "flow_heatmap.avi")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_vectors = cv2.VideoWriter(output_vectors_path, fourcc, fps, (frame_width, frame_height))
out_heatmap = cv2.VideoWriter(output_heatmap_path, fourcc, fps, (frame_width, frame_height))

ret, prev_frame = cap.read()
if not ret:
    print("Nie udało się odczytać pierwszej klatki.")
    cap.release()
    exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_idx = 0

# Do analizy intensywności ruchu w czasie
global_movement = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wyznaczanie przepływu optycznego za pomocą metody Farnebäcka
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Wizualizacja jako wektory ruchu
    flow_vectors = draw_optical_flow(gray, flow)
    out_vectors.write(flow_vectors)  # Zapis do wideo

    # Wizualizacja jako mapa ciepła
    heatmap = visualize_flow_as_heatmap(flow)
    heatmap_colored = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    out_heatmap.write(heatmap_colored)  # Zapis do wideo

    # Analiza ruchu w regionach obrazu
    region_intensities = analyze_regions(flow, grid_size=(4, 4))
    plt.imshow(region_intensities, cmap='cool', interpolation='nearest')
    plt.colorbar(label='Średnia intensywność ruchu')
    plt.title(f'Regionalna intensywność ruchu - klatka {frame_idx}')
    plt.savefig(os.path.join(output_folder, f"frame_{frame_idx:03d}_regions.png"))
    plt.close()

    # Analiza intensywności ruchu w czasie
    global_movement.append(np.mean(heatmap))

    prev_gray = gray
    frame_idx += 1

cap.release()
out_vectors.release()
out_heatmap.release()

# Wizualizacja intensywności ruchu w czasie
plt.plot(global_movement)
plt.xlabel("Numer klatki")
plt.ylabel("Średnia intensywność ruchu")
plt.title("Intensywność ruchu w czasie")
plt.savefig(os.path.join(output_folder, "global_movement_over_time.png"))
plt.close()

print(f"Przetwarzanie zakończone. Wyniki zapisano w folderze: {output_folder}")
print(f"Pliki wideo zapisane: {output_vectors_path}, {output_heatmap_path}")