import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

# Wymuszenie backendu bez GUI
matplotlib.use('Agg')

# Funkcja do wizualizacji wektorów przepływu optycznego z podziałem na kierunki
def draw_directional_flow(img, flow, step=16):
    """
    Rysuje wektory ruchu z różnymi kolorami w zależności od kierunku.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)
    x, y = x.flatten(), y.flatten()  # Spłaszczenie współrzędnych
    fx, fy = flow[y, x, 0], flow[y, x, 1]  # Pobranie wartości przepływu dla współrzędnych
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)  # Przepływ w polarnych

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Kolory dla różnych zakresów kątów
    colors = [
        (0, 255, 0),    # Zielony (0°–45°)
        (255, 0, 0),    # Niebieski (45°–90°)
        (0, 0, 255),    # Czerwony (90°–135°)
        (255, 255, 0),  # Żółty (135°–180°)
        (255, 0, 255),  # Fioletowy (180°–225°)
        (0, 255, 255),  # Turkusowy (225°–270°)
        (128, 128, 128),# Szary (270°–315°)
        (255, 165, 0)   # Pomarańczowy (315°–360°)
    ]

    for i, (x1, y1, fx1, fy1, ang) in enumerate(zip(x, y, fx, fy, angle)):
        # Określenie koloru na podstawie kąta
        color_idx = int(ang // 45) % len(colors)
        color = colors[color_idx]

        # Rysowanie wektora
        x2, y2 = int(x1 + fx1), int(y1 + fy1)
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.3)

    return vis

# Ścieżka do pliku wideo
video_path = "Highway.mp4"  # Podaj tutaj ścieżkę do pliku wideo
output_folder = "optical_flow_analysis_directions"

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

output_directions_path = os.path.join(output_folder, "flow_directions.avi")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_directions = cv2.VideoWriter(output_directions_path, fourcc, fps, (frame_width, frame_height))

ret, prev_frame = cap.read()
if not ret:
    print("Nie udało się odczytać pierwszej klatki.")
    cap.release()
    exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wyznaczanie przepływu optycznego za pomocą metody Farnebäcka
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Wizualizacja jako wektory ruchu podzielone według kierunków
    flow_directions = draw_directional_flow(gray, flow)
    out_directions.write(flow_directions)  # Zapis do wideo

    prev_gray = gray
    frame_idx += 1

cap.release()
out_directions.release()

print(f"Przetwarzanie zakończone. Wyniki zapisano w folderze: {output_folder}")
print(f"Plik wideo zapisany: {output_directions_path}")