import numpy as np
import cv2
import os

# Tworzenie folderu na zapisane obrazy
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

BLUE = (0, 0, 255)

# ### Wczytaj obraz

image = cv2.imread("trash_images/trash1.png")
cv2.imwrite(os.path.join(output_dir, "original_image1.png"), image)

# ### Zmiana przestrzeni barw z BGR na RGB

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[150:, :]
cv2.imwrite(os.path.join(output_dir, "rgb_image1.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# ### Przekształcenie obrazu na tablicę 2D

pixel_vals = image.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

# ### Implementacja klasteryzacji K-means

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 6
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))
cv2.imwrite(os.path.join(output_dir, "segmented_image1.png"), cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

labels_reshape = labels.reshape(image.shape[0], image.shape[1])

# ### Maskowanie obrazu za pomocą klastra

cluster = 4
masked_image = np.copy(image)
masked_image[labels_reshape == cluster] = [BLUE]
cv2.imwrite(os.path.join(output_dir, "masked_image1.png"), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

# ### Wczytaj nowy obraz

image2 = cv2.imread("trash_images/trash3.jpg")
cv2.imwrite(os.path.join(output_dir, "original_image3.png"), image2)

# ### Przycinanie obrazu i przekształcenie na tablicę

image2 = image2[200:, :]
pixel_vals = image2.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

# ### Implementacja K-means na nowym obrazie

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 10
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image2.shape))
cv2.imwrite(os.path.join(output_dir, "segmented_image3.png"), cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

labels_reshape = labels.reshape(image2.shape[0], image2.shape[1])

# ### Maskowanie obrazu za pomocą klastra

cluster = 4
masked_image = np.copy(image2)
masked_image[labels_reshape == cluster] = [BLUE]
cv2.imwrite('output_images/masked_image3.jpg', masked_image)

# ### Konwersja obrazu do przestrzeni HSV

#im = cv2.imread('output_images/masked_image3.jpg')
im=masked_image
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
cv2.imwrite(os.path.join(output_dir, "hsv_image3.png"), hsv_img)

# ## Znajdź dolne i górne wartości koloru z BGR do HSV

# Debug: zapisanie obrazu HSV do analizy zakresów kolorów
cv2.imwrite(os.path.join(output_dir, "hsv_debug3.png"), hsv_img)

# Konwersja BGR do HSV i wyświetlenie odpowiadających wartości
blue = np.uint8([[[0, 0, 255]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print("HSV dla koloru czerwonego:", hsv_blue)

# ## Ustaw dowolny wybrany kolor

#lower_red = (0, 255, 255)
#upper_red = (176, 255, 255)
lower_red = (0, 120, 120)
upper_red = (10, 255, 255)

lower_yellow = (20, 80, 80)
upper_yellow = (40, 255, 255)

lower_blue = (120, 255, 150)
upper_blue = (120, 255, 255)

# ## Rysowanie prostokątnego obramowania wokół wykrytych obiektów

COLOR_MIN = np.array([lower_red], np.uint8)
COLOR_MAX = np.array([upper_red], np.uint8)
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
cv2.imwrite(os.path.join(output_dir, "frame_threshed3.png"), frame_threshed)

# Znajdź kontury na obrazie
imgray = frame_threshed
ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Zapis obrazu progowego do debugowania
cv2.imwrite(os.path.join(output_dir, "thresh_debug3.png"), thresh)

# Sprawdź, czy znaleziono kontury
areas = [cv2.contourArea(c) for c in contours]
if not areas:
    print("Nie znaleziono żadnych konturów. Sprawdź zakres koloru i obraz wejściowy.")
    exit(1)

# Znajdź indeks największego konturu
max_index = np.argmax(areas)
cnt = contours[max_index]

# Wyznacz prostokąt ograniczający (bounding box)
x, y, w, h = cv2.boundingRect(cnt)

pad_w = 4
pad_h = 5
pad_x = 3
pad_y = 4

cv2.rectangle(im, (x - pad_x, y - pad_y), (x + w + pad_w, y + h + pad_h), (255, 0, 0), 2)
cv2.imwrite(os.path.join(output_dir, "bounding_box_image3.png"), im)








import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_hsv_space(hsv_img, mask=None):
    """
    Funkcja do wizualizacji pikseli w przestrzeni HSV w 3D.
    Jeśli podana jest maska, to wyświetla tylko maskowane piksele.
    """
    # Rozmiar obrazu
    h, w, _ = hsv_img.shape

    # Rozdzielenie kanałów HSV
    H, S, V = cv2.split(hsv_img)

    # Tworzenie płaskiego widoku pikseli
    H_flat = H.flatten()
    S_flat = S.flatten()
    V_flat = V.flatten()

    # Tworzenie maski, jeśli podana
    if mask is not None:
        mask_flat = mask.flatten()
        mask_indices = mask_flat > 0  # Piksele, które przechodzą przez maskę
        H_flat = H_flat[mask_indices]
        S_flat = S_flat[mask_indices]
        V_flat = V_flat[mask_indices]

    # Rysowanie przestrzeni HSV
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Kolory reprezentujące wartości w przestrzeni HSV
    hsv_colors = np.zeros((len(H_flat), 3))
    hsv_colors[:, 0] = H_flat / 180.0  # Normalizacja H (zakres 0-180 w OpenCV)
    hsv_colors[:, 1] = S_flat / 255.0  # Normalizacja S
    hsv_colors[:, 2] = V_flat / 255.0  # Normalizacja V
    rgb_colors = cv2.cvtColor((hsv_colors * 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
    rgb_colors = rgb_colors.reshape(-1, 3) / 255.0  # Normalizacja do zakresu [0, 1]

    # Rysowanie punktów w przestrzeni 3D
    ax.scatter(H_flat, S_flat, V_flat, c=rgb_colors, marker='o', s=2)

    # Etykiety osi
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    # Tytuł wykresu
    if mask is not None:
        ax.set_title('Piksele maskowane w przestrzeni HSV')
    else:
        ax.set_title('Wszystkie piksele w przestrzeni HSV')

    # Wyświetlenie wykresu
    plt.show()

# Wywołanie funkcji dla wszystkich pikseli
plot_hsv_space(hsv_img)

# Wywołanie funkcji dla pikseli maskowanych
plot_hsv_space(hsv_img, mask=frame_threshed)