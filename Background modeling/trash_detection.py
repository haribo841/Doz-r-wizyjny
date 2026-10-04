from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


BLUE = (0, 0, 255)
LOWER_RED = (0, 120, 120)
UPPER_RED = (10, 255, 255)


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {path}")
    return image


def segment_image(image, clusters, initialization):
    """Run the original K-means configuration and mask cluster number four."""
    pixels = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 10, initialization)
    segmented = np.uint8(centers)[labels.flatten()].reshape(image.shape)
    masked = image.copy()
    masked[labels.reshape(image.shape[:2]) == 4] = BLUE
    return segmented, masked


def red_color_mask(image):
    """Threshold the original red HSV range, including empty foreground masks."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_min = np.array([LOWER_RED], np.uint8)
    color_max = np.array([UPPER_RED], np.uint8)
    mask = cv2.inRange(hsv_image, color_min, color_max)
    _, threshold = cv2.threshold(mask, 127, 255, 0)
    return hsv_image, mask, threshold


def largest_object_bounds(threshold):
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Nie znaleziono żadnych konturów. Sprawdź zakres koloru i obraz wejściowy.")
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def largest_red_object(image):
    """Threshold the original red HSV range and return its largest contour."""
    hsv_image, mask, threshold = red_color_mask(image)
    return hsv_image, mask, threshold, largest_object_bounds(threshold)


def hsv_points(hsv_image, mask=None):
    """Return flattened HSV channels, optionally restricted to nonzero mask pixels."""
    hue, saturation, value = (channel.flatten() for channel in cv2.split(hsv_image))
    if mask is not None:
        selected = mask.flatten() > 0
        hue, saturation, value = hue[selected], saturation[selected], value[selected]
    return hue, saturation, value


def plot_hsv_space(hsv_image, mask=None):
    hue, saturation, value = hsv_points(hsv_image, mask)
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection='3d')
    hsv_colors = np.zeros((len(hue), 3))
    hsv_colors[:, 0] = hue / 180.0
    hsv_colors[:, 1] = saturation / 255.0
    hsv_colors[:, 2] = value / 255.0
    rgb_colors = cv2.cvtColor(
        (hsv_colors * 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_HSV2RGB
    ).reshape(-1, 3) / 255.0
    axes.scatter(hue, saturation, value, c=rgb_colors, marker='o', s=2)
    axes.set_xlabel('Hue')
    axes.set_ylabel('Saturation')
    axes.set_zlabel('Value')
    title = 'Piksele maskowane w przestrzeni HSV' if mask is not None else 'Wszystkie piksele w przestrzeni HSV'
    axes.set_title(title)
    plt.show()


def save_first_example(output_dir):
    image = read_image("trash_images/trash1.png")
    cv2.imwrite(str(output_dir / "original_image1.png"), image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[150:, :]
    cv2.imwrite(str(output_dir / "rgb_image1.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    segmented, masked = segment_image(image, 6, cv2.KMEANS_RANDOM_CENTERS)
    cv2.imwrite(str(output_dir / "segmented_image1.png"), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "masked_image1.png"), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))


def save_second_example(output_dir):
    image = read_image("trash_images/trash3.jpg")
    cv2.imwrite(str(output_dir / "original_image3.png"), image)
    image = image[200:, :]
    segmented, masked = segment_image(image, 10, cv2.KMEANS_PP_CENTERS)
    cv2.imwrite(str(output_dir / "segmented_image3.png"), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "masked_image3.jpg"), masked)
    hsv_image, mask, threshold = red_color_mask(masked)
    cv2.imwrite(str(output_dir / "hsv_image3.png"), hsv_image)
    cv2.imwrite(str(output_dir / "hsv_debug3.png"), hsv_image)
    cv2.imwrite(str(output_dir / "frame_threshed3.png"), mask)
    cv2.imwrite(str(output_dir / "thresh_debug3.png"), threshold)
    red = np.uint8([[[0, 0, 255]]])
    print("HSV dla koloru czerwonego:", cv2.cvtColor(red, cv2.COLOR_BGR2HSV))
    x, y, width, height = largest_object_bounds(threshold)
    cv2.rectangle(masked, (x - 3, y - 4), (x + width + 4, y + height + 5), (255, 0, 0), 2)
    cv2.imwrite(str(output_dir / "bounding_box_image3.png"), masked)
    return hsv_image, mask


def main():
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)
    save_first_example(output_dir)
    try:
        hsv_image, mask = save_second_example(output_dir)
    except ValueError as error:
        print(error)
        return 1
    plot_hsv_space(hsv_image)
    plot_hsv_space(hsv_image, mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
