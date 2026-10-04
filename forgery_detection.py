from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def list_images(folder):
    extensions = {'.png', '.jpg', '.tif'}
    return sorted(
        path for path in Path(folder).glob('*')
        if path.is_file() and path.suffix.lower() in extensions
    )


def zigzag_values(block):
    """Read DCT coefficients in the original alternating diagonal order."""
    rows, columns = block.shape
    diagonals = [[] for _ in range(rows + columns - 1)]
    for row in range(rows):
        for column in range(columns):
            diagonal = row + column
            if diagonal % 2 == 0:
                diagonals[diagonal].insert(0, block[row, column])
            else:
                diagonals[diagonal].append(block[row, column])
    return np.asarray([value for diagonal in diagonals for value in diagonal], dtype=float)


def block_descriptors(gray, block_size=8, quantization=16):
    """Describe the same block origins as the original laboratory script."""
    row_count = max(0, gray.shape[0] - block_size)
    column_count = max(0, gray.shape[1] - block_size)
    descriptors = np.empty((row_count * column_count, 18))
    for row in range(row_count):
        for column in range(column_count):
            block = gray[row:row + block_size, column:column + block_size]
            coefficients = cv2.dct(np.float32(block) / 255.0)
            scaled = np.uint8(np.float32(coefficients) * 255.0)
            features = np.floor(zigzag_values(scaled)[:16] / quantization)
            descriptors[row * column_count + column] = np.append(features, [row, column])
    return descriptors


def find_similar_blocks(descriptors, similarity=5, distance=20, search_range=10):
    """Keep nearby feature vectors only when their image positions are distant."""
    matches = []
    for index in range(len(descriptors) - search_range):
        for offset in range(1, search_range):
            first, second = descriptors[index], descriptors[index + offset]
            if np.linalg.norm(first[:16] - second[:16]) > similarity:
                continue
            position1, position2 = first[-2:], second[-2:]
            if np.linalg.norm(position1 - position2) < distance:
                continue
            matches.append(np.concatenate((position1, position2, position1 - position2)))
    return np.asarray(matches, dtype=float).reshape(-1, 6)


def consistent_shift_matches(matches, vector_limit=20):
    """Reject shift vectors that occur less often than the selected threshold."""
    counts = Counter(tuple(match[4:6]) for match in matches)
    retained = [match for match in matches if counts[tuple(match[4:6])] >= vector_limit]
    return np.asarray(retained, dtype=float).reshape(-1, 6)


def forgery_mask(shape, matches, block_size=8):
    mask = np.zeros(shape, dtype=np.uint8)
    for match in matches:
        y1, x1, y2, x2 = (int(value) for value in match[:4])
        cv2.rectangle(mask, (x1, y1), (x1 + block_size, y1 + block_size), 255, -1)
        cv2.rectangle(mask, (x2, y2), (x2 + block_size, y2 + block_size), 255, -1)
    if len(matches) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def show_scaled(name, image, max_height=800):
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / height
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
    cv2.imshow(name, image)


def copy_move_forgery_detection(img_path: Path):
    img_path = img_path.resolve()
    print(f"1. Wczytywanie obrazu: {img_path.name}")
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print("2. Przetwarzanie bloków i DCT...")
    descriptors = block_descriptors(gray)
    print(f"   Przeanalizowano {len(descriptors)} bloków.")

    print("3. Sortowanie wektorów cech...")
    descriptors = descriptors[np.lexsort(np.rot90(descriptors[:, :16]))]
    print("4. Szukanie podobnych fragmentów...")
    matches = find_similar_blocks(descriptors)
    print(f"   Wstępnie znaleziono {len(matches)} par.")

    print("5. Weryfikacja spójności przesunięć...")
    final_matches = consistent_shift_matches(matches)
    print("6. Generowanie mapy fałszerstwa...")
    prediction_mask = forgery_mask(gray.shape, final_matches)
    if len(final_matches) > 0:
        print(f"WYNIK: Znaleziono {len(final_matches)} pasujących bloków.")
    else:
        print("WYNIK: Po filtracji obraz uznano za czysty.")

    show_scaled("Oryginalny Obraz", image)
    show_scaled("Wykryte Falszerstwo (Biale pola)", prediction_mask)
    print("Naciśnij dowolny klawisz, aby zamknąć...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return prediction_mask


if __name__ == "__main__":
    for img_path in list_images(Path(__file__).resolve().parent):
        print("=" * 60)
        print(f"Przetwarzam: {img_path.name}")
        copy_move_forgery_detection(img_path)
        print("Zakończono obraz:", img_path.name)
