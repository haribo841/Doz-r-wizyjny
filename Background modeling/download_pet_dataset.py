import os
import tarfile
import tensorflow as tf

# Pobranie zbioru danych
dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

image_path = tf.keras.utils.get_file("images.tar.gz", dataset_url, extract=False)
annotation_path = tf.keras.utils.get_file("annotations.tar.gz", annotations_url, extract=False)

# Rozpakowanie plików do odpowiednich lokalizacji
with tarfile.open(image_path, "r:gz") as tar:
    tar.extractall(path=os.path.dirname(image_path))
with tarfile.open(annotation_path, "r:gz") as tar:
    tar.extractall(path=os.path.dirname(annotation_path))

# Poprawione ścieżki do katalogów
base_dir = os.path.dirname(image_path)
image_dir = os.path.join(base_dir, "images")  # Bez dodatkowego podkatalogu
mask_dir = os.path.join(base_dir, "annotations", "trimaps")

# Weryfikacja, czy katalogi istnieją
assert os.path.exists(image_dir), f"Katalog {image_dir} nie istnieje! Sprawdź ścieżkę."
assert os.path.exists(mask_dir), f"Katalog {mask_dir} nie istnieje! Sprawdź ścieżkę."

print("Dane zostały pomyślnie pobrane i rozpakowane.")
