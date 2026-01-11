import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

# Instrukcja dla studentów:
# 1. Upewnij się, że masz zainstalowane odpowiednie biblioteki:
#    pip install torch torchvision torchaudio matplotlib segment-anything
# 2. Pobierz model SAM ze strony: https://github.com/facebookresearch/segment-anything
# 3. Upewnij się, że plik modelu (np. sam_vit_h.pth) jest w tej samej lokalizacji co ten skrypt.
# 4. Zastąp ścieżkę do obrazu ("image.jpg") własnym plikiem obrazu, który chcesz segmentować.

# Ścieżka do obrazu wejściowego i modelu
image_path = "car.jpg"
model_path = "sam_vit_h_4b8939.pth"#sam_vit_h.pth"

# Ładowanie modelu SAM
sam = sam_model_registry["vit_h"](checkpoint=model_path)
predictor = SamPredictor(sam)

# Wczytanie obrazu
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Ustawienie obrazu do predykcji
predictor.set_image(image_np)

# Przykładowe punkty wejściowe (na środku obrazu)
input_points = np.array([[image_np.shape[1] // 2, image_np.shape[0] // 2]])
input_labels = np.array([1])  # 1 oznacza "obiekt"

# Predykcja maski
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

# Wizualizacja wyników
plt.figure(figsize=(10, 10))

# Oryginalny obraz
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Oryginalny obraz")
plt.axis("off")

# Nałożenie maski
plt.subplot(1, 2, 2)
plt.imshow(image)
for mask in masks:
    plt.imshow(mask, alpha=0.5)  # Półprzezroczysta maska
plt.title("Segmentacja obiektu")
plt.axis("off")

# Zapisanie wyników na dysk
plt.savefig("segmentacja_wynik.png")
plt.show()

# Opcjonalnie: Wyświetlenie wyników predykcji
for i, (mask, score) in enumerate(zip(masks, scores)):
    print(f"Maska {i+1}: Wynik dopasowania: {score:.2f}")