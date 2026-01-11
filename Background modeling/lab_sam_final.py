import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import os

# --- KONFIGURACJA (Punkty 1-4) ---
image_path = "car.jpg"         # Obraz
model_path = "sam_vit_h_4b8939.pth"   # Model pobrany z internetu

# Sprawdzenie czy pliki istnieją (dla bezpieczeństwa)
if not os.path.exists(image_path):
    print(f"BŁĄD: Nie znaleziono pliku {image_path}")
    exit()
if not os.path.exists(model_path):
    print(f"BŁĄD: Nie znaleziono modelu {model_path}. Pobierz go najpierw!")
    exit()

print("1. Ładowanie modelu SAM (może to chwilę potrwać)...")
sam = sam_model_registry["vit_h"](checkpoint=model_path)
predictor = SamPredictor(sam)

print(f"2. Wczytywanie obrazu: {image_path}")
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Ustawienie obrazu do predykcji
predictor.set_image(image_np)

# Punkt centralny (dla car.jpg środek to samochód)
input_points = np.array([[image_np.shape[1] // 2, image_np.shape[0] // 2]])
input_labels = np.array([1])  # 1 = obiekt

print("3. Generowanie maski...")
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

# Wybór najlepszej maski
best_mask_idx = np.argmax(scores)
best_mask = masks[best_mask_idx]

# --- PUNKT 5: WYNIKI (Zapis i Wyświetlanie) ---

print("4. Przygotowanie wizualizacji...")
plt.figure(figsize=(10, 5))

# Panel lewy: Oryginał
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Oryginalny obraz")
plt.axis("off")

# Panel prawy: Wynik z maską
plt.subplot(1, 2, 2)
plt.imshow(image)

# Tworzenie nakładki (overlay)
# Używamy prostego sposobu wyświetlania maski
show_mask = np.zeros((*best_mask.shape, 4))
show_mask[best_mask, 3] = 0.5                  # Przezroczystość (Alpha)
show_mask[best_mask, :3] = [30/255, 144/255, 255/255] # Kolor niebieski

plt.imshow(show_mask)
plt.title(f"Segmentacja (Pewność: {scores[best_mask_idx]:.2f})")
plt.axis("off")

# ZAPISANIE WYNIKU (Kluczowy element punktu 5)
output_filename = "segmentacja_wynik.png"
plt.savefig(output_filename)
print(f"--> SUKCES: Wynikowy obraz zapisano jako: {output_filename}")

# WYŚWIETLENIE OKNA (Kluczowy element punktu 5)
# Program zatrzyma się w tym miejscu do momentu zamknięcia okna przez użytkownika
print("--> Wyświetlanie okna graficznego...")
plt.show()