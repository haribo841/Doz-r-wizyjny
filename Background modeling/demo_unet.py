import os
import numpy as np
import tensorflow as tf  # Główny import naprawiający problemy w Visual Studio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# Zmienione importy - teraz linter w Visual Studio poprawnie zaindeksuje klasy
# Korzystamy bezpośrednio z aliasu tf.keras
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
concatenate = tf.keras.layers.concatenate
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array

# **Budowa sieci U-Net dla wieloklasowej segmentacji**
def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder (Ścieżka kompresji)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder (Ścieżka ekspansji / dekonwolucji)
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    # Output layer - wieloklasowa segmentacja (num_classes)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# **Funkcja do wczytywania danych**
def load_pet_data(image_dir, mask_dir, img_size=(128, 128)):#(256, 256)):
    """
    Wczytuje obrazy i maski z odpowiednich katalogów.
    Maski są przekształcane z wartości [1, 2, 3] na [0, 1, 2].
    """
    images, masks = [], []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".jpg"):
            # Wczytanie obrazu
            img = load_img(os.path.join(image_dir, img_file), target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalizacja obrazów
            images.append(img_array)

            # Wczytanie maski
            mask_file = img_file.replace(".jpg", ".png")
            mask_path = os.path.join(mask_dir, mask_file)
            mask = Image.open(mask_path).resize(img_size, resample=Image.NEAREST)
            mask_array = np.array(mask, dtype=np.uint8)

            # Przekształcenie masek z [1, 2, 3] na [0, 1, 2]
            mask_array = mask_array - 1
            masks.append(mask_array)

    return np.array(images), np.expand_dims(np.array(masks), axis=-1)


# **Ścieżki do danych**
base_dir = os.path.expanduser("~/.keras/datasets")
image_dir = os.path.join(base_dir, "images")
mask_dir = os.path.join(base_dir, "annotations", "trimaps")

# **Wczytanie danych**
images, masks = load_pet_data(image_dir, mask_dir)# miejsce wywołania
#images, masks = load_pet_data(image_dir, mask_dir, img_size=(256, 256))

# **Podział danych na treningowe i testowe**
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# **Budowa modelu**
num_classes = 3
input_shape = (128, 128, 3) #definicja wejścia modelu
#input_shape = (256, 256, 3)
model = build_unet(input_shape, num_classes)

# **Kompilacja modelu**
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# **Trening modelu**
#history = model.fit(X_train, y_train, validation_split=0.1, batch_size=8, epochs=10)#30)
# Wariant A (Większy wsad): Stabilniejszy, szybszy obliczeniowo
history = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=10)
# Wariant B (Mniejszy wsad): Częstsze aktualizacje wag
#history = model.fit(X_train, y_train, validation_split=0.1, batch_size=4, epochs=10)

# **Testowanie modelu**
predictions = model.predict(X_test)

# **Zapisywanie wyników**
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

def save_results(image, mask, prediction, idx):
    plt.figure(figsize=(12, 4))

    # Obraz wejściowy
    plt.subplot(1, 3, 1)
    plt.title("Obraz")
    plt.imshow(image)
    plt.axis('off')

    # Prawdziwa maska
    plt.subplot(1, 3, 2)
    plt.title("Prawdziwa maska")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')

    # Przewidywana maska
    plt.subplot(1, 3, 3)
    plt.title("Przewidywana maska")
    plt.imshow(np.argmax(prediction, axis=-1).squeeze(), cmap='gray')
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f"result_{idx}.png"))
    plt.close()

# Zapis wyników dla pierwszych kilku próbek
for i in range(5):
    save_results(X_test[i], y_test[i], predictions[i], i)

# **Zapisywanie historii treningu**
plt.figure()
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.title('Historia treningu')
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()