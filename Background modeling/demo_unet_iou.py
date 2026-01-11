import os
import numpy as np
import tensorflow as tf  # Główny import rozwiązujący problemy z IntelliSense w Visual Studio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# Definicja aliasów dla Keras - dzięki temu Visual Studio poprawnie widzi te moduły
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
concatenate = tf.keras.layers.concatenate
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array

# **Definicja metryki IoU**
def iou_metric(y_true, y_pred, num_classes=3):
    """
    Oblicza średnią wartość IoU (Intersection over Union) dla wszystkich klas.
    """
    y_pred = tf.argmax(y_pred, axis=-1)  # Przekształcenie na klasy
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)  # Rzeczywiste maski

    iou_list = []
    for c in range(num_classes):
        true_class = tf.cast(y_true == c, tf.float32)
        pred_class = tf.cast(y_pred == c, tf.float32)

        intersection = tf.reduce_sum(true_class * pred_class)
        union = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) - intersection

        # Unikanie dzielenia przez zero
        iou = tf.where(union == 0, 1.0, intersection / union)
        iou_list.append(iou)

    return tf.reduce_mean(iou_list)

# **Budowa sieci U-Net dla wieloklasowej segmentacji**
def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
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

    # Decoder
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

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# **Funkcja do wczytywania danych**
def load_pet_data(image_dir, mask_dir, img_size=(128, 128)):
    """
    Wczytuje obrazy i maski z katalogów.
    Maski są przekształcane z [1, 2, 3] na [0, 1, 2].
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
images, masks = load_pet_data(image_dir, mask_dir)

# **Podział danych na treningowe i testowe**
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# **Budowa modelu**
num_classes = 3
input_shape = (128, 128, 3)
model = build_unet(input_shape, num_classes)

# **Kompilacja modelu**
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', iou_metric])

# **Trening modelu**
history = model.fit(X_train, y_train, validation_split=0.1, batch_size=8, epochs=10)

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

# Wizualizacja IoU
plt.figure()
plt.plot(history.history['iou_metric'], label='IoU - trening')
plt.plot(history.history['val_iou_metric'], label='IoU - walidacja')
plt.xlabel('Epoka')
plt.ylabel('IoU')
plt.legend()
plt.title('Historia metryki IoU')
plt.savefig(os.path.join(output_dir, 'iou_history.png'))
plt.close()

