import cv2
import numpy as np
import os
from collections import Counter
from pathlib import Path


def list_images(folder):
    folder = Path(folder)
    exts = {'.png', '.jpg', '.tif'}
    files = [
        p for p in folder.glob('*')
        if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort()
    return files  # ← Path!

def copy_move_forgery_detection(img_path: Path):
    quantization = 16 #Stopień uproszczenia (kwantyzacja DCT)
    tsimilarity = 5 # euclid distance similarity threshhold Próg podobieństwa euklidesowego (im mniejszy, tym bardziej rygorystyczny)
    tdistance = 20 # euclid distance between pixels threshold Minimalny dystans fizyczny (żeby nie wykrywać gładkiego tła obok siebie)
    vector_limit = 20 # shift vector elimination limit Ile takich samych przesunięć musi wystąpić, by uznać to za fałszerstwo
    block_counter = 0
    block_size = 8 # Ile takich samych przesunięć musi wystąpić, by uznać to za fałszerstwo


    # 1. WCZYTANIE OBRAZU
    img_path = img_path.resolve()
    print(f"1. Wczytywanie obrazu: {img_path.name}")

    image = cv2.imread(str(img_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Konwersja do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    arr = np.array(gray)
    temp = []
    # Pusta maska predykcji (wynikowa) - czarne tło
    prediction_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)

    column = arr.shape[1] - block_size
    row = arr.shape[0] - block_size
    dcts = np.empty((((column+1)*(row+1)), quantization+2))

    #----------------------------------------------------------------------------------------

    # 2. ANALIZA BLOKOWA I DCT
    print("2. Przetwarzanie bloków i DCT...")

    for i in range(0, row):
        for j in range(0, column):

            blocks = arr[i:i+block_size, j:j+block_size]
            imf = np.float32(blocks) / 255.0  # float conversion/scale
            dst = cv2.dct(imf)  # the dct
            blocks = np.uint8(np.float32(dst) * 255.0 ) # convert back
            # zigzag scan
            solution = [[] for k in range(block_size + block_size - 1)]
            for k in range(block_size):
                for l in range(block_size):
                    sum = k + l
                    if (sum % 2 == 0):
                        # add at beginning
                        solution[sum].insert(0, blocks[k][l])
                    else:
                        # add at end of the list
                        solution[sum].append(blocks[k][l])

            for item in range(0,(block_size*2-1)):
                temp += solution[item]

            temp = np.asarray(temp, dtype=float)
            temp = np.array(temp[:16])
            temp = np.floor(temp/quantization)
            temp = np.append(temp, [i, j])

            np.copyto(dcts[block_counter], temp)

            block_counter += 1
            temp = []
    print(f"   Przeanalizowano {block_counter} bloków.")

    #----------------------------------------------------------------------------------------

    # 3. SORTOWANIE LEKSYKOGRAFICZNE
    print("3. Sortowanie wektorów cech...")
    # Sortujemy po wartościach cech, aby podobne bloki znalazły się obok siebie na liście
    dcts = dcts[np.lexsort(np.rot90(dcts[:, :16]))]

    #----------------------------------------------------------------------------------------

    # 4. DOPASOWYWANIE (MATCHING)
    print("4. Szukanie podobnych fragmentów...")
    sim_array = []
    search_range = 10  # Jak daleko w posortowanej liście szukać "bliźniaków"

    # build list
    for i in range(len(dcts) - search_range):
        for j in range(1, search_range):
            # Porównanie wektorów cech (pierwsze 16 wartości)
            if np.linalg.norm(dcts[i, :16] - dcts[i+j, :16]) <= tsimilarity:
            
                # Współrzędne na obrazie (ostatnie 2 wartości)
                coord1 = dcts[i, -2:]
                coord2 = dcts[i+j, -2:]
            
                # Sprawdzenie dystansu fizycznego na obrazie
                dist = np.linalg.norm(coord1 - coord2)
            
                if dist >= tdistance:
                    y1, x1 = coord1
                    y2, x2 = coord2
                
                    # Obliczamy wektor przesunięcia (Shift Vector)
                    shift_y = y1 - y2
                    shift_x = x1 - x2
                
                    # Zapisujemy parę i wektor przesunięcia
                    sim_array.append([y1, x1, y2, x2, shift_y, shift_x])

    # convert once, after loops
    sim_array = np.array(sim_array)

    if len(sim_array) == 0:
        print("\n--- WYNIK: Obraz wydaje się autentyczny (brak podejrzanych powtórzeń). ---")
        cv2.imshow("Original", image)
        cv2.waitKey(0)
        exit()

    print(f"   Wstępnie znaleziono {len(sim_array)} par.")

    # 5. ELIMINACJA FAŁSZYWYCH DOPASOWAŃ (SHIFT VECTOR FILTERING)
    print("5. Weryfikacja spójności przesunięć...")

    # Zliczamy najpopularniejsze wektory przesunięcia
    # (Prawdziwe fałszerstwo copy-move ma wiele bloków przesuniętych o ten sam wektor)
    shift_vectors = [(row[4], row[5]) for row in sim_array]
    shift_counts = Counter(shift_vectors)

    final_matches = []
    for row in sim_array:
        shift = (row[4], row[5])
        # Jeśli dany wektor przesunięcia występuje rzadziej niż limit, odrzucamy go jako szum
        if shift_counts[shift] >= vector_limit:
            final_matches.append(row)

    final_matches = np.array(final_matches)

    # 6. RYSOWANIE WYNIKU
    print("6. Generowanie mapy fałszerstwa...")

    if len(final_matches) > 0:
        for match in final_matches:
            y1, x1, y2, x2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
        
            # Malujemy na biało oba pasujące do siebie bloki
            cv2.rectangle(prediction_mask, (x1, y1), (x1+block_size, y1+block_size), 255, -1)
            cv2.rectangle(prediction_mask, (x2, y2), (x2+block_size, y2+block_size), 255, -1)
    
        # Opcjonalnie: Operacje morfologiczne, aby połączyć bliskie kropki w plamy
        kernel = np.ones((5,5), np.uint8)
        prediction_mask = cv2.morphologyEx(prediction_mask, cv2.MORPH_CLOSE, kernel)
    
        print(f"--- WYNIK Znaleziono {len(final_matches)} pasujących bloków. ---")
    else:
        print("--- WYNIK: Po filtracji obraz uznano za czysty. ---")

    # 7. WYŚWIETLANIE
    def show_scaled(name, img, max_h=800):
        h, w = img.shape[:2]
        if h > max_h:
            scale = max_h / h
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        cv2.imshow(name, img)

    show_scaled("Oryginalny Obraz", image)
    show_scaled("Wykryte Falszerstwo (Biale pola)", prediction_mask)

    print("Naciśnij dowolny klawisz, aby zamknąć...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    full = Path(__file__).resolve()
    base = full.parent

    images = list_images(base)

    for img_path in images:
        print("=" * 60)
        print(f"Przetwarzam: {img_path.name}")

        copy_move_forgery_detection(img_path)

        print("Zakończono obraz:", img_path.name)