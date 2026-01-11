import cv2
import mediapipe as mp

# Inicjalizacja MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Ustawienia FaceMesh z obsługą detekcji tęczówki
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Maksymalna liczba twarzy
    refine_landmarks=True,  # Włączenie detekcji tęczówki
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicjalizacja rysowania
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Otwórz kamerę lub wideo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja obrazu na RGB (wymagane przez MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Przetwarzanie obrazu w MediaPipe
    results = face_mesh.process(rgb_frame)

    # Rysowanie wykrytych punktów twarzy i tęczówki
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style()
            )

    # Wyświetlanie obrazu
    cv2.imshow('MediaPipe Iris Detection', frame)

    # Przerwij po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()