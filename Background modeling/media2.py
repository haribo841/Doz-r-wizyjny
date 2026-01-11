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

# Otwórz kamerę
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
            #mp_drawing.draw_landmarks(
            #    image=frame,
            #    landmark_list=face_landmarks,
            #    connections=mp_face_mesh.FACEMESH_IRISES,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles
            #        .get_default_face_mesh_iris_connections_style()
            #)
            # Współrzędne źrenic
            left_pupil = face_landmarks.landmark[468]
            right_pupil = face_landmarks.landmark[473]

            # Konwersja współrzędnych na piksele
            h, w, _ = frame.shape
            left_x, left_y = int(left_pupil.x * w), int(left_pupil.y * h)
            right_x, right_y = int(right_pupil.x * w), int(right_pupil.y * h)

            # Rysowanie punktów reprezentujących źrenice
            cv2.circle(frame, (left_x, left_y), 1, (0, 255, 0), -1)  # Zielony punkt dla lewego oka
            cv2.circle(frame, (right_x, right_y), 1, (255, 0, 0), -1)  # Niebieski punkt dla prawego oka

    # Wyświetlanie obrazu
    cv2.imshow('MediaPipe Iris Detection', frame)

    # Przerwij po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()