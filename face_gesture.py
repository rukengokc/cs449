import cv2
import mediapipe as mp

# MediaPipe yapılandırması
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamera akışı
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü işlemeye hazırlama
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Sol göz işaretlerini kontrol et
            left_eye = [face_landmarks.landmark[i] for i in range(133, 144)]  # Sol göz işaretleri
            left_eye_ratio = calculate_eye_aspect_ratio(left_eye)
            
            if left_eye_ratio < 0.2:  # Göz kırpma algılama eşiği
                trigger_event("Button Clicked!")
    
    # Görüntü gösterimi
    cv2.imshow('Face Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def calculate_eye_aspect_ratio(eye_landmarks):
    # Göz alanını hesapla
    return eye_area / total_area  # Basitleştirilmiş bir örnek

def trigger_event(action):
    print(f"{action}")
