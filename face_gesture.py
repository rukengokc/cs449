import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])*2 + (p1[1]-p2[1])*2)

def get_eye_aspect_ratio(landmarks, eye_points):
    p = [ (landmarks[x].x, landmarks[x].y) for x in eye_points ]
    vertical_1 = distance(p[1], p[5])
    vertical_2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.22
BLINK_FRAMES = 3
blink_flag = False
blink_timer = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = get_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
            right_ear = get_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)

            # If eyes are closed
            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                blink_flag = True
            else:
                if blink_flag:
                    # When blink is detected
                    print("Blink detected!")
                    
                    # Display text on the screen
                    cv2.putText(frame, 'Blink detected!', (50,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    blink_flag = False

    if blink_flag:
        cv2.putText(frame, 'Blink detected!', (50,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()