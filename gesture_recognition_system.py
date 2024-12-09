import cv2
import mediapipe as mp
import math
import os
import time

# TensorFlow log suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

frame_width, frame_height = 640, 480
grid_rows, grid_cols = 3, 3
zone_width, zone_height = frame_width // grid_cols, frame_height // grid_rows

selected_row, selected_col = 1, 1
blink_counters = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

threshold = 0.015
last_move_time = time.time()
move_delay = 0.3
last_blink_time = time.time()
blink_delay = 0.5
blink_color_duration = 0.2

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.22
blink_flag = False

last_nose_position = None
cursor_position = (frame_width // 2, frame_height // 2)
movement_history = []

# Store last detected direction and time
last_direction_detected = None
last_direction_time = 0
direction_display_duration = 1.0  # seconds

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])*2 + (p1[1] - p2[1])*2)

def detect_scroll_direction(nose_position, last_position):
    if not last_position:
        return None
    dx = nose_position[0] - last_position[0]
    dy = nose_position[1] - last_position[1]

    if abs(dx) > threshold and abs(dy) < threshold:
        return "RIGHT" if dx > 0 else "LEFT"
    elif abs(dy) > threshold and abs(dx) < threshold:
        return "DOWN" if dy > 0 else "UP"
    return None

def stabilize_movement(nose_x, nose_y, history, max_history=5):
    history.append((nose_x, nose_y))
    if len(history) > max_history:
        history.pop(0)
    avg_x = sum([pos[0] for pos in history]) / len(history)
    avg_y = sum([pos[1] for pos in history]) / len(history)
    return avg_x, avg_y

def get_eye_aspect_ratio(landmarks, eye_points):
    try:
        p = [(landmarks[x].x, landmarks[x].y) for x in eye_points]
        vertical_1 = distance(p[1], p[5])
        vertical_2 = distance(p[2], p[4])
        horizontal = distance(p[0], p[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0
    except IndexError:
        return 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı. Lütfen bağlantıyı kontrol edin.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera çerçevesi alınamadı.")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            nose_x, nose_y = nose_tip.x, nose_tip.y

            stabilized_x, stabilized_y = stabilize_movement(nose_x, nose_y, movement_history)
            direction = detect_scroll_direction((stabilized_x, stabilized_y), last_nose_position)
            last_nose_position = (stabilized_x, stabilized_y)

            current_time = time.time()

            if direction and current_time - last_move_time > move_delay:
                if direction == "UP" and selected_row > 0:
                    selected_row -= 1
                elif direction == "DOWN" and selected_row < grid_rows - 1:
                    selected_row += 1
                elif direction == "LEFT" and selected_col > 0:
                    selected_col -= 1
                elif direction == "RIGHT" and selected_col < grid_cols - 1:
                    selected_col += 1

                # Record direction and time
                last_direction_detected = direction
                last_direction_time = current_time
                last_move_time = current_time

            # Update cursor position
            cursor_position = ((selected_col * zone_width) + zone_width // 2,
                               (selected_row * zone_height) + zone_height // 2)

            # Eye blink detection
            left_ear = get_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
            right_ear = get_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)

            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                if not blink_flag and current_time - last_blink_time > blink_delay:
                    blink_flag = True
                    blink_counters[selected_row][selected_col] += 1
                    last_blink_time = current_time
            else:
                blink_flag = False

    # Draw grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c * zone_width, r * zone_height
            x2, y2 = (c + 1) * zone_width, (r + 1) * zone_height

            if r == selected_row and c == selected_col:
                current_time = time.time()
                if current_time - last_blink_time < blink_color_duration:
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Blinks: {blink_counters[r][c]}", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw cursor
    cv2.putText(frame, "X", (cursor_position[0] - 10, cursor_position[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display direction if recently detected
    if last_direction_detected and (time.time() - last_direction_time < direction_display_duration):
        message = f"Head Move: {last_direction_detected}"
        cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)

    cv2.imshow("Interface", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()