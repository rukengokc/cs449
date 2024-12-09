import cv2
import mediapipe as mp
import math
import time

frame_width, frame_height = 640, 480
grid_rows, grid_cols = 3, 3
zone_width, zone_height = frame_width // grid_cols, frame_height // grid_rows

selected_row, selected_col = 1, 1

# Thresholds for detecting movement
threshold = 0.015
move_delay = 0.3
last_move_time = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

last_nose_position = None
movement_history = []

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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Nose tip is landmark #1
            nose_tip = face_landmarks.landmark[1]
            nose_x, nose_y = nose_tip.x, nose_tip.y

            stabilized_x, stabilized_y = stabilize_movement(nose_x, nose_y, movement_history)

            direction = detect_scroll_direction((stabilized_x, stabilized_y), last_nose_position)
            last_nose_position = (stabilized_x, stabilized_y)

            current_time = time.time()
            # Move selection if direction detected and delay passed
            if direction and current_time - last_move_time > move_delay:
                if direction == "UP" and selected_row > 0:
                    selected_row -= 1
                elif direction == "DOWN" and selected_row < grid_rows - 1:
                    selected_row += 1
                elif direction == "LEFT" and selected_col > 0:
                    selected_col -= 1
                elif direction == "RIGHT" and selected_col < grid_cols - 1:
                    selected_col += 1
                last_move_time = current_time

    # Draw grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c * zone_width, r * zone_height
            x2, y2 = (c + 1) * zone_width, (r + 1) * zone_height
            if r == selected_row and c == selected_col:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (255, 0, 0)
                thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw cursor on selected cell
    cursor_x = (selected_col * zone_width) + zone_width // 2
    cursor_y = (selected_row * zone_height) + zone_height // 2
    cv2.putText(frame, "X", (cursor_x - 10, cursor_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Interface - With Directions", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
