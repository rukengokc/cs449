import cv2

frame_width, frame_height = 640, 480
grid_rows, grid_cols = 3, 3
zone_width, zone_height = frame_width // grid_cols, frame_height // grid_rows

# Initially selected cell at center
selected_row, selected_col = 1, 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    # Draw a 3x3 grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c * zone_width, r * zone_height
            x2, y2 = (c + 1) * zone_width, (r + 1) * zone_height

            # Highlight the selected cell differently
            if r == selected_row and c == selected_col:
                color = (0, 255, 0)  # Green
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw a simple cursor (X) in the selected cell
    cursor_x = (selected_col * zone_width) + zone_width // 2
    cursor_y = (selected_row * zone_height) + zone_height // 2
    cv2.putText(frame, "X", (cursor_x - 10, cursor_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Interface - Basic", frame)

    # Press Esc to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()