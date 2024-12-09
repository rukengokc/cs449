import cv2
import mediapipe as mp
import math
import tkinter as tk
from PIL import Image, ImageTk

def on_button_click():
    global button_click_count
    button_click_count += 1
    print("Button clicked! Total clicks:", button_click_count)
    status_label.config(text=f"Button Clicked! Counter: {button_click_count}")

def update_frame():
    global blink_flag
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = get_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
            right_ear = get_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)

            # Are the eyes closed?
            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                if not blink_flag:
                    blink_flag = True
            else:
                # Eyes opened and blink_flag is True, meaning a blink was completed
                if blink_flag:
                    print("Blink detected! Clicking the button...")
                    my_button.invoke()  # Programmatically click the button (calls on_button_click)
                    blink_flag = False

    im = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_eye_aspect_ratio(landmarks, eye_points):
    p = [(landmarks[x].x, landmarks[x].y) for x in eye_points]
    vertical_1 = distance(p[1], p[5])
    vertical_2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])
    if horizontal == 0:
        return 0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Create Tkinter interface
root = tk.Tk()
root.title("Blink Button Interface")

status_label = tk.Label(root, text="Blink to click the button...")
status_label.pack()

my_button = tk.Button(root, text="Button to be clicked", command=on_button_click)
my_button.pack()

video_label = tk.Label(root)
video_label.pack()

# Counter variable
button_click_count = 0

# MediaPipe settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.22
blink_flag = False

root.after(10, update_frame)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
