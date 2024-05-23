import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision.transforms import functional as F
import pathlib

# Set default file dialog to Windows style
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'D:/python_work/recycling/best_v4.pt', force_reload=True)
class_names = ['Glass', 'metal', 'plastic', 'vinyl']

# Initialize camera
cap = cv2.VideoCapture(0)

# Specify box color for class
box_colors = {
    'Glass': (255, 0, 0),    # Blue
    'metal': (0, 255, 0),  # Green
    'plastic': (0, 0, 255), # Red
    'vinyl': (255, 255, 0)  # Cyan
}

# Function to perform object detection
def detect_objects(panel):
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        results = model(image)
    
        for box, score, cls in zip(results.xyxy[0][:, :4], results.xyxy[0][:, 4], results.xyxy[0][:, 5]):
            if score > 0.5:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_colors[class_names[int(cls)]], 2)
                label = f"{class_names[int(cls)]}: {score:.2f}"
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colors[class_names[int(cls)]], 2)

        # Show frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        panel.config(image=photo)
        panel.image = photo

    if not paused:
        panel.after(10, detect_objects, panel)

# Stop function
paused = True
def toggle_pause():
    global paused
    paused = not paused
    if paused:
        Pause_button_text.set("Resume")
    else:
        Pause_button_text.set("Pause")
        detect_objects(panel)

# Initialize GUI
window = tk.Tk()
window.title("Recycling Detection")

# Add logo or icon
logo = tk.PhotoImage(file="D:/python_work/recycling/logo1.png")
logo_label = tk.Label(window, image=logo)
logo_label.pack()

# Add title label
title_label = tk.Label(window, text="Recycling Detection Application", font=("Helvetica", 16))
title_label.pack(pady=10)

# Create a panel to display the video feed
panel = tk.Label(window)
panel.pack(padx=10, pady=10)

# Start/pause detection button
Pause_button_text = tk.StringVar()
Pause_button_text.set("Resume")
Pause_button = tk.Button(window, textvariable=Pause_button_text, command=toggle_pause)
Pause_button.pack(padx=10, pady=5)

# Exit button
exit_button = tk.Button(window, text="Exit", command=window.quit)
exit_button.pack(padx=10, pady=5)

window.mainloop()

# When the application closes, turn off the camera and close the window.
cap.release()
cv2.destroyAllWindows()