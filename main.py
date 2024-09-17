import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image
import cv2

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select an image or video",
        filetypes=[("Image/Video files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi")]
    )
    return file_path

def display_image(img_array):
    im = Image.fromarray(img_array[..., ::-1])
    im.show()

def predict(file_path):
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        results = model(file_path)
        for r in results:
            print(r.boxes)
            im_array = r.plot()
            display_image(im_array)
    elif file_path.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for r in results:
                frame = r.plot()
            cv2.imshow('YOLO Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def create_gui():
    root = tk.Tk()
    root.title("Select an image or video for prediction")

    btn_select_file = tk.Button(root, text="Select an image or video", command=lambda: predict(open_file()))
    btn_select_file.pack(pady=20)

    root.mainloop()

model = YOLO("E:/Python_Project/Python_project1/runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    create_gui()
