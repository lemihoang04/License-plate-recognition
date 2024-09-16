from ultralytics import YOLO
from PIL import Image

model = YOLO("E:/Python_Project/Python_project1/runs/detect/train/weights/best.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model("E:/Python_Project/Python_project1/bien3.jpg")

for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
