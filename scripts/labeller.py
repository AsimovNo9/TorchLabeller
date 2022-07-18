import torch
import os
import glob
import cv2


class Model:
    def __init__(self, model_origin, model_type, model_path):
        self.model_origin = model_origin
        self.model_type = model_type
        self.model_path = model_path
        self.makeLabels("./images")

    def loadModel(self):
        self.model = torch.hub.load(
            self.model_origin, self.model_type, path=self.model_path
        )

    def modelDetect(self, image):
        self.results = self.model(image)
        self.detections = self.results.pandas().xyxy[0].values.tolist()
        return self.detections

    def makeLabels(self, img_path):
        self.loadModel()
        images = []
        for img in glob.glob(f"{img_path}/*.png"):
            print(img)
            read_image = cv2.imread(img)
            images.append(read_image)

        for image in images:
            detections = self.makeLabels(image)
            print(detections)


if __name__ == "__main__":
    Model("ultralytics/yolov5", "custom", "../utils/yolov5n6.pt")
