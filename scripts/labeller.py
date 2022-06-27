import torch
import os 

class Model():
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name

    def loadModel(self):
        self.model = torch.hub.load('self.model_path', self.model_name, pretrained=True)

    def modelDetect(self, img_path):
        self.results = self.model(img_path)
        return self.results

    def makeLabels(self, img_path):
        
