import cv2 
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
class MyVGG16:

    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)


    def extract(self, img):
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feature = self.model.predict(img, verbose=0)

        return feature.flatten()
