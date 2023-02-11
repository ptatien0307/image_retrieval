import cv2 
from skimage.feature import hog
import argparse

class MyHOG:

    def __init__(self):
        pass

    def extract(self, img):
        resized_img = cv2.resize(img, (128, 64))
        feature = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2))
        
        return feature

 