import numpy as np
import cv2
from numpy.matrixlib.defmatrix import N
import scipy
import argparse
import os
from tqdm import tqdm
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
import pickle
import joblib
import numpy as np


class MyVW:
    def __init__(self):
        self.descriptor = None
        k , self.visual_words = joblib.load('/content/drive/MyDrive/cs336/feature_dataset/BOVW_F.pkl')
        self.tfidf_vectorize = pickled_model = pickle.load(open('/content/drive/MyDrive/cs336/model/tfidf_vectorizer.pickle', 'rb'))
        self.img_visual_word = None

    def SIFT(self, image):
        extractor = cv2.SIFT_create(nfeatures=200)
        keypoint, self.descriptor = extractor.detectAndCompute(image, None)


    def vector_quantization(self):
        img_visual_word, distance = vq(self.descriptor, self.visual_words)
        self.img_visual_word = ' '.join(img_visual_word.astype(str))


    def extract(self, image):
        self.SIFT(image)
        self.vector_quantization()
        tfidf_vector = self.tfidf_vectorize.transform([self.img_visual_word])
        return tfidf_vector.A[0]

