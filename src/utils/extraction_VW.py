import os
import cv2
import math
import faiss
import scipy
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from imutils import paths
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans
from sklearn.feature_extraction.text import TfidfVectorizer


from indexer.indexing import indexing


class MySIFT:
    def __init__(self):
        self.extractor = cv2.SIFT_create(nfeatures=200)
        self.visual_word = None
    
    def extract(self, image):
        """
            Extract SIFT feature 
        """
        keypoint, descriptor = self.extractor.detectAndCompute(image, None)

        return keypoint, descriptor

class MyVW:
    def __init__(self):
        self.sift_extractor = MySIFT()
        self.descriptors = list()

    
    def extract_SIFT(self, input_folder):
        """
            Extract SIFT feature of all image in the input folder and
            save to a .npz file
        """

        # Extract features of each image
        for label in os.listdir(input_folder):
            for image_name in tqdm(os.listdir(os.path.join(input_folder, label))):

                image_path = os.path.join(input_folder, label, image_name)
                image = cv2.imread(image_path)

                keypoint, descriptor = self.sift_extractor.extract(image)
                self.descriptors.append(descriptor)

        # Save file
        self.descriptors = np.array(self.descriptors, dtype='object')
        np.savez_compressed('/content/drive/MyDrive/cs336/feature_dataset/features/descriptors.npz', descriptos=self.descriptors)

    def create_visual_words(self):
        """
            Create a book of 200 visual words
        """
        k = 200
        iters = 1
        self.visual_word, variance = kmeans(np.vstack(self.descriptors), k, iters)
        k, codebook = joblib.load("/content/drive/MyDrive/cs336/feature_dataset/features/bovw.pkl")

        return self.visual_word

    def vector_quantization(self, visual_words):
        """
            Quantize each vector based on 200 visual words
        """
        img_visual_words = []
        for img_descriptors in self.descriptors:
            # for each image, map each descriptor to the nearest codebook entry
            img_vw, distance = vq(img_descriptors, visual_words)
            img_vw = ' '.join(img_vw.astype(str))
            img_visual_words.append(img_vw)

        return img_visual_words



    def tf_idf(self):
        """
            Calculate TF-IDF
        """
        vocab = np.array(list(range(200)))
        vocab = vocab.astype(str)

        visual_words = self.create_visual_words()
        image_visual_words = self.vector_quantization(visual_words)

        vectorizer = TfidfVectorizer(vocabulary=vocab) 
        vectors = vectorizer.fit_transform(image_visual_words)

        return vectors.A


def main(args):
    extractor = MyVW()
    print('[INFO]: Calculate SIFT')
    extractor.extract_SIFT(args['input_path'])

    print('[INFO]: Calculate TFIDF')
    myTFIDF = extractor.tf_idf()
    print(myTFIDF)


def args_parser():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_path',  required=True,
                        help="The path of the input image.")
    return vars(parser.parse_args())

if __name__ == "__main__":

    args = args_parser()

    main(args)  