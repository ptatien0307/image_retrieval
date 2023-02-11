
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from imutils import paths

from indexer.indexing import indexing
from feature_extraction.HOG import MyHOG
from feature_extraction.VGG16 import MyVGG16

from siam.GeM import GeMPoolingLayer


def extract_dataset():
    extractor = None
    features = [] 

    # extractor
    if args['method'] == 'VGG16':
        extractor = MyVGG16()
    if args['method'] == 'HOG':
        extractor = MyHOG()
    if args['method'] == 'Triplet':
        extractor = tf.keras.models.load_model(args['triplet_model'], custom_objects={'GeMPoolingLayer': GeMPoolingLayer})
    
    # extraction
    # collection = list(paths.list_images(args['input_folder']))

    collection = np.load('/content/drive/MyDrive/cs336/dataset/ds_numpy.npy')
    for image in tqdm(collection):
      image = np.expand_dims(image, axis=0)
      with tf.device('/gpu:0'):
        feature = extractor.predict(image, verbose=0)[0]

      # image = cv2.imread(image_path)
      # feature = extractor.extract(image)
      features.append(feature)
        
        
    # Save feature 
    features = np.array(features, dtype='float32')
    np.savez_compressed(f"/content/drive/MyDrive/cs336/feature_dataset/{args['method']}-features.npz", features=features)
    return features
        

def main(args):

    # Feature extraction
    # features = extract_dataset()

    # Indexing
    features = np.load('/content/drive/MyDrive/cs336/feature_dataset/ResNet50_SIAM_GEM_F.npz')['features']
    name_index = None
    if args['index'] == 1:
        name_index = args['method']
    elif args['index'] == 2:
        name_index = args['method'] + '_LSH' 
    elif args['index'] == 3:
        name_index = args['method'] + '-IVF'
    elif args['index'] == 4:
        name_index = args['method'] + '-PQ'

    indexing(features, name_index, args['index'])



def args_parse():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    
    parser.add_argument('-i', '--input_folder', required=True)
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-t', '--index', required=True, type=int)

    parser.add_argument('-e', '--triplet_model', required=False)

    return vars(parser.parse_args())

if __name__ == "__main__":
    
    args = args_parse()

    main(args)