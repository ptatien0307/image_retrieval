import os
import cv2
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from imutils import paths
import tensorflow as tf
from utils.evaluation import cal_AP
from siam.GeM import GeMPoolingLayer

from feature_extraction.VW import MyVW
from feature_extraction.HOG import MyHOG
from feature_extraction.SIAM import MySIAM
from feature_extraction.DELF import MyDELF
from feature_extraction.VGG16 import MyVGG16

def load_extractor():

    # Extractor
    GeM = None
    index_path = args['index']
    index_name = index_path.split('/')[-1]
    index_name_sep = index_name.split('.')
    model_name_sep = index_name_sep[0].split('_')
    if ('SIAM' not in model_name_sep):
      MODEL_NAME = f'model/{model_name_sep[0]}.h5'
    else:
      if ('GeM' in model_name_sep):
        MODEL_NAME = f'model/{model_name_sep[0]}_{model_name_sep[1]}_{model_name_sep[2]}.h5'
        GeM = True
      else:
        MODEL_NAME = f'model/{model_name_sep[0]}_{model_name_sep[1]}.h5'
        GeM = False

    # Choose extractor
    name_length = len(MODEL_NAME.split('_'))
    if name_length == 1:
        # Using deep feature vector extracted from VGG16 
        if MODEL_NAME[6:-3] == 'VGG16':
            extractor = MyVGG16()
        # Using HOG feature
        elif MODEL_NAME[6:-3] == 'HOG':
            extractor = MyHOG()
        # Using Bag of Visual Word feature
        elif MODEL_NAME[6:-3] == 'BOVW':
            extractor = MyVW()
        # Using DELF feature
        else: 
            extractor = MyDELF()
    else:
        # Using embedding model from siamese network
        extractor = MySIAM(MODEL_NAME, GeM)

    return extractor


def load_index():
    """
        Read file index
    """
    indexer = faiss.read_index(args['index'])
    return indexer

        


def extract_feature(image_path, extractor):
    """
        Extract features of query
    """
    image = cv2.imread(image_path)
    feature = extractor.extract(image)
    feature = np.array(feature, dtype='float32')

    return feature


def compute_MAP(indexer, extractor, queries, collection):
    """
        Calculate AP for each query and MAP 
        for all queries at rank k
    """
    
    MAP = 0
    print('[INFO]: Calculate MAP ...')
    for image_path in tqdm(queries):
        query_label = image_path.split(os.path.sep)[-2]
      
        feature = extract_feature(image_path, extractor)

        # Search
        _, indices = indexer.search(feature.reshape(1,-1),k=args['top'])
        result_top_k = collection[indices[0]]

        ap = cal_AP(query_label, result_top_k)

        MAP += ap

    return MAP / len(queries)


def main(args):

   # Load dataset features
    indexer = load_index()

    # # Load extractor
    # extractor = load_extractor()

    extractor = tf.keras.models.load_model('/content/drive/MyDrive/cs336/model/ResNet50_SIAM_GeM.h5', custom_objects={'GeMPoolingLayer': GeMPoolingLayer})



    # Evaluation
    queries_paths = list(paths.list_images(args['queries']))
    collection_paths = np.array(list(paths.list_images(args['collection'])))

    # Sort collection_paths if using DELF
    if args['index'][8:12] == 'DELF':
        collection_paths = sorted(collection_paths, key=lambda x: x.split('/')[-1])
        collection_paths = np.array(collection_paths)
        
    print(compute_MAP(indexer, extractor, queries_paths, collection_paths))

def args_parse():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-q', '--queries', required=True)
    parser.add_argument('-i', '--collection', required=True)
    parser.add_argument('-o', '--index', required=True)
    parser.add_argument('-t', '--top', required=True, type=int)


    # End default optional arguments

    return vars(parser.parse_args())

if __name__ == "__main__":
    
    args = args_parse()

    main(args)