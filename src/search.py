import os
import cv2
import time
import faiss
import argparse
import numpy as np
from imutils import paths


from feature_extraction.VW import MyVW
from feature_extraction.HOG import MyHOG
from feature_extraction.SIAM import MySIAM
from feature_extraction.DELF import MyDELF
from feature_extraction.VGG16 import MyVGG16


def load_extractor(index_path):

    # Extractor
    GeM = None
    index_path = index_path
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

def load_index(index_path):
    """
        Read file index
    """
    indexer = faiss.read_index(index_path)
    return indexer


def extract_feature(query, extractor):
    """
        Extract features of query
    """
    image = cv2.imread(query)
    feature = extractor.extract(image)
    feature = np.array(feature, dtype='float32')

    return feature



def search(query, indexer, extractor, collection_paths, top_k):
    """
        Search the top k image which has 
        the most relevant to the image query
    """

    # Extract feature of image query
    feature = extract_feature(query, extractor)

    # Search
    _, indices = indexer.search(feature.reshape(1,-1),k=top_k)
    
    # Take top k from collection_paths
    result_top_k = collection_paths[indices[0]]


    return result_top_k

    
    # # Write top k path into file
    # with open("/content/drive/MyDrive/cs336/output.txt", "w") as f:
    #     f.write(query + '\n')
    #     for i in result_top_k:
    #         f.write(i + '\n')


def main(args):

    query_path = args['query']
    index_path = args['index']
    data_path = args['collection']

    # Load dataset features
    indexer = load_index(index_path)

    # Load extractor
    extractor = load_extractor(index_path)
    

    collection_paths = np.array(list(paths.list_images(data_path)))
    top_k = args['top']
    # Sort collection_paths if using DELF
    if args['index'][8:12] == 'DELF':
        collection_paths = sorted(collection_paths, key=lambda x: x.split('/')[-1])
        collection_paths = np.array(collection_paths)

    # Search
    search(query_path, indexer, extractor, collection_paths, top_k)




def args_parse():

    parser = argparse.ArgumentParser(description="Methods extract image.")


    parser.add_argument('-q', '--query', required=True)
    parser.add_argument('-i', '--collection', required=True)
    parser.add_argument('-o', '--index', required=True)
    parser.add_argument('-t', '--top', required=True, type=int)



    # End default optional arguments

    return vars(parser.parse_args())

if __name__ == "__main__":
    
    args = args_parse()

    main(args)