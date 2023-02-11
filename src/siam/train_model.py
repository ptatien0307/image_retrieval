import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input

import random
import numpy as np
import os
from imutils import paths
import cv2
from tqdm import tqdm

from GeM import GeMPoolingLayer
from loss_function import triplet_loss, contrastive_loss
from load_data import create_triplets, create_pairs, data_generator
from build_model import embedding_model, triplet_siamese_net, pair_siamese_net


import random
random.seed(25)







def main():

    collection_paths = list(paths.list_images('/content/drive/MyDrive/cs336/dataset/collection'))
    y_train = np.array([path.split(os.path.sep)[-2] for path in collection_paths])
    x_train = np.load('/content/drive/MyDrive/cs336/dataset/ds_numpy.npy')


    embed_model = embedding_model()
    net = triplet_siamese_net(embed_model)
    batch_size= 64
    steps_per_epoch = int(x_train.shape[0]/64)

    net.compile(loss=triplet_loss, optimizer='adam')
    with tf.device('/gpu:0'):
        _ = net.fit(
            data_generator(batch_size, x_train, y_train),
            steps_per_epoch=steps_per_epoch,
            epochs=10,
        )

        
    embed_model.save('/content/drive/MyDrive/cs336/model/triplet_model.h5')

main()