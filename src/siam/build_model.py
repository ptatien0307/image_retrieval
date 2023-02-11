import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from GeM import GeMPoolingLayer

def embedding_model():
    """
        Embedding model:
        - Pretrain_model: for feature extraction. A global pooling layer will be 
                    added at the end of the pretraied model
        - embed_model: embedding vector for triplet 
    """

    # Pretrained model
    base_model = ResNet50(
        weights="imagenet", input_shape=(224,224,3), include_top=False
    )
    base_model.trainable = False

    # # Embedding model
    # embed_model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='sigmoid')
    # ])


    # Build full model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    outputs = GeMPoolingLayer()(x)
    # outputs = embed_model(x, training=True)
    full_model = tf.keras.Model(inputs, outputs)

    return full_model


def triplet_siamese_net(embedding_model):
    """
        Create siamese network with 
        three embedding model (for Anchor, Positive and Negative)
        these embedding model will share the same weights
    """
    input_anchor = tf.keras.layers.Input(shape=(224,224,3))
    input_positive = tf.keras.layers.Input(shape=(224,224,3))
    input_negative = tf.keras.layers.Input(shape=(224,224,3))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
    return net

def pair_siamese_net(embedding_model):
   
    input_1 = tf.keras.layers.Input(shape=(224,224,3))
    input_2 = tf.keras.layers.Input(shape=(224,224,3))

    embedding_1 = embedding_model(input_1)
    embedding_2 = embedding_model(input_2)

    output = tf.keras.layers.concatenate([embedding_1, embedding_2], axis=1)

    net = tf.keras.models.Model([input_1, input_2], output)
    return net
