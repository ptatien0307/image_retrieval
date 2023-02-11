import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers

class GeMPoolingLayer(layers.Layer):
    def __init__(self, init_norm=3.0, **kwargs):
        self.init_norm = init_norm=3.0

        super(GeMPoolingLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'init_norm': self.init_norm ,
        })
        return config


    def build(self, input_shape):
        self.p = self.add_weight(name='norms', 
                                 shape=(1,),
                                 initializer=initializers.constant(self.init_norm),
                                 trainable=True)
        super(GeMPoolingLayer, self).build(input_shape)


    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1,2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        inputs = tf.math.l2_normalize(inputs)
        return inputs



    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])


class MySIAM:
  def __init__(self, MODEL_NAME, GeM=False):
    self.MODEL_NAME = MODEL_NAME
    if GeM:
        self.model = load_model(MODEL_NAME, custom_objects={'GeMPoolingLayer': GeMPoolingLayer})
    else:
        self.model = load_model(MODEL_NAME)

  def extract(self, image):
    image = cv2.resize(image, (224,224))
    image = np.expand_dims(image, axis=0)
    feature = self.model.predict(image, verbose=0)[0]

    return feature