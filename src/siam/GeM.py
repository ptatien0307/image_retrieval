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