import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from imutils import paths
import cv2
from tqdm import tqdm

class MyDELF:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    def extract(self, image):
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np_image = np.array(image)
        float_image = tf.image.convert_image_dtype(np_image, tf.float32)
        
        # Extract DELF descriptors
        descriptors =  self.model(
                image=float_image,
                score_threshold=tf.constant(50.0),
                image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
                max_feature_num=tf.constant(500))['descriptors']

        # Choose first 65 descriptors from all descriptors
        # if there's not enough 65 then use cv2.resize to make it 65
        if descriptors.shape[0] < 65:
            descriptors = cv2.resize(descriptors, (40,65))
            return descriptors.numpy().ravel()
            
        return descriptors[:65].numpy().ravel()




