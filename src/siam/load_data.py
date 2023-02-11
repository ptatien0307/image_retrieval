import numpy as np
import random
random.seed(25)

def create_triplets(batch_size, x_train, y_train):
    """
        Create triplet batch for training 
    """

    x_anchors = np.zeros((batch_size, 224,224,3))
    x_positives = np.zeros((batch_size, 224,224,3))
    x_negatives = np.zeros((batch_size, 224,224,3))
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, len(x_train) - 1)

        x_anchor = x_train[random_index]
        y = y_train[random_index]
        
        indices_for_pos = np.squeeze(np.where(y_train == y))
        indices_for_neg = np.squeeze(np.where(y_train != y))
        
        x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
        
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        
    return [x_anchors, x_positives, x_negatives]

def create_pairs(batch_size, x_train, y_train):
    """
        Create triplet batch for training 
    """

    x_image1s = np.zeros((batch_size, 224,224,3))
    x_image2s = np.zeros((batch_size, 224,224,3))
    y = []
    for i in range(0, batch_size//2):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, len(x_train) - 1)

        x_anchor = x_train[random_index]
        y = y_train[random_index]
        
        indices_for_pos = np.squeeze(np.where(y_train == y))
        indices_for_neg = np.squeeze(np.where(y_train != y))
        
        x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
        
        x_image1s[i] = x_anchor
        x_image2s[i] = x_positive
        x_image2s[i+1] = x_negative
        
        y.append(1)
        y.append(0)

    return [x_image1s, x_image2s], np.array(y)

def data_generator(batch_size, x_train, y_train):
    name = 'pairs'
    while True:
        if name == 'triplet':
            x = create_triplets(batch_size, x_train, y_train)
            y = np.zeros((batch_size, 3*64))
        else:
            x, y = create_pairs(batch_size, x_train, y_train)
        yield x, y