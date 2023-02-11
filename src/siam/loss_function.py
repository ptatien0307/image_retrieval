import tensorflow as tf

def triplet_loss(y_true, y_pred):
    alpha = 0.2

    anchor, positive, negative = y_pred[:,:256], y_pred[:,256:2*256], y_pred[:,2*256:]


    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    loss = tf.maximum(0., alpha + positive_dist - negative_dist)

    loss = tf.reduce_mean(loss)

    return loss

def contrastive_loss(y_true, y_pred):
    y = tf.cast(y_true, y_pred.dtype)
    image_1, image_2 = y_pred[:,:2048], y_pred[:,2048:]

    dist = tf.reduce_sum(tf.square(image_1 - image_2), axis=1)
    
    loss = 0.5 * (1 - y) * tf.square(tf.maximum(0., 0.2 - dist)) +  0.5 * y * dist

    loss = tf.reduce_mean(loss)
    
    return loss