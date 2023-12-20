import cv2
import tensorflow as tf
import numpy as np


def get_image_tensor(path,rows,cols):
    """process an image in the manner of the VGG16 training set."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(cols, rows))
    img_to_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    processed_image = tf.keras.applications.vgg16.preprocess_input(img_to_tensor)    
    return processed_image



def deprocess_image(x,rows,cols):
    """convert a tensor that has been processed by VGG19 into a valid image by applying the inverse of the preprocessing function."""
    x = x.reshape((rows, cols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
