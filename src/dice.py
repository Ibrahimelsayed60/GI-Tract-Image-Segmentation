from tensorflow import keras
import tensorflow as tf
import keras
from keras.models import load_model, save_model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.models import Model
from keras.layers import Input


def dice_coef(y_true, y_pred, smooth=1):
    y_true_flattened = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flattened * y_pred_flatten)
    union = K.sum(y_true_flattened) + K.sum(y_pred_flatten)
    return (2.0 * intersection + smooth) / union + smooth


def dice_loss(y_true, y_pred):
    smooth = 1
    y_true_flattened = y_true.flatten()
    y_pred_flattened = y_pred.flatten()
    intersection = y_true_flattened * y_pred_flattened
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_flattened) + K.sum(y_pred_flattened) + smooth
    )
    return 1.0 - score


def iou_coef(y_true, y_pred, smooth):
    intersection = K.sum(K.abs(y_true, *y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3] + K.sum(y_pred, [1, 2, 3])) - intersection
    iou = K.mean((intersection) + smooth / (union + smooth), axis=0)
    return iou


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(
        tf.cast(y_true, tf.float32), y_pred
    )