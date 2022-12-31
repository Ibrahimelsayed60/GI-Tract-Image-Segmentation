import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2
from utils import *
from rle import *
import tensorflow as tf
BATCH_SIZE = 16 

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,  df, batch_size=BATCH_SIZE, shuffle=False) :
        super().__init__()
        self.df =  df
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _load_grayscaled_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img_size = (128,128)
        img = cv2.resize(img,img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img,  axis=-1)
        return img

    def __getitem__(self,  index):
        X = np.empty((self.batch_size, 128, 128, 3))
        y = np.empty((self.batch_size, 128, 128, 3))

        indexes  = self.indexes[index * self.batch_size: (index+1) * self.batch_size]

        for i, img_path in enumerate(self.df['full_path'].iloc[indexes]):
            w = self.df['width'].iloc[indexes[i]]
            h = self.df['height'].iloc[indexes[i]]
            img = self._load_grayscaled_img(img_path)
            X[i] = img
            
            for k,j in enumerate(['large_bowel','small_bowel', 'stomach']):
                rles  = self.df[j].iloc[indexes[i]]
                mask = rle_decode(rles, shape=(h,w,1))
                mask = cv2.resize(mask, (128,128))
                y[i, :, :, k] = mask

        return X,y






