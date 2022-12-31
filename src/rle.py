import numpy as np
import pandas as pd

def rle_encode(masked_image):
    pixel  = masked_image.flatten()
    pixel = np.concatenate([0], pixel, [0])
    runs = np.where(pixel[1:] != pixel[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    s = mask_rle.split()
    starts, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + length
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)














