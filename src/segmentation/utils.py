import random

import numpy as np

class Batchgen(object):
    """generate batches
    data is a list of the class Data
    batch_size is int type
    """
    def __init__(self, X, y, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.size = len(y)
        if shuffle:
            indices = list(range((self.size)))
            random.shuffle(indices)
            self.X = [X[i] for i in indices]
            self.y = [y[i] for i in indices]

        self.batches = [(self.X[i:i + batch_size], self.y[i:i + batch_size]) for i in range(0, self.size, batch_size)]
        

    def __len__(self):
        return self.size

    def __iter__(self):
        for batch in self.batches:
            yield batch

        raise StopIteration

def normalize(img):
    max_ = np.max(img)
    img = img / max_
    return img

def centralize(img):
    img = img.astype('float64')
    mean = np.mean(img)
    img -= mean
    return img



