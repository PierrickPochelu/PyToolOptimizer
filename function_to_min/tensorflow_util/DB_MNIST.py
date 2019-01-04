import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import gzip

from function_to_min.tensorflow_util.Abstract_DB import Abstract_DB

class MNIST(Abstract_DB):
    def __init__(self,path):
        Abstract_DB.__init__(self)
        self.path=path

    def onehot(self,labels):
        Y = np.zeros((labels.shape[0], 10))
        Y[np.arange(labels.shape[0]), labels] = 1
        return Y

    def __get_mnist_x_and_y(self, path="."):
        url="http://deeplearning.net/data/mnist/mnist.pkl.gz"
        filepath=path+os.sep+'mnist.pkl.gz'

        # maybe downlad
        if not os.path.exists(filepath):
            urlretrieve(url=url, filename=filepath)

        # read from file
        f = gzip.open(filepath, 'rb')
        train_set, validate_set, test_set = pickle.load(f, encoding="latin1")
        f.close()

        # format data
        X_train = train_set[0]
        Y_train = self.onehot(train_set[1])
        X_test=test_set[0]
        Y_test=self.onehot(test_set[1])

        # process
        X_train=X_train*2.-1.
        X_test=X_test*2.-1.

        return X_train, Y_train,X_test,Y_test


    def get_np_dataset(self):
        dataset = self.__get_mnist_x_and_y(path=".")
        return dataset
    def get_input_shape(self):
        return (28,28,1)
    def get_output_shape(self):
        return (10,)