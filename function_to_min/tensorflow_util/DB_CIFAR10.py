import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys

from function_to_min.tensorflow_util.Abstract_DB import Abstract_DB
class CIFAR10(Abstract_DB):
    def __init__(self,path):
        Abstract_DB.__init__(self)
        self.path=path

    def __get_cifar10_toronto_x_and_y(self,name="train", path="."):
        x = None
        y = None

        self.__maybe_download_and_extract(path)

        folder_name = "cifar_10"
        f = open(path+'/data_set/'+folder_name+'/batches.meta', 'rb')
        f.close()

        if name is "train":
            for i in range(5):
                f = open(path+'/data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
                datadict = pickle.load(f, encoding='latin1')
                f.close()

                _X = datadict["data"]
                _Y = datadict['labels']

                _X = np.array(_X, dtype=float) / 255.0
                _X = _X.reshape([-1, 3, 32, 32])
                _X = _X.transpose([0, 2, 3, 1])
                _X = _X.reshape(-1, 32*32*3)

                if x is None:
                    x = _X
                    y = _Y
                else:
                    x = np.concatenate((x, _X), axis=0)
                    y = np.concatenate((y, _Y), axis=0)

        elif name is "test":
            f = open(path+'/data_set/'+folder_name+'/test_batch', 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            x = datadict["data"]
            y = np.array(datadict['labels'])

            x = np.array(x, dtype=float) / 255.0
            x = x.reshape([-1, 3, 32, 32])
            x = x.transpose([0, 2, 3, 1])
            x = x.reshape(-1, 32*32*3)

        return x, self.__dense_to_one_hot(y)


    def __dense_to_one_hot(self,labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot


    def _print_download_progress(self,count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()


    def __maybe_download_and_extract(self,path="."):
        main_directory = path+"/data_set/"
        cifar_10_directory = main_directory+"cifar_10/"
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)

            url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            filename = url.split('/')[-1]
            file_path = os.path.join(main_directory, filename)
            zip_cifar_10 = file_path
            file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=self._print_download_progress)

            print()
            print("Download finished. Extracting files.")
            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
            print("Done.")

            os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
            os.remove(zip_cifar_10)

    def get_np_dataset(self):
        X_train, Y_train = self.__get_cifar10_toronto_x_and_y(name="train", path=self.path)
        X_test, Y_test = self.__get_cifar10_toronto_x_and_y(name="test", path=self.path)

        X_train=X_train.reshape((len(X_train),32,32,3)).astype(np.float32)
        X_test=X_test.reshape((len(X_test),32,32,3)).astype(np.float32)

        return (X_train,Y_train,X_test,Y_test)

    def get_input_shape(self):
        return (32,32,3)
    def get_output_shape(self):
        return (10,)
