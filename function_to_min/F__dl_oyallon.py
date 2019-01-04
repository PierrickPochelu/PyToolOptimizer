import numpy as np
from numpy.linalg import inv
from function_to_min.tensorflow_util.TensorflowObject import TensorflowObject
from function_to_min.tensorflow_util.CNN_defaultCNNs import CNN_2conv

from function_to_min import InterfaceFunctionToMin
#from util import util
import tensorflow as tf

class Deeplearning(InterfaceFunctionToMin):
    def __init__(self, simpleCNN, batch_size_memory, batch_size_algo, np_dataset, mini_batch_mode=False):
        InterfaceFunctionToMin.__init__(self)

        # LOAD DATA
        self.X_train, self.Y_train, self.X_test, self.Y_test = np_dataset

        # CREATE OBJECT WHICH ABSTRACT TENSORFLOW COMPLEXITY
        self.tensorflowObject = TensorflowObject(simpleCNN, cpu=False)

        # activation
        self.list_activation_layer = [np.ones(np.prod(shape)) for shape in self.tensorflowObject.list_shape_layers]

        # batch management
        self.mini_batch_mode=mini_batch_mode
        self.batch_size_memory = batch_size_memory
        self.batch_size_algo = batch_size_algo
        self.i=0
        self.__next_batch_id=0
        self.cur_batch_start_id=0
        self.cur_batch_end_id=0
        self.next_batch_sometime_shuffle()

        self.oyallon_j=0

    def f(self, w):
        """
        :param w: weights of the neural network
        :return:  loss of the current train batch
        """
        data_batch= self.X_train[self.cur_batch_start_id:self.cur_batch_end_id]
        labels_batch = self.Y_train[self.cur_batch_start_id:self.cur_batch_end_id]
        loss = self.tensorflowObject.loss(w, data_batch, labels_batch, self.batch_size_memory)

        # change batch for next time ?
        if self.mini_batch_mode:
            self.next_batch_sometime_shuffle()
        return loss

    def df(self,w):
        data_batch= self.X_train[self.cur_batch_start_id:self.cur_batch_end_id]
        labels_batch = self.Y_train[self.cur_batch_start_id:self.cur_batch_end_id]
        np_gradients = self.tensorflowObject.gradients(w,data_batch,labels_batch,self.batch_size_memory)

        # change batch for next time ?
        if self.mini_batch_mode:
            self.next_batch_sometime_shuffle()

        return np_gradients

    def ddf(self,x):
        return None

    def __get_size_filt_or_colu(self, shape):
        if len(shape)==4:
            return shape[0]*shape[1]*shape[2]
        elif len(shape)==2:
            return shape[0]
        else :
            print("Error get_size_filt_or_colu()")
            return None

    def set_J(self,J):
        """
        Update activation_vec.
        :param id_layer: if -1 all the network is trained. Otherwise select layer id.
        :param id_filter:  if -1 all the layer is trained. Otherwise select filter id.
        :return:
        """
        self.tensorflowObject.cnn.set_current_j(J)

    def get_effort_training(self):
        """
        :return:
        list_effort_layer[id_layer] : for each layer nb of parameters
        list_effort_filter(nb_filters, weights_filter) : for each layer 1) nb filters 2) filter size
        """
        list_effort_network=self.get_nb_variables()
        list_effort_layer=[]
        list_effort_filter=[]
        for i in range(len(self.list_activation_layer)):
            size_filt=self.__get_size_filt_or_colu(self.tensorflowObject.list_shape_layers[i])
            nb_filt=self.tensorflowObject.list_shape_layers[i][-1]
            list_effort_filter.append([nb_filt , size_filt])
            list_effort_layer.append(nb_filt*size_filt)
        return list_effort_network,list_effort_layer, list_effort_filter

    def next_batch_sometime_shuffle(self):
        # compute id this batch
        self.cur_batch_start_id = self.__next_batch_id * self.batch_size_algo
        self.cur_batch_end_id = np.min([(self.__next_batch_id + 1) * self.batch_size_algo, self.X_train.shape[0]])

        # compute next batch id
        if self.mini_batch_mode:
            if self.cur_batch_end_id==self.X_train.shape[0]:
                self.__next_batch_id=0
                # shuffle train data
                self.__shuffle()
            else:
                self.__next_batch_id+=1


    def __shuffle(self):
        ids=np.array( range(self.X_train.shape[0]) )
        np.random.shuffle(ids)
        self.X_train=self.X_train[ids]
        self.Y_train=self.Y_train[ids]

    def accuracy(self, w):
        output=self.tensorflowObject.forward(w, self.X_test, batch_size_memory=self.batch_size_memory)
        output_nb=np.argmax( output ,axis=1)
        y_nb=np.argmax(self.Y_test , axis=1)
        nb_good_answers=np.sum( output_nb == y_nb )
        acc=nb_good_answers / output.shape[0]
        return acc

    def loss_testDataset(self,w):
        output=self.tensorflowObject.loss(w, self.X_test,self.Y_test, batch_size_memory=self.batch_size_memory)
        return output

    def loss_trainDataset(self,w):
        output=self.tensorflowObject.loss(w, self.X_train,self.Y_train, batch_size_memory=self.batch_size_memory)
        return output

    def loss_alone_trainDataset_debug(self,w):
        output=self.tensorflowObject.loss_alone(w, self.X_train,self.Y_train, batch_size_memory=self.batch_size_memory)
        return output

    def glorout_init_and_get_weights(self):
        return self.tensorflowObject.glorout_init_and_get_weights()

    def forward_init_and_get_weights(self):
        w0=np.zeros(self.get_nb_variables())
        return self.apriori_nn(w0,1.)

    def apriori_nn(self,w,sigma):
        start_i=0
        end_i = 0
        list_variance_layers=self.tensorflowObject.layer_std
        list_shape_layers=self.tensorflowObject.get_list_shape_layers()

        w2=np.zeros(w.shape)
        for std_layer,shape_layer,A in zip(list_variance_layers, list_shape_layers, self.list_activation_layer):

            nb_values=np.prod(shape_layer)
            end_i+=nb_values

            """
            if is_trainable:
                apriori_layer=np.random.standard_t(dof,(nb_values,))*std_layer*sigma + w[start_i:end_i]
                #apriori_layer = np.random.normal(w[start_i:end_i] ,std_layer*sigma, (nb_values,))
            else: # forward
                apriori_layer=w[start_i:end_i]
            """
            #apriori_layer = (np.random.standard_t(dof, (nb_values,)) * A) * std_layer * sigma + w[start_i:end_i]
            delta=(np.random.normal(0,scale=std_layer*sigma , size=(nb_values,)) * A)
            #delta=np.random.normal(0,scale=1.,size=(1,))*delta
            apriori_layer = delta + w[start_i:end_i]

            w2[start_i:end_i]=apriori_layer

            start_i=end_i
        return w2

    def set_callback_after_batch(self,callback_after_batch):
        self.callback_after_batch=callback_after_batch

    def get_nb_variables(self):
        return self.tensorflowObject.count_trainable_variables()

    def get_nb_layers(self):
        return len(self.tensorflowObject.get_trainable_variables())

    def freeze_layer(self, list_trainable_variables):
        """
        :param list_trainable_variables: list of boolean. When a layer is freezed it is no longer update
        :return: nothing
        """
        assert(self.get_nb_layers()==len(list_trainable_variables))
        self.list_istrainable_layers=list_trainable_variables

    def close(self):
        self.tensorflowObject.close()