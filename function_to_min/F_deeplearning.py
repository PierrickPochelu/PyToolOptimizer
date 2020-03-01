import numpy as np
from numpy.linalg import inv
import tensorflow as tf
from tensorflow.keras import layers,models

from function_to_min import  InterfaceFunctionToMin

class Deeplearning(InterfaceFunctionToMin):
    def __init__(self, np_dataset,bs=32):

        # LOAD DATA
        self.X_train, self.Y_train, self.X_test, self.Y_test = np_dataset

        self.batch_size=bs

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10,activation='softmax'))
        model.compile(optimizer='sgd',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model=model


        self.cur_batch_start_id,self.cur_batch_end_id=0,self.batch_size

    def get_w(self):
        tf_w=self.model.trainable_weights
        return [w.numpy() for w in tf_w]

    def set_w(self,w):
        tf_weights=self.model.trainable_weights
        for tf_w_old, np_w_new in zip(tf_weights,w):
            tf_w_old.assign(np_w_new)

    def f(self, w):
        """
        :param w: weights of the neural network
        :return:  loss of the current train batch
        """
        data_batch= self.X_train[self.cur_batch_start_id:self.cur_batch_end_id]
        labels_batch = self.Y_train[self.cur_batch_start_id:self.cur_batch_end_id]
        loss, acc = self.model.evaluate(data_batch, labels_batch,verbose=0)


        return loss

    def tf_loss(self,x, y):
        y_=self.model(x)
        ce=tf.losses.categorical_crossentropy(y,y_)
        return tf.reduce_mean(ce)

    def df(self,w):
        x= self.X_train[self.cur_batch_start_id:self.cur_batch_end_id]
        y = self.Y_train[self.cur_batch_start_id:self.cur_batch_end_id]

        with tf.GradientTape() as tape:
            loss=self.tf_loss(x,y)
        weights = self.model.trainable_variables
        tf_gradients=tape.gradient(loss, weights)

        np_gradients=[ tf_grad.numpy() for tf_grad in tf_gradients]
        #np_gradients = tf.keras.backend.gradients( loss , weights )
        #np_gradients = tf.GradientTape()

        return np_gradients

    #def ddf(self,x):#https://www.programcreek.com/python/example/93762/keras.backend.gradients
    def ddf(self,w):
        x= self.X_train[self.cur_batch_start_id:self.cur_batch_end_id]
        y = self.Y_train[self.cur_batch_start_id:self.cur_batch_end_id]

        weights = self.model.trainable_variables

        """
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                loss=self.tf_loss(x,y)
            tf_gradients1=tape1.gradient(loss, weights)
        tf_gradients2=tape2.gradient(loss,tf_gradients1)
        """
        with tf.GradientTape(persistent=True) as tape:
            loss=self.tf_loss(x,y)
            tf_gradients1=tape.gradient(loss, weights)
        tf_gradients2=tape.gradient(tf_gradients1, weights)

        #tf.hessians()
        np_gradients=[ tf_grad.numpy() for tf_grad in tf_gradients2]

        return np_gradients


    def shuffle(self):
        ids=np.array( range(self.X_train.shape[0]) )
        np.random.shuffle(ids)
        self.X_train=self.X_train[ids]
        self.Y_train=self.Y_train[ids]


    def next_batch(self):
        self.cur_batch_start_id=self.cur_batch_start_id+self.batch_size

        if self.cur_batch_end_id >= len(self.X_train) and self.cur_batch_start_id >= len(self.X_train):# need to resest
            self.cur_batch_start_id=0

        self.cur_batch_end_id=self.cur_batch_start_id+self.batch_size

    def accuracy(self, w):
        acc, _ = self.model.evaluate(self.X_test, self.Y_test, batch_size=self.batch_size)
        return acc

    def accuracy_train(self, w):
        acc, _ = self.model.evaluate(self.X_train, self.Y_train, batch_size=self.batch_size)
        return acc
