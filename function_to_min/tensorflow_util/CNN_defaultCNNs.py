from function_to_min.tensorflow_util.Abstract_CNN import CNN
import tensorflow as tf
import numpy as np



class CNN_2conv(CNN):
    def __init__(self,input_shape,output_shape):
        """
        :param input_shape: tuple ex : (32,32,3) for cifar10
        :param output_shape: tuple ex : (10,) for cifar10
        """
        CNN.__init__(self)

        activ=tf.nn.relu
        bn=False
        bias = False

        self.placeholder_y = tf.placeholder(dtype=tf.float32, shape=(None, output_shape[0]), name="y")
        labels = tf.reshape(self.placeholder_y, [-1, output_shape[0]])

        self.placeholder_x = tf.placeholder(dtype=tf.float32, shape=(None, np.prod(input_shape)), name="x")
        x = tf.reshape(self.placeholder_x, shape=[-1, input_shape[0],input_shape[1],input_shape[2]])


        # FIRST CNN
        x = tf.layers.conv2d(inputs=x, use_bias=bias, filters=32, kernel_size=[5, 5], activation=activ,
                             strides=2, padding='same')

        # SECOND CNN
        x = tf.layers.conv2d(inputs=x, use_bias=bias, filters=32, kernel_size=[5, 5], activation=activ,
                             strides=2, padding='same')

        # x=tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = tf.layers.flatten(x)
        x = tf.layers.dense(inputs=x, activation=activ, use_bias=bias, units=64)
        x_last = x
        y_ = tf.layers.dense(inputs=x_last, activation=tf.nn.softmax, use_bias=bias, units=10)

        y = tf.cast(labels, dtype=tf.float32)
        epsilon = tf.constant(value=1e-15, shape=output_shape)
        f_with_cross_entropy = -tf.reduce_mean(y * tf.log(y_ + epsilon) + (1. - y) * tf.log(1. - y_ + epsilon),
                                               axis=1)
        loss = tf.reduce_mean(f_with_cross_entropy, keep_dims=True)[0]

        self.loss_model=loss
        self.forward_model=y_

class CNN_3conv(CNN):
    def __init__(self,input_shape,output_shape):
        """

        :param input_shape: tuple ex : (32,32,3) for cifar10
        :param output_shape: tuple ex : (10,) for cifar10
        """
        CNN.__init__(self)
        use_bias=False

        # construct model
        num_output=10
        self.placeholder_x = tf.placeholder(dtype=tf.float32, shape=(None,  np.prod(input_shape)), name="x")

        x = tf.reshape(self.placeholder_x, shape=[-1, input_shape[0], input_shape[1], input_shape[2]])

        x = tf.layers.conv2d(x, 16, 3, activation=tf.nn.relu, strides=1, use_bias=use_bias, padding='same')
        x = tf.layers.max_pooling2d(x, 3, 2)
        x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, strides=1, use_bias=use_bias, padding='same')
        x = tf.layers.max_pooling2d(x, 3, 2)
        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, strides=1, use_bias=use_bias, padding='same')
        x = tf.layers.max_pooling2d(x, 3, 2)

        flat = tf.layers.flatten(x)
        fc1 = tf.layers.dense(flat, 64, activation=tf.nn.relu, use_bias=use_bias)
        out = tf.layers.dense(fc1, output_shape[0], activation=tf.nn.softmax, use_bias=use_bias)
        self.forward_model = out

        # add loss layers
        self.placeholder_y= tf.placeholder(dtype=tf.float32, shape=(None, 10), name="y")



        epsilon = tf.constant(value=1e-15, shape=output_shape)
        f_with_cross_entropy = -tf.reduce_mean(self.placeholder_y * tf.log(out + epsilon) +
                                               (1. - self.placeholder_y) * tf.log(1. - out + epsilon), axis=1)
        loss = tf.reduce_mean(f_with_cross_entropy, keep_dims=True)[0]

        self.loss_model = loss