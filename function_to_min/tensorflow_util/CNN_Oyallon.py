from function_to_min.tensorflow_util.Abstract_CNN import CNN
import tensorflow as tf
import numpy as np

# TODO : secret link below
#https://anonymous.4open.science/repository/75115ffe-d110-4814-ade0-060dd12b9f7e/
# Article :
# shallow deep learning (2019)

class CNN_oyallon(CNN):
    def __init__(self,input_shape,output_shape,K,J,M,M_,bigger_and_bigger=True):
        """

        :param input_shape: tuple ex : (32,32,3) for cifar10
        :param output_shape: tuple ex : (10,) for cifar10
        """
        CNN.__init__(self)
        use_bias=False

        # instantiate
        self.current_j=0
        self.placeholder_x = tf.placeholder(dtype=tf.float32, shape=(None,  np.prod(input_shape)), name="x")
        self.placeholder_y= tf.placeholder(dtype=tf.float32, shape=(None, output_shape[0]), name="y")
        self.epsilon = tf.constant(value=1e-15, shape=output_shape)
        x = tf.reshape(self.placeholder_x, shape=[-1, input_shape[0], input_shape[1], input_shape[2]])

        def nb_filter(M,j,bigger_and_bigger):
            if not bigger_and_bigger:
                return M
            else:
                return M*2**j

        # construct model
        def K_block(x,j):
            for k_id in range(K-1):
                x = tf.layers.conv2d(x, nb_filter(M_,j,bigger_and_bigger), 3, activation=tf.nn.relu, strides=1, use_bias=use_bias, padding='same')
            x = tf.layers.average_pooling2d(x, 3, 2)#TODO : spatial averaging operator
            x = tf.layers.flatten(x)
            predictions_of_block = tf.layers.dense(x, output_shape[0], activation=tf.nn.softmax, use_bias=use_bias)
            return predictions_of_block
        def J_block(x,j):
            x = tf.layers.conv2d(x, nb_filter(M,j,bigger_and_bigger), 3, activation=tf.nn.relu, strides=1, use_bias=use_bias, padding='same')
            x = tf.layers.max_pooling2d(x, 3, 2)#TODO : invertible down sampling
            return x
        def cross_entropy_loss(out):
            f_with_cross_entropy = -tf.reduce_mean(
                self.placeholder_y * tf.log(out + self.epsilon) +
                (1. - self.placeholder_y) * tf.log(1. - out + self.epsilon),
                axis=1)
            loss = tf.reduce_mean(f_with_cross_entropy, keep_dims=True)[0]
            return loss


        def oyallon_ensemble(ordered_predictions_ensemble,ensemble_index):
            """
            :param ordered_predictions_ensemble: predictions of J blocks
            :param ensemble_index: <=J
            :return:
            """

            compute_value_to_divide = np.sum( [2**j for j in range(ensemble_index+1) ]  )

            # ref : page 5 chapter 4
            # 2 puissance 0
            ensemble_predictions = ordered_predictions_ensemble[0]

            # 2 puissance j
            for j in range(1,ensemble_index+1):
                ensemble_predictions += (2**j) * ordered_predictions_ensemble[j] # TODO : warning article begin start with j=1

            return ensemble_predictions / compute_value_to_divide

        # create outputs ALONE mode
        self.forward_model_alone=[]
        for j in range(J):
            next_x=J_block(x,j)
            pred=K_block(next_x,j)
            self.forward_model_alone.append(pred)
            x=next_x

        self.loss_model_alone=[]
        for j in range(J):
            out_j=self.forward_model_alone[j]
            loss_alone_j=cross_entropy_loss(out_j)
            self.loss_model_alone.append(loss_alone_j)



        # create outputs ENSEMBLE mode
        ordered_predictions_ensemble = []
        for j in range(J):
            ensemble_j_outputs=oyallon_ensemble(self.forward_model_alone,j)
            ordered_predictions_ensemble.append(ensemble_j_outputs)

        # create loss ENSEMBLE
        ordered_losses_ensemble = []
        for j in range(J):
            ensemble_j_loss=cross_entropy_loss(ordered_predictions_ensemble[j])
            ordered_losses_ensemble.append(ensemble_j_loss)


        self.forward_model = ordered_predictions_ensemble
        self.loss_model = ordered_losses_ensemble

    def set_current_j(self,current_j):
        self.current_j=current_j

    def get_loss_model(self):
        return self.loss_model[self.current_j]

    def get_forward_model(self):
        return self.forward_model[self.current_j]

    def get_forward_model_debug(self):
        return self.forward_model_alone

    def get_loss_alone_model(self):
        return self.loss_model_alone[self.current_j]