import tensorflow as tf
import numpy as np

class TensorflowObject():

    def __init__(self, cnn, cpu = False):
        # new TF session
        if cpu:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            config = tf.ConfigProto()
        self.tf_sess=tf.Session(config=config)

        # build model
        self.cnn = cnn

        # init model
        self.tf_sess.run(tf.global_variables_initializer())

        # compute only once some long computing
        self.layer_std=self.get_glorout_std()
        self.list_shape_layers=self.get_list_shape_layers()
        # compute link between weight<-assign_op<-placeholder<-numpyarray
        self.list_assign_op=[]
        self.list_placeholders = []
        self.__create_assignOP_and_placeholder()

        # compute gradients
        opt = tf.train.GradientDescentOptimizer(learning_rate=1.)
        self.tf_gradients = opt.compute_gradients(self.cnn.loss_model)

    def get_tensor(self, c):
        return tf.get_default_graph().get_tensor_by_name(c + ':0')

    def get_trainable_variables(self):
        return tf.trainable_variables()

    def get_list_shape_layers(self):
        list_layer_shapes=[]
        for i, var in enumerate(tf.trainable_variables()):
            list_layer_shapes.append( [tf_dim.value for tf_dim in var.get_shape()] )
        return list_layer_shapes

    def debug_display_state(self):
        for v in tf.trainable_variables():
            print(v)
            if self.tf_sess is not None:
                print(self.tf_sess.run(v))

    def count_trainable_variables(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def glorout_init_and_get_weights(self):
        self.tf_sess.run(tf.global_variables_initializer())
        w = self.__get_weights_from_model()
        return w

    def get_glorout_std(self):
        array_std_layer=np.zeros((len(tf.trainable_variables()),))
        list_shapes=self.get_list_shape_layers()

        for i,list_var_shape in enumerate(list_shapes):
            n_input=list_var_shape[-2]
            n_output=list_var_shape[-1]
            # if FC 2 dim,
            # if Conv 4 weights.shape=(3,3,n_input,n_output) , n_input=n_input*3*3
            if len(list_var_shape)==4:
                other_dim=list_var_shape[0]*list_var_shape[1]
                n_input*=other_dim
                n_output*=other_dim
            if len(list_var_shape)==3:
                other_dim=list_var_shape[0]
                n_input*=other_dim
                n_output*=other_dim
            #array_std_layer[i]= np.sqrt( 1./(0.5*(n_output+n_input)) )
            array_std_layer[i] = np.sqrt(1. / n_input)

            # for debug purpose
            # print(str(array_variables_variances[i]) +"  "+  str(np.var(self.tf_sess.run(var)))  )
        return array_std_layer


    def close(self):
        self.tf_sess.close()


    def __create_assignOP_and_placeholder(self):
        """
        https://github.com/tensorflow/tensorflow/issues/4151
        :return: nothing
        """

        for variable in tf.trainable_variables():
            placeholder = tf.placeholder(variable.dtype,  variable.get_shape())
            assign_op=tf.assign(variable,placeholder)
            self.list_assign_op.append(assign_op)
            self.list_placeholders.append(placeholder)

    def __set_weights_to_model(self, w):
        # build list_assign_op
        start_i=0
        end_i = 0
        list_assign_op=[]
        for variable,assign_op,placeholder in zip(tf.trainable_variables(),self.list_assign_op,self.list_placeholders):
            # count next i
            variable_parameters = np.prod(variable.get_shape())
            end_i += variable_parameters

            # numpy object
            np_layer=w[start_i:end_i]
            np_layer=np.reshape(np_layer,variable.get_shape())

            # run assign operation
            self.tf_sess.run(assign_op,feed_dict={placeholder:np_layer})

            # for next iteration
            start_i=end_i

        # assign it
        self.tf_sess.run(list_assign_op)

    def __get_weights_from_model(self):
        # build list_assign_op
        start_i=0
        end_i = 0
        nbW=self.count_trainable_variables()
        w=np.zeros((nbW,))

        for variable in tf.trainable_variables():
            # count next i
            variable_parameters = np.prod(variable.get_shape())
            end_i += variable_parameters

            # numpy object
            np_layer=self.tf_sess.run(variable)
            np_layer=np.reshape(np_layer,(variable_parameters,))
            w[start_i:end_i]=np_layer

            # for next iteration
            start_i=end_i
        return w

    def basic_forward(self, w, x):
        # assign weights
        self.__set_weights_to_model(w)

        # forward
        Y_ = self.tf_sess.run(self.cnn.get_forward_model(), feed_dict={self.cnn.get_placeholder_x(): x})
        return Y_

    def forward(self, w, x, batch_size_memory=100):
        y_batch_=np.zeros([x.shape[0],self.basic_forward(w, x[0:1]).shape[1]])
        for i in range(0, x.shape[0], batch_size_memory):
            x_batch= x[i:i + batch_size_memory]
            y_batch_[i:i + batch_size_memory]=self.basic_forward(w, x_batch)
        return y_batch_

    def basic_loss(self, w, x,truey):

        # assign weights
        self.__set_weights_to_model(w)

        # forward
        Y_ = self.tf_sess.run(self.cnn.get_loss_model(), feed_dict={self.cnn.get_placeholder_x(): x,
                                                                    self.cnn.get_placeholder_y(): truey})

        return Y_

    def loss(self, w, x_test, y_test, batch_size_memory=100):
        loss_cumul=0.
        for i in range(0, x_test.shape[0], batch_size_memory):
            x_batch= x_test[i:i + batch_size_memory]
            truey_batch= y_test[i:i + batch_size_memory]
            loss_cumul+= ( self.basic_loss(w,x_batch,truey_batch) * x_batch.shape[0])
        loss_averaged_sample= loss_cumul / x_test.shape[0]
        return loss_averaged_sample

    def basic_gradients(self, w, x,truey):
        # Computing the gradient of cost with respect to W and b
        #test = tf.gradients(xs=self.cnn.get_forward_model(), ys=self.cnn.get_loss_model())
        list_np_gradients = self.tf_sess.run(self.tf_gradients, feed_dict={
            self.cnn.get_placeholder_x(): x,
            self.cnn.get_placeholder_y(): truey})

        # flat gradients
        np_gradient_flat=np.zeros((self.count_trainable_variables()))
        i_start=0
        i_end=0
        for layer_grd,_ in list_np_gradients:
            nb_weights_layer=np.prod(layer_grd.shape)
            nplayer=np.reshape(layer_grd,nb_weights_layer)
            i_end+=nb_weights_layer
            np_gradient_flat[i_start:i_end] = nplayer
            i_start=i_end

        return np_gradient_flat

    def gradients(self, w, x_test, y_test, batch_size_memory=100):
        # set weights
        self.__set_weights_to_model(w)

        # compute gradients
        grads_cumul = 0.
        for i in range(0, x_test.shape[0], batch_size_memory):
            x_batch = x_test[i:i + batch_size_memory]
            truey_batch = y_test[i:i + batch_size_memory]
            grads_batch=self.basic_gradients(w, x_batch, truey_batch)
            if i==0:
                grads_cumul = (grads_batch * x_batch.shape[0])
            else:
                grads_cumul += (grads_batch * x_batch.shape[0])
            grads_cumul += (grads_batch * x_batch.shape[0])
        loss_averaged_sample = grads_cumul / x_test.shape[0]
        return loss_averaged_sample