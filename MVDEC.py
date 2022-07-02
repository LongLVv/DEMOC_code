from keras import backend as K
import numpy as np
import tensorflow as tf
# #-----------------------------Keras reproducible------------------#
# SEED = 1234
#
# tf.set_random_seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# np.random.seed(SEED)
# rn.seed(SEED)
#
# session_conf = tf.ConfigProto(
#     intra_op_parallelism_threads=1,
#     inter_op_parallelism_threads=1
# )
# sess = tf.Session(
#     graph=tf.get_default_graph(),
#     config=session_conf
# )
# K.set_session(sess)
from time import time
#
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.layers import Dense, Input, GaussianNoise, Layer, Activation, InputSpec
# from keras.layers import Dense, Input, GaussianNoise, Layer, Activation,InputSpec
from keras.models import Model
from keras import callbacks
from sklearn.cluster import KMeans
import Nmetrics
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def MAE(view=2, noise_sd=2.5, init='glorot_uniform', filters1=[2000, 256, 64, 32], filters2=[2000, 256, 64, 32],
        filters3=[2000, 256, 64, 32], act="relu"):  # 10 14 input_dim
    if view == 2:
        n_stack1 = len(filters1) - 1
        n_stack2 = len(filters2) - 1
        # view1
        sf_layer = Input(shape=(1,), name='size_factors')
        input1 = Input((filters1[0],), name='input1')
        h1 = input1
        h1 = GaussianNoise(noise_sd, name='input_noise')(h1)

        for i in range(n_stack1 - 1):
            h1 = Dense(filters1[i + 1], activation=act, kernel_initializer=init, name='view1_encoder_%d' % i)(h1)
            h1 = GaussianNoise(noise_sd, name='noise_%d' % i)(h1)  # add Gaussian noise
            h1 = Activation(act)(h1)
        ### hidden layer
        h1 = Dense(filters1[-1], kernel_initializer=init, name="hidden1")(h1)  ## act是否保留待定
        for i in range(n_stack1 - 1, 0, -1):
            h1 = Dense(filters1[i], activation=act, kernel_initializer=init, name='view1_decoder_%d' % i)(h1)
        pi = Dense(filters1[0], activation='sigmoid', kernel_initializer=init, name='pi')(h1)
        disp = Dense(filters1[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h1)
        mean = Dense(filters1[0], activation=MeanAct, kernel_initializer=init, name='mean')(h1)
        output = ColWiseMultLayer(name='output')([mean, sf_layer])
        output = SliceLayer(0, name='slice')([output, disp, pi])

        # view2
        input2 = Input((filters2[0],), name='input2')
        h2 = input2
        for i in range(n_stack2 - 1):
            h2 = Dense(filters2[i + 1], activation=act, kernel_initializer=init, name='view2_encoder_%d' % i)(h2)
        ### hidden layer
        h2 = Dense(filters2[-1], kernel_initializer=init, name="hidden2")(h2)  ## act
        hidden2 = h2

        for i in range(n_stack2 - 1, -1, -1):
            h2 = Dense(filters2[i], activation=act, kernel_initializer=init, name='view2_decoder_%d' % i)(h2)
        ae1 = Model(inputs=[input1, sf_layer], outputs=[output])
        encoder2 = Model(inputs=input2, outputs=hidden2)
        ae2 = Model(inputs=input2, outputs=h2)
        return ae1, ae2, encoder2
    else:
        n_stack1 = len(filters1) - 1
        n_stack2 = len(filters2) - 1
        n_stack3 = len(filters3) - 1
        # view1
        sf_layer = Input(shape=(1,), name='size_factors')
        input1 = Input((filters1[0],), name='input1')
        h1 = input1
        h1 = GaussianNoise(noise_sd, name='input_noise')(h1)

        for i in range(n_stack1 - 1):
            h1 = Dense(filters1[i + 1], activation=act, kernel_initializer=init, name='view1_encoder_%d' % i)(h1)
            h1 = GaussianNoise(noise_sd, name='noise_%d' % i)(h1)  # add Gaussian noise
            h1 = Activation(act)(h1)
        ### hidden layer
        h1 = Dense(filters1[-1], kernel_initializer=init, name="hidden1")(h1)  ## act是否保留待定
        for i in range(n_stack1 - 1, 0, -1):
            h1 = Dense(filters1[i], activation=act, kernel_initializer=init, name='view1_decoder_%d' % i)(h1)
        pi = Dense(filters1[0], activation='sigmoid', kernel_initializer=init, name='pi')(h1)
        disp = Dense(filters1[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h1)
        mean = Dense(filters1[0], activation=MeanAct, kernel_initializer=init, name='mean')(h1)
        output = ColWiseMultLayer(name='output')([mean, sf_layer])
        output = SliceLayer(0, name='slice')([output, disp, pi])

        # view2
        input2 = Input((filters2[0],), name='input2')
        h2 = input2
        for i in range(n_stack2 - 1):
            h2 = Dense(filters2[i + 1], activation=act, kernel_initializer=init, name='view2_encoder_%d' % i)(h2)
        ### hidden layer
        h2 = Dense(filters2[-1], kernel_initializer=init, name="hidden2")(h2)  ## act
        hidden2 = h2
        for i in range(n_stack2 - 1, -1, -1):
            h2 = Dense(filters2[i], activation=act, kernel_initializer=init, name='view2_decoder_%d' % i)(h2)

        # view3
        input3 = Input((filters3[0],), name='input3')
        h3 = input3
        for i in range(n_stack3 - 1):
            h3 = Dense(filters3[i + 1], activation=act, kernel_initializer=init, name='view3_encoder_%d' % i)(h3)
        ### hidden layer
        h3 = Dense(filters3[-1], kernel_initializer=init, name="hidden3")(h3)  ## act
        hidden3 = h3
        for i in range(n_stack3 - 1, -1, -1):
            h3 = Dense(filters3[i], activation=act, kernel_initializer=init, name='view3_decoder_%d' % i)(h3)

        ae1 = Model(inputs=[input1, sf_layer], outputs=[output])
        encoder2 = Model(inputs=input2, outputs=hidden2)
        ae2 = Model(inputs=input2, outputs=h2)
        encoder3 = Model(inputs=input3, outputs=hidden3)
        ae3 = Model(inputs=input3, outputs=h3)
        return ae1, ae2, encoder2, ae3, encoder3


class ClusteringLayer(Layer):
    """
    Clustering layer
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha1 = alpha
        self.alpha2 = alpha
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2  # Assertion if not, raises an exception
        input_dim = input_shape[1]  # input_dim = input_shape.as_list()[1]  input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MvDEC(object):
    def __init__(self,
                 input1_shape, input2_shape, input3_shape=2000,
                 filters=[32, 64, 128, 10], view=2,
                 n_clusters=7, noise_sd=2.5, ridge=0, debug=False,
                 alpha=1.0):

        super(MvDEC, self).__init__()

        self.input1_shape = input1_shape
        self.input2_shape = input2_shape
        self.input3_shape = input3_shape
        self.filters = filters
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        # prepare MvDEC model
        self.view = view
        self.ridge = ridge
        self.debug = debug
        self.noise_sd = noise_sd
        if self.view == 2:
            self.ae1, self.ae2, self.encoder2 = MAE(view=self.view, noise_sd=self.noise_sd,
                                                    filters1=[self.input1_shape, 256, 64, 32],
                                                    filters2=[self.input2_shape, 256, 64, 32])
            ae1_layers = [l for l in self.ae1.layers]
            hidden1 = self.ae1.input[0]
            for i in range(1, len(ae1_layers)):
                if "noise" in ae1_layers[i].name:
                    next
                elif "dropout" in ae1_layers[i].name:
                    next
                else:
                    hidden1 = ae1_layers[i](hidden1)
                if "hidden1" in ae1_layers[i].name:  # only get encoder layers
                    break

            self.encoder1 = Model(inputs=self.ae1.input, outputs=hidden1)

            pi = self.ae1.get_layer(name='pi').output
            disp = self.ae1.get_layer(name='dispersion').output
            mean = self.ae1.get_layer(name='mean').output
            zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
            self.loss = zinb.loss

            self.autoencoder = Model(inputs=[self.ae1.input[0], self.ae1.input[1], self.ae2.input],
                                     outputs=[self.ae1.output, self.ae2.output])
            self.encoder = Model(inputs=[self.encoder1.input[0], self.encoder1.input[1], self.encoder2.input],
                                 outputs=[self.encoder1.output,
                                          self.encoder2.output])
            clustering_layer1 = ClusteringLayer(self.n_clusters, name='clustering1')(self.encoder1.output)
            clustering_layer2 = ClusteringLayer(self.n_clusters, name='clustering2')(self.encoder2.output)
            self.model = Model(inputs=[self.ae1.input[0], self.ae1.input[1], self.ae2.input],
                               outputs=[clustering_layer1, self.ae1.output,
                                        clustering_layer2, self.ae2.output])
        if self.view == 3:
            self.ae1, self.ae2, self.encoder2, self.ae3, self.encoder3 = MAE(view=self.view, noise_sd=self.noise_sd,
                                                                             filters1=[self.input1_shape, 256, 64, 32],
                                                                             filters2=[self.input2_shape, 8, 4],
                                                                             filters3=[self.input3_shape, 256, 64, 32])
            ae1_layers = [l for l in self.ae1.layers]
            hidden1 = self.ae1.input[0]
            for i in range(1, len(ae1_layers)):
                if "noise" in ae1_layers[i].name:
                    next
                elif "dropout" in ae1_layers[i].name:
                    next
                else:
                    hidden1 = ae1_layers[i](hidden1)
                if "hidden1" in ae1_layers[i].name:  # only get encoder layers
                    break

            self.encoder1 = Model(inputs=self.ae1.input, outputs=hidden1)

            pi = self.ae1.get_layer(name='pi').output
            disp = self.ae1.get_layer(name='dispersion').output
            mean = self.ae1.get_layer(name='mean').output
            zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
            self.loss = zinb.loss

            self.autoencoder = Model(inputs=[self.ae1.input[0], self.ae1.input[1], self.ae2.input, self.ae3.input],
                                     outputs=[self.ae1.output, self.ae2.output, self.ae3.output])
            self.encoder = Model(
                inputs=[self.encoder1.input[0], self.encoder1.input[1], self.encoder2.input, self.encoder3.input],
                outputs=[self.encoder1.output,
                         self.encoder2.output, self.encoder3.output])
            clustering_layer1 = ClusteringLayer(self.n_clusters, name='clustering1')(self.encoder1.output)
            clustering_layer2 = ClusteringLayer(self.n_clusters, name='clustering2')(self.encoder2.output)
            clustering_layer3 = ClusteringLayer(self.n_clusters, name='clustering3')(self.encoder3.output)
            self.model = Model(inputs=[self.ae1.input[0], self.ae1.input[1], self.ae2.input, self.ae3.input],
                               outputs=[clustering_layer1, self.ae1.output,
                                        clustering_layer2, self.ae2.output,
                                        clustering_layer3, self.ae3.output])

    def pretrain2(self, xn, x, yn, y, optimizer='adam', epochs=200, batch_size=256,
                  save_dir='results/temp', verbose=0):
        print('Begin pretraining: ', '-' * 60)
        self.autoencoder.compile(optimizer=optimizer, loss=[self.loss, 'mse'])
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain2_ae2_log.csv')
        save = '/ae2_weights.h5'
        cb = [csv_logger]
        if yn is not None and verbose > 0:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y, flag=1):
                    self.x = x
                    self.y = y
                    self.flag = flag
                    super(PrintACC, self).__init__()
        t0 = time()
        self.autoencoder.fit([xn[0], xn[1], x], [xn[2], x], batch_size=batch_size, epochs=epochs, callbacks=cb,
                             verbose=verbose)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + save)
        print('Pretrained weights are saved to ' + save_dir + save)
        self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def pretrain3(self, xn, x1, x2, yn, y1, y2, optimizer='adam', epochs=200, batch_size=256,
                  save_dir='results/temp', verbose=0):
        print('Begin pretraining: ', '-' * 60)
        self.autoencoder.compile(optimizer=optimizer, loss=[self.loss, 'mse', 'mse'])
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain2_ae2_log.csv')
        save = '/ae2_weights.h5'
        cb = [csv_logger]
        if yn is not None and verbose > 0:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y, flag=1):
                    self.x = x
                    self.y = y
                    self.flag = flag
                    super(PrintACC, self).__init__()
        t0 = time()
        self.autoencoder.fit([xn[0], xn[1], x1, x2], [xn[2], x1, x2], batch_size=batch_size, epochs=epochs,
                             callbacks=cb, verbose=verbose)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + save)
        print('Pretrained weights are saved to ' + save_dir + save)
        self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def load_weights(self, weights):  # load weights of model
        self.model.load_weights(weights)

    def get_layers2(self, ):
        c1 = self.model.get_layer(name='clustering1')
        c2 = self.model.get_layer(name='clustering2')
        return c1, c2

    def get_layers3(self, ):
        c1 = self.model.get_layer(name='clustering1')
        c2 = self.model.get_layer(name='clustering2')
        c3 = self.model.get_layer(name='clustering3')
        return c1, c2, c3

    def predict_labels2(self, x1, x2, x3):  # predict cluster labels using the output of clustering layer
        [q1, _, q2, _] = self.model.predict([x1, x2, x3])
        return np.argmax(q1, 1), np.argmax(q2, 1), q1, q2

    def predict_labels3(self, x1, x2, x3, x4):  # predict cluster labels using the output of clustering layer
        [q1, _, q2, _, q3, _] = self.model.predict([x1, x2, x3, x4], verbose=0)
        return np.argmax(q1, 1), np.argmax(q2, 1), np.argmax(q3, 1), q1, q2, q3

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='adam', loss=['kld', 'mse'], loss_weights=[0.1, 1.0]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch2(self, x1, x2, y1, y2, sample_weight=None):
        return self.model.train_on_batch([x1, x2], [y1, x1, y2, x2], sample_weight)

    def train_on_batch3(self, x1, x2, x3, y1, y2, y3, sample_weight=None):
        return self.model.train_on_batch([x1, x2, x3], [y1, x1, y2, x2, y3, x3], sample_weight)

    def fit2(self, arg, x1, x2, x3, x1p, y, maxiter=2e4, batch_size=256, tol=1e-3, optimizer='adam',
             UpdateCoo=200, save_dir='./results/tmp'):

        self.model.compile(optimizer=optimizer, loss=['kld', self.loss, 'kld', 'mse'],
                           loss_weights=[0.1, 1, 0.1, 1])  # [0.1, args.Idec, 0.1, args.Idec])

        print('Begin clustering:', '-' * 60)
        print('Update Coo:', UpdateCoo)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        ting = time() - t1
        print(ting)
        plotdata_ami_1 = []
        plotdata_ari_1 = []
        plotdata_ami_2 = []
        plotdata_ari_2 = []
        time_record = []
        time_record.append(int(ting))
        print(time_record)
        print('Initializing cluster centers with k-means.')
        kmeans1 = KMeans(n_clusters=self.n_clusters, n_init=40, random_state=1)
        kmeans2 = KMeans(n_clusters=self.n_clusters, n_init=40, random_state=1)

        [features1, features2] = self.encoder.predict({'input1': x1, 'size_factors': x2, 'input2': x1p})
        y_pred1 = kmeans1.fit_predict(features1)
        y_pred2 = kmeans2.fit_predict(features2)

        acc1 = np.round(Nmetrics.acc(y, y_pred1), 5)
        ami1 = np.round(Nmetrics.ami(y, y_pred1, "max"), 5)
        vmea1 = np.round(Nmetrics.vmeasure(y, y_pred1), 5)
        ari1 = np.round(Nmetrics.ari(y, y_pred1), 5)
        print('Start-1: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f' % (acc1, ami1, vmea1, ari1))

        acc2 = np.round(Nmetrics.acc(y, y_pred2), 5)
        ami2 = np.round(Nmetrics.ami(y, y_pred2, "max"), 5)
        vmea2 = np.round(Nmetrics.vmeasure(y, y_pred2), 5)
        ari2 = np.round(Nmetrics.ari(y, y_pred2), 5)
        print('Start-2: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f' % (acc2, ami2, vmea2, ari2))
        plotdata_ami_1.append(ami1)
        plotdata_ari_1.append(ari1)
        plotdata_ami_2.append(ami2)
        plotdata_ari_2.append(ari2)
        y_pred1_last = np.copy(y_pred1)
        y_pred2_last = np.copy(y_pred2)
        np.save("V1C1.npy", [kmeans1.cluster_centers_])
        np.save("V2C2.npy", [kmeans2.cluster_centers_])
        center1 = np.load("V1C1.npy")
        center2 = np.load("V2C2.npy")

        if arg.K12q == 1:
            c1 = center1
            c2 = center1
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
        elif arg.K12q == 2:
            c1 = center2
            c2 = center2
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
        elif arg.K12q == 3:
            c1 = center1
            c2 = center2
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
        elif arg.K12q == 4:
            c1 = center1
            c2 = center2
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
        else:
            c1 = 0
            c2 = 0
        # Step 2: deep clustering
        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w', newline='')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ami', 'vmea', 'ari', 'loss'])
        logwriter.writeheader()

        loss1 = 0
        index1 = 0
        index_array1 = np.arange(x1.shape[0])
        loss2 = 0
        index2 = 0
        index_array2 = np.arange(x2.shape[0])
        avg_loss1 = 0
        avg_loss2 = 0
        flag = 1
        vf = arg.view_first
        print('First view:', vf)
        update_interval = arg.UpdateCoo
        x1_sp = np.copy(x1)  ## count
        x2_sp = np.copy(x2)  ## factor
        x1p_sp = np.copy(x1p)  ## protein
        y_sp = np.copy(y)  ## label
        y_pred_sp1 = y_pred1
        y_pred_sp2 = y_pred2
        all_s = -1

        for ite in range(int(maxiter) + 1):  # fine-turn
            if ite % update_interval == 0:  # {'input1': x1, 'size_factors': x2, 'input2':x1p}
                [q1, _, q2, _] = self.model.predict({'input1': x1_sp, 'size_factors': x2_sp, 'input2': x1p_sp})
                # Alternate to align the z
                y_pred_sp1 = q1.argmax(1)
                y_pred_sp2 = q2.argmax(1)
                if vf == 1:
                    if flag == 1:
                        p1 = self.target_distribution(q1)  # normalizing
                        p = p1
                        flag = -flag
                        print('next corresponding: p1')
                    else:
                        p2 = self.target_distribution(q2)
                        p = p2
                        flag = -flag
                        print('next corresponding: p2')
                else:
                    if flag == 1:
                        p2 = self.target_distribution(q2)
                        p = p2
                        flag = -flag
                        print('next corresponding: p2')
                    else:
                        p1 = self.target_distribution(q1)
                        p = p1
                        flag = -flag
                        print('next corresponding: p1')
                if arg.Coo == 1:
                    p1 = p
                    p2 = p
                else:
                    p1 = self.target_distribution(q1)
                    p2 = self.target_distribution(q2)
                # evaluate the clustering performance
                avg_loss1 = loss1 / update_interval
                avg_loss2 = loss2 / update_interval
                loss1 = 0.
                loss2 = 0.

                if y_sp is not None:
                    for num in range(10):
                        same = np.where(y_sp == num)  # true value
                        same = np.array(same)[0]
                        out1 = y_pred_sp1[same]  # The true value corresponds to the predicted value of v1
                        # print(out1[0:37])
                        out2 = y_pred_sp2[same]
                        # print(out2[0:37])
                        out = out1 - out2  # The two v's are the same
                        if len(out) != 0:
                            print('%d, %.2f%%, %d' % (
                                num, len(np.array(np.where(out == 0))[0]) * 100 / len(out), len(same)))
                        else:
                            print('%d, %.2f%%. %d' % (num, 0, len(same)))
                    acc1 = np.round(Nmetrics.acc(y_sp, y_pred_sp1), 5)
                    ami1 = np.round(Nmetrics.ami(y_sp, y_pred_sp1, "max"), 5)
                    vme1 = np.round(Nmetrics.vmeasure(y_sp, y_pred_sp1), 5)
                    ari1 = np.round(Nmetrics.ari(y_sp, y_pred_sp1), 5)
                    fmi1 = np.round(Nmetrics.fmi(y_sp, y_pred_sp1), 5)
                    logdict = dict(iter=ite, ami=ami1, vmea=vme1, ari=ari1, loss=avg_loss1)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('V1-Iter %d: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f; loss=%.5f' % (
                        ite, acc1, ami1, vme1, ari1, avg_loss1))

                    acc2 = np.round(Nmetrics.acc(y_sp, y_pred_sp2), 5)
                    ami2 = np.round(Nmetrics.ami(y_sp, y_pred_sp2, "max"), 5)
                    vme2 = np.round(Nmetrics.vmeasure(y_sp, y_pred_sp2), 5)
                    ari2 = np.round(Nmetrics.ari(y_sp, y_pred_sp2), 5)
                    fmi2 = np.round(Nmetrics.fmi(y_sp, y_pred_sp2), 5)
                    logdict = dict(iter=ite, ami=ami2, vmea=vme2, ari=ari2, loss=avg_loss2)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('V2-Iter %d: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f; loss=%.5f' % (
                        ite, acc2, ami2, vme2, ari2, avg_loss2))

                    ting = time() - t1
                    plotdata_ami_1.append(ami1)
                    plotdata_ari_1.append(ari1)
                    plotdata_ami_2.append(ami2)
                    plotdata_ari_2.append(ari2)
                    time_record.append(int(ting))
                    print(time_record)

                # check stop criterion
                # delta_label1 = np.sum(y_pred1 != y_pred1_last).astype(np.float32) / y_pred1.shape[0]
                # y_pred1_last = np.copy(y_pred1)
                # if ite > 0 and delta_label1 < tol:
                #     print('delta_label ', delta_label1, '< tol ', tol)
                #     print('Reached tolerance threshold. Stopping training.')
                #     logfile.close()
                #     break
                #
                # delta_label2 = np.sum(y_pred2 != y_pred2_last).astype(np.float32) / y_pred2.shape[0]
                # y_pred2_last = np.copy(y_pred2)
                # if ite > 0 and delta_label2 < tol:
                #     print('delta_label ', delta_label2, '< tol ', tol)
                #     print('Reached tolerance threshold. Stopping training.')
                #     logfile.close()
                #     break

            # train on batch
            idx1 = index_array1[index1 * batch_size: min((index1 + 1) * batch_size, x1_sp.shape[0])]
            x_batch1 = x1_sp[idx1]  # view1 count
            idx2 = idx1
            x_batch2 = x2_sp[idx2]  # view1 factor
            x_batch3 = x1p_sp[idx2]  # view2 input
            x_batch4 = x3[idx2]  # view1 rowcount
            tmp = self.model.train_on_batch([x_batch1, x_batch2, x_batch3], [p1[idx1], x_batch4, p2[idx2], x_batch3])
            loss1 += (tmp[1] + tmp[2])  # ignoring gamma ， just to see the loss
            loss2 += (tmp[3] + tmp[4])

            index1 = index1 + 1 if (index1 + 1) * batch_size <= x1_sp.shape[0] else 0
            index2 = index2 + 1 if (index2 + 1) * batch_size <= x2_sp.shape[0] else 0

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)
        np.save("plotdata/" + arg.Address + "_ari1_V1.npy", plotdata_ari_1)
        np.save("plotdata/" + arg.Address + "_ari2_V2.npy", plotdata_ari_2)
        np.save("plotdata/" + arg.Address + "_ami1_V1.npy", plotdata_ami_1)
        np.save("plotdata/" + arg.Address + "_ami2_V2.npy", plotdata_ami_2)
        np.save("plotdata/" + arg.Address + "_time.npy", time_record)
        # tmp = self.model.train_on_batch([x_batch1, x_batch2, x_batch3], [p1[idx1], x_batch4, p2[idx2], x_batch3])   # [y, xn, y, x]
        [q1, _, q2, _] = self.model.predict([x1_sp, x2_sp, x1p_sp])
        y_pred1 = q1.argmax(1)
        y_pred2 = q2.argmax(1)
        y_q = (q1 + q2) / 2
        y_pred = y_q.argmax(1)
        return y_pred1, y_pred2, y_pred

    def fit3(self, arg, x1, x2, x3, x1p, x2p, y, maxiter=2e4, batch_size=256, tol=1e-3, optimizer='adam',
             UpdateCoo=200, save_dir='./results/tmp'):
        self.model.compile(optimizer=optimizer, loss=['kld', self.loss, 'kld', 'mse', 'kld', 'mse'],
                           loss_weights=[0.1, 1, 0.1, 1, 0.1, 1])  # [0.1, args.Idec, 0.1, args.Idec])

        print('Begin clustering:', '-' * 60)
        print('Update Coo:', UpdateCoo)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        ting = time() - t1
        print(ting)
        plotdata_ami_1 = []
        plotdata_ari_1 = []
        plotdata_ami_2 = []
        plotdata_ari_2 = []
        plotdata_ami_3 = []
        plotdata_ari_3 = []
        time_record = []
        time_record.append(int(ting))
        print(time_record)
        print('Initializing cluster centers with k-means.')
        kmeans1 = KMeans(n_clusters=self.n_clusters, n_init=40, random_state=1)
        kmeans2 = KMeans(n_clusters=self.n_clusters, n_init=40, random_state=1)
        kmeans3 = KMeans(n_clusters=self.n_clusters, n_init=40, random_state=1)
        [features1, features2, features3] = self.encoder.predict(
            {'input1': x1, 'size_factors': x2, 'input2': x1p, 'input3': x2p})
        y_pred1 = kmeans1.fit_predict(features1)
        y_pred2 = kmeans2.fit_predict(features2)
        y_pred3 = kmeans3.fit_predict(features3)
        acc1 = np.round(Nmetrics.acc(y, y_pred1), 5)
        ami1 = np.round(Nmetrics.ami(y, y_pred1, "max"), 5)
        vmea1 = np.round(Nmetrics.vmeasure(y, y_pred1), 5)
        ari1 = np.round(Nmetrics.ari(y, y_pred1), 5)
        fmi1 = np.round(Nmetrics.fmi(y, y_pred1), 5)
        print('Start-1: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f, fmi=%.5f' % (acc1, ami1, vmea1, ari1, fmi1))

        acc2 = np.round(Nmetrics.acc(y, y_pred2), 5)
        ami2 = np.round(Nmetrics.ami(y, y_pred2, "max"), 5)
        vmea2 = np.round(Nmetrics.vmeasure(y, y_pred2), 5)
        ari2 = np.round(Nmetrics.ari(y, y_pred2), 5)
        fmi2 = np.round(Nmetrics.fmi(y, y_pred2), 5)
        print('Start-2: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f, fmi=%.5f' % (acc2, ami2, vmea2, ari2, fmi2))

        acc3 = np.round(Nmetrics.acc(y, y_pred3), 5)
        ami3 = np.round(Nmetrics.ami(y, y_pred3, "max"), 5)
        vmea3 = np.round(Nmetrics.vmeasure(y, y_pred3), 5)
        ari3 = np.round(Nmetrics.ari(y, y_pred3), 5)
        fmi3 = np.round(Nmetrics.fmi(y, y_pred3), 5)
        print('Start-3: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f, fmi=%.5f' % (acc3, ami3, vmea3, ari3, fmi3))
        plotdata_ami_1.append(ami1)
        plotdata_ari_1.append(ari1)
        plotdata_ami_2.append(ami2)
        plotdata_ari_2.append(ari2)
        plotdata_ami_3.append(ami3)
        plotdata_ari_3.append(ari3)
        y_pred1_last = np.copy(y_pred1)
        y_pred2_last = np.copy(y_pred2)
        y_pred3_last = np.copy(y_pred3)
        np.save("V1C1.npy", [kmeans1.cluster_centers_])
        np.save("V2C2.npy", [kmeans2.cluster_centers_])
        np.save("V3C3.npy", [kmeans3.cluster_centers_])
        center1 = np.load("V1C1.npy")
        center2 = np.load("V2C2.npy")
        center3 = np.load("V3C3.npy")
        if arg.K12q == 1:
            c1 = center1
            c2 = center1
            c3 = center1
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
            self.model.get_layer(name='clustering3').set_weights(c3)
        elif arg.K12q == 2:
            c1 = center2
            c2 = center2
            c3 = center2
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
            self.model.get_layer(name='clustering3').set_weights(c3)
        elif arg.K12q == 3:
            c1 = center1
            c2 = center2
            c3 = center3
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
            self.model.get_layer(name='clustering3').set_weights(c3)
        elif arg.K12q == 4:
            c1 = center3
            c2 = center3
            c3 = center3
            self.model.get_layer(name='clustering1').set_weights(c1)
            self.model.get_layer(name='clustering2').set_weights(c2)
            self.model.get_layer(name='clustering3').set_weights(c3)
        else:
            c1 = 0
            c2 = 0
            c3 = 0
        # Step 2: deep clustering
        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w', newline='')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ami', 'vmea', 'ari', 'loss'])
        logwriter.writeheader()

        loss1 = 0
        index1 = 0
        index_array1 = np.arange(x1.shape[0])
        loss2 = 0
        index2 = 0
        index_array2 = np.arange(x2.shape[0])
        loss3 = 0
        index3 = 0
        index_array3 = np.arange(x3.shape[0])
        avg_loss1 = 0
        avg_loss2 = 0
        avg_loss3 = 0
        flag = 1
        vf = arg.view_first
        print('First view:', vf)
        update_interval = arg.UpdateCoo
        x1_sp = np.copy(x1)  ## count
        x2_sp = np.copy(x2)  ## factor
        x1p_sp = np.copy(x1p)  ## protein
        x2p_sp = np.copy(x2p)  ## impute
        y_sp = np.copy(y)  ## label
        y_pred_sp1 = y_pred1
        y_pred_sp2 = y_pred2
        y_pred_sp3 = y_pred3
        all_s = -1

        for ite in range(int(maxiter) + 1):  # fine-turn
            if ite % update_interval == 0:  # {'input1': x1, 'size_factors': x2, 'input2':x1p}
                [q1, _, q2, _, q3, _] = self.model.predict(
                    {'input1': x1_sp, 'size_factors': x2_sp, 'input2': x1p_sp, 'input3': x2p_sp})
                # Alternate to align the z
                y_pred_sp1 = q1.argmax(1)
                y_pred_sp2 = q2.argmax(1)
                y_pred_sp3 = q3.argmax(1)
                if vf == 1:
                    if flag % 3 == 1:
                        p1 = self.target_distribution(q1)  # normalizing
                        p = p1
                        flag += 1
                        print('next corresponding: p1')
                    elif flag % 3 == 2:
                        p2 = self.target_distribution(q2)
                        p = p2
                        flag += 1
                        print('next corresponding: p2')
                    else:
                        p3 = self.target_distribution(q3)
                        p = p3
                        flag += 1
                        print('next corresponding: p3')

                elif vf == 2:
                    if flag % 3 == 1:
                        p2 = self.target_distribution(q2)
                        p = p2
                        flag += 1
                        print('next corresponding: p2')
                    elif flag % 3 == 2:
                        p3 = self.target_distribution(q3)
                        p = p3
                        flag += 1
                        print('next corresponding: p3')
                    else:
                        p1 = self.target_distribution(q1)  # normalizing
                        p = p1
                        flag += 1
                        print('next corresponding: p1')
                else:
                    if flag % 3 == 1:
                        p3 = self.target_distribution(q3)
                        p = p3
                        flag += 1
                        print('next corresponding: p3')
                    elif flag % 3 == 2:
                        p1 = self.target_distribution(q1)
                        p = p1
                        flag += 1
                        print('next corresponding: p1')
                    else:
                        p2 = self.target_distribution(q2)  # normalizing
                        p = p2
                        flag += 1
                        print('next corresponding: p2')
                if arg.Coo == 1:
                    p1 = p
                    p2 = p
                    p3 = p
                else:
                    p1 = self.target_distribution(q1)
                    p2 = self.target_distribution(q2)
                    p2 = self.target_distribution(q3)
                # evaluate the clustering performance
                avg_loss1 = loss1 / update_interval
                avg_loss2 = loss2 / update_interval
                avg_loss3 = loss3 / update_interval
                loss1 = 0.
                loss2 = 0.
                loss3 = 0.

                if y_sp is not None:
                    for num in range(10):
                        same = np.where(y_sp == num)  # true value
                        same = np.array(same)[0]
                        out1 = y_pred_sp1[same]
                        # print(out1[0:37])
                        out2 = y_pred_sp2[same]
                        # print(out2[0:37])
                        out3 = y_pred_sp3[same]
                        # print(out2[0:37])
                        out12 = out1 - out2
                        out23 = out2 - out3
                        out13 = out1 - out3
                        out = out12 - out23
                        if (len(out) != 0) and (len(out12) != 0) and (len(out23) != 0) and (len(out13) != 0):
                            print('%d, 1-2-3:%.2f%%, 1-2:%.2f%%, 2-3:%.2f%%, 1-3:%.2f%%, %d' % (num,
                                                                                                len(np.array(
                                                                                                    np.where(out == 0))[
                                                                                                        0]) * 100 / len(
                                                                                                    out),
                                                                                                len(np.array(np.where(
                                                                                                    out12 == 0))[
                                                                                                        0]) * 100 / len(
                                                                                                    out),
                                                                                                len(np.array(np.where(
                                                                                                    out23 == 0))[
                                                                                                        0]) * 100 / len(
                                                                                                    out),
                                                                                                len(np.array(np.where(
                                                                                                    out13 == 0))[
                                                                                                        0]) * 100 / len(
                                                                                                    out),
                                                                                                len(same)))
                        else:
                            print('%d, %.2f%%. %d' % (num, 0, len(same)))

                    acc1 = np.round(Nmetrics.acc(y_sp, y_pred_sp1), 5)
                    ami1 = np.round(Nmetrics.ami(y_sp, y_pred_sp1, "max"), 5)
                    vme1 = np.round(Nmetrics.vmeasure(y_sp, y_pred_sp1), 5)
                    ari1 = np.round(Nmetrics.ari(y_sp, y_pred_sp1), 5)
                    fmi1 = np.round(Nmetrics.fmi(y_sp, y_pred_sp1), 5)
                    logdict = dict(iter=ite, ami=ami1, vmea=vme1, ari=ari1, loss=avg_loss1)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('V1-Iter %d: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f, fmi=%.5f; loss=%.5f' % (
                        ite, acc1, ami1, vme1, ari1, fmi1, avg_loss1))

                    acc2 = np.round(Nmetrics.acc(y_sp, y_pred_sp2), 5)
                    ami2 = np.round(Nmetrics.ami(y_sp, y_pred_sp2, "max"), 5)
                    vme2 = np.round(Nmetrics.vmeasure(y_sp, y_pred_sp2), 5)
                    ari2 = np.round(Nmetrics.ari(y_sp, y_pred_sp2), 5)
                    fmi2 = np.round(Nmetrics.fmi(y_sp, y_pred_sp2), 5)
                    logdict = dict(iter=ite, ami=ami2, vmea=vme2, ari=ari2, loss=avg_loss2)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('V2-Iter %d: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f, fmi=%.5f; loss=%.5f' % (
                        ite, acc2, ami2, vme2, ari2, fmi2, avg_loss2))

                    acc3 = np.round(Nmetrics.acc(y_sp, y_pred_sp3), 5)
                    ami3 = np.round(Nmetrics.ami(y_sp, y_pred_sp3, "max"), 5)
                    vme3 = np.round(Nmetrics.vmeasure(y_sp, y_pred_sp3), 5)
                    ari3 = np.round(Nmetrics.ari(y_sp, y_pred_sp3), 5)
                    fmi3 = np.round(Nmetrics.fmi(y_sp, y_pred_sp3), 5)
                    logdict = dict(iter=ite, ami=ami3, vmea=vme3, ari=ari3, loss=avg_loss3)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('V3-Iter %d: acc=%.5f, ami=%.5f, v-measure=%.5f, ari=%.5f, fmi=%.5f; loss=%.5f' % (
                        ite, acc3, ami3, vme3, ari3, fmi3, avg_loss3))

                    ting = time() - t1
                    plotdata_ami_1.append(ami1)
                    plotdata_ari_1.append(ari1)
                    plotdata_ami_2.append(ami2)
                    plotdata_ari_2.append(ari2)
                    plotdata_ami_3.append(ami3)
                    plotdata_ari_3.append(ari3)
                    time_record.append(int(ting))
                    print(time_record)

                # check stop criterion
                # delta_label1 = np.sum(y_pred1 != y_pred1_last).astype(np.float32) / y_pred1.shape[0]
                # y_pred1_last = np.copy(y_pred1)
                # if ite > 0 and delta_label1 < tol:
                #     print('delta_label ', delta_label1, '< tol ', tol)
                #     print('Reached tolerance threshold. Stopping training.')
                #     logfile.close()
                #     break
                #
                # delta_label2 = np.sum(y_pred2 != y_pred2_last).astype(np.float32) / y_pred2.shape[0]
                # y_pred2_last = np.copy(y_pred2)
                # if ite > 0 and delta_label2 < tol:
                #     print('delta_label ', delta_label2, '< tol ', tol)
                #     print('Reached tolerance threshold. Stopping training.')
                #     logfile.close()
                #     break
            # train on batch
            idx1 = index_array1[index1 * batch_size: min((index1 + 1) * batch_size, x1_sp.shape[0])]
            x_batch1 = x1_sp[idx1]  # view1 count
            idx2 = idx1
            idx3 = idx1
            x_batch2 = x2_sp[idx2]  # view1 factor
            x_batch3 = x1p_sp[idx2]  # view2 input
            x_batch4 = x3[idx2]  # view1 rowcount
            x_batch5 = x2p_sp[idx2]  # view3 impute
            tmp = self.model.train_on_batch([x_batch1, x_batch2, x_batch3, x_batch5],
                                            [p1[idx1], x_batch4, p2[idx2], x_batch3, p3[idx3], x_batch5])
            loss1 += (tmp[1] + tmp[2])  # ignoring gamma ， just to see the loss
            loss2 += (tmp[3] + tmp[4])
            loss3 += (tmp[5] + tmp[6])
            index1 = index1 + 1 if (index1 + 1) * batch_size <= x1_sp.shape[0] else 0
            index2 = index1
            index3 = index1
            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)
        np.save("plotdata/" + arg.Address + "_ari1_V1.npy", plotdata_ari_1)
        np.save("plotdata/" + arg.Address + "_ari2_V2.npy", plotdata_ari_2)
        np.save("plotdata/" + arg.Address + "_ari2_V3.npy", plotdata_ari_3)
        np.save("plotdata/" + arg.Address + "_ami1_V1.npy", plotdata_ami_1)
        np.save("plotdata/" + arg.Address + "_ami2_V2.npy", plotdata_ami_2)
        np.save("plotdata/" + arg.Address + "_ami2_V3.npy", plotdata_ami_3)
        np.save("plotdata/" + arg.Address + "_time.npy", time_record)
        [q1, _, q2, _, q3, _] = self.model.predict([x1_sp, x2_sp, x1p_sp, x2p_sp])
        y_pred1 = q1.argmax(1)
        y_pred2 = q2.argmax(1)
        y_pred3 = q3.argmax(1)
        y_q = (q1 + q2 + q3) / 2
        y_pred = y_q.argmax(1)
        return y_pred1, y_pred2, y_pred3, y_pred
