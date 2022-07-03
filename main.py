from tensorflow import set_random_seed
import random as rn
from keras import backend as K
from numpy.random import seed
from preprocess import *
import os
from time import time
import Nmetrics
from MVDEC import *
import warnings
import pandas as pd
from keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _make_data_and_model(args, view=2):
    # prepare dataset
    if view == 2:
        adata = sc.AnnData(pd.read_csv("./data/" + args.dataset + "/RNA.csv", index_col=0).T.values)
        adata = normalize(adata, copy=True, highly_genes=None, size_factors=True, normalize_input=True,
                          logtrans_input=True)
        x1 = adata.X.astype(np.float32)
        y1 = adata.raw.X.astype(np.float32)
        sf = adata.obs.size_factors
        x2 = pd.read_csv("./data/" + args.dataset + "/ImputedRNA.csv").values.T
        y = pd.read_csv("./data/" + args.dataset + "/Truth.csv", index_col=0).values.flatten()
    else:
        adata = sc.AnnData(pd.read_csv("./data/" + args.dataset + "/RNA.csv", index_col=0).T.values)
        adata = normalize(adata, copy=True, highly_genes=None, size_factors=True, normalize_input=True,
                          logtrans_input=True)
        x1 = adata.X.astype(np.float32)
        y1 = adata.raw.X.astype(np.float32)
        sf = adata.obs.size_factors
        x2 = np.log1p(pd.read_csv("./data/" + args.dataset + "/ADT.csv").values.T)
        x3 = pd.read_csv("./data/" + args.dataset + "/ImputedRNA.csv", index_col=0).T.values
        y = pd.read_csv("./data/" + args.dataset + "/Truth.csv", index_col=0).values.flatten()
    # prepare the model
    n_clusters = len(np.unique(y))
    print('cluster:' + str(n_clusters))
    if view == 2:
        model = MvDEC(input1_shape=x1.shape[1], input2_shape=x2.shape[1], filters=[32, 64, 128, n_clusters], view=view,
                      noise_sd=args.noise_sd, n_clusters=n_clusters)
        return (x1, sf, y1, x2, y), model
    if view == 3:
        model = MvDEC(input1_shape=x1.shape[1], input2_shape=x2.shape[1], input3_shape=x3.shape[1],
                      filters=[32, 64, 128, n_clusters], view=view, noise_sd=args.noise_sd, n_clusters=n_clusters)
        return (x1, sf, y1, x2, x3, y), model


def train(args, view=2):
    # get data and mode
    if view == 2:
        (x_counts, sf, raw_counts, protein_count, y), model = _make_data_and_model(args, view)
    else:
        (x_counts, sf, raw_counts, protein_count, impute_count, y), model = _make_data_and_model(args, view)
    model.model.summary()
    model.autoencoder.summary()
    optimizer = Adam(lr=args.lr)
    # pretraining
    t0 = time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.pretrain_dir is not None and os.path.exists(args.pretrain_dir):  # load pretrained weights
        model.autoencoder.load_weights(args.pretrain_dir)
    else:  # train

        if view == 2:
            model.pretrain2([x_counts, sf, raw_counts], protein_count, y, y, optimizer=optimizer,
                            epochs=args.pretrain_epochs, batch_size=args.batch_size,
                            save_dir=args.save_dir, verbose=args.verbose)
        if view == 3:
            model.pretrain3([x_counts, sf, raw_counts], protein_count, impute_count, y, y, y, optimizer=optimizer,
                            epochs=args.pretrain_epochs, batch_size=args.batch_size,
                            save_dir=args.save_dir, verbose=args.verbose)

    t1 = time()
    print("Time for pretraining: %ds" % (t1 - t0))

    # clustering
    if view == 2:
        # (x_counts, sf, raw_counts, protein_count, y)
        y_pred1, y_pred2, y_pred = model.fit2(optimizer=optimizer, arg=args, x1=x_counts, x2=sf, x3=raw_counts,
                                              x1p=protein_count, y=y, maxiter=args.maxiter,
                                              batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
                                              save_dir=args.save_dir)
        if y is not None:
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred1), Nmetrics.ami(y, y_pred1, "max"), Nmetrics.ari(y, y_pred1)))
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred2), Nmetrics.ami(y, y_pred2, "max"), Nmetrics.ari(y, y_pred2)))
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred), Nmetrics.ami(y, y_pred, "max"), Nmetrics.ari(y, y_pred)))

    if view == 3:
        y_pred1, y_pred2, y_pred3, y_pred = model.fit3(optimizer=optimizer, arg=args, x1=x_counts, x2=sf, x3=raw_counts,
                                                       x1p=protein_count, x2p=impute_count, y=y, maxiter=args.maxiter,
                                                       batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
                                                       save_dir=args.save_dir)
        if y is not None:
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred1), Nmetrics.ami(y, y_pred1, "max"), Nmetrics.ari(y, y_pred1)))
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred2), Nmetrics.ami(y, y_pred2, "max"), Nmetrics.ari(y, y_pred2)))
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred3), Nmetrics.ami(y, y_pred3, "max"), Nmetrics.ari(y, y_pred3)))
            print('Final: acc=%.4f, ami=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_pred), Nmetrics.ami(y, y_pred, "max"), Nmetrics.ari(y, y_pred)))
    t2 = time()
    print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
    print('=' * 60)
    return Nmetrics.ami(y, y_pred, "max"), Nmetrics.ari(y, y_pred), Nmetrics.vmeasure(y, y_pred), Nmetrics.fmi(y,
                                                                                                               y_pred)


def validate(args, view=2):
    # assert args.weights is not None
    if view == 2:
        (x_counts, sf, raw_counts, protein_count, y), model = _make_data_and_model(args, view)
        model.model.summary()
        print('Begin testing:', '-' * 60)
        model.load_weights(args.weights)
        [y_pred1, y_pred2, q1, q2] = model.predict_labels2(x_counts, sf, protein_count)
        q_pred = (q1 + q2) / 2
        y_pred = np.argmax(q_pred, 1)
        print('V1:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred1), Nmetrics.ami(y, y_pred1, "max"),
            Nmetrics.vmeasure(y, y_pred1), Nmetrics.ari(y, y_pred1), Nmetrics.fmi(y, y_pred1)))
        print('V2:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred2), Nmetrics.ami(y, y_pred2, "max"),
            Nmetrics.vmeasure(y, y_pred2), Nmetrics.ari(y, y_pred2), Nmetrics.fmi(y, y_pred2)))
        print('DEMOC:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred), Nmetrics.ami(y, y_pred, "max"),
            Nmetrics.vmeasure(y, y_pred), Nmetrics.ari(y, y_pred), Nmetrics.fmi(y, y_pred)))
        print('End testing:', '-' * 60)

    if view == 3:
        (x_counts, sf, raw_counts, protein_count, impute_count, y), model = _make_data_and_model(args, view)
        model.model.summary()
        print('Begin testing:', '-' * 60)
        model.load_weights(args.weights)
        [y_pred1, y_pred2, y_pred3, q1, q2, q3] = model.predict_labels3(x_counts, sf, protein_count, impute_count)
        # [features1, features2, features3] = model.encoder.predict({'input1': x1, 'input2': x2, 'input3': x3})
        # c1, c2, c3 = model.get_layers3()
        q_pred = (q1 + q2 + q3) / 3
        y_pred = np.argmax(q_pred, 1)
        print('V1:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred1), Nmetrics.ami(y, y_pred1, "max"),
            Nmetrics.vmeasure(y, y_pred1), Nmetrics.ari(y, y_pred1), Nmetrics.fmi(y, y_pred1)))
        print('V2:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred2), Nmetrics.ami(y, y_pred2, "max"),
            Nmetrics.vmeasure(y, y_pred2), Nmetrics.ari(y, y_pred2), Nmetrics.fmi(y, y_pred2)))
        print('V3:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred3), Nmetrics.ami(y, y_pred3, "max"),
            Nmetrics.vmeasure(y, y_pred3), Nmetrics.ari(y, y_pred3), Nmetrics.fmi(y, y_pred3)))
        print('DEMOC:\t acc=%.4f, ami=%.4f, v-measure=%.4f, ari=%.4f, fmi=%.4f' % (
            Nmetrics.acc(y, y_pred), Nmetrics.ami(y, y_pred, "max"),
            Nmetrics.vmeasure(y, y_pred), Nmetrics.ari(y, y_pred), Nmetrics.fmi(y, y_pred)))
        print('End testing:', '-' * 60)


def set_seed():
    K.clear_session()
    K.set_floatx('float32')
    ###设置随机种子
    seed(2211)
    rn.seed(2)
    set_random_seed(2211)
    print("RANDOM SEEDS RESET")


if __name__ == "__main__":

    set_seed()
    TEST = True
    train_ae = False
    Coo = 1  # alternate
    data = 'In_house_PBMC2000'
    import os

    if not os.path.exists("./plotdata/" + data):
        os.mkdir("./plotdata/" + data)
    Address = 'Test'  # this address is the process of fine-tune(ACC ami ARI v-measure)
    C123q = 1  # kmeans-------1：k1 , 2：k2, 3：k3, 4: self_kmeans
    View = 1  # Coo target view First round of guidance
    noise_sd = 2.5
    # Determined trainin g parameters
    epochs = 500
    Update_Coo = 200
    Update_samples = 2000
    Maxiter = 20000
    Batch = 256
    Idec = 1.0  # dec 0.0 , idec 1.0
    lrate = 0.001  # keras defult lr 0.001
    testing = False

    Multi_view = 3
    path = 'results/' + data
    if train_ae:
        load = None
    else:
        load = path + '/ae2_weights.h5'

    if TEST:
        load_test = path + '/model_final.h5'
    else:
        load_test = None
    addressACC = data + '/'

    import argparse

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--Multi_view', default=Multi_view,
                        help="Dataset name to train on")
    parser.add_argument('--dataset', default=data,
                        help="Dataset name to train on")
    parser.add_argument('--save_dir', default=path,
                        help="Dir to save the results")
    # Parameters for pretraining
    parser.add_argument('--pretrain', default=train_ae, type=str,
                        help="Pretrain autoencoder")
    parser.add_argument('--pretrain_dir', default=load, type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain_epochs', default=epochs, type=int,  # 500
                        help="Number of epochs for pretraining")
    parser.add_argument('--verbose', default=1, type=int,
                        help="Verbose for pretraining")
    # Parameters for clustering
    parser.add_argument('--testing', default=testing, type=bool,  ## 决定是训练还是测试
                        help="Testing the clustering performance with provided weights")
    parser.add_argument('--weights', default=load_test, type=str,
                        help="Model weights, used for testing")
    parser.add_argument('--lr', default=lrate, type=float,
                        help="learning rate during clustering")
    parser.add_argument('--batch_size', default=Batch, type=int,  # 256
                        help="Batch size")
    parser.add_argument('--maxiter', default=5000, type=int,  # 4500 5000
                        # e4
                        help="Maximum number of iterations")
    parser.add_argument('--UpdateCoo', default=Update_Coo, type=int,  # 200  Alternate iteration
                        help="Number of iterations to update the target distribution")
    parser.add_argument('--tol', default=0.001, type=float,
                        help="Threshold of stopping training")
    parser.add_argument('--view_first', default=View, type=int,
                        help="first target view")
    parser.add_argument('--Coo', default=Coo, type=int,
                        help="is co-trainning")
    parser.add_argument('--K12q', default=3, type=int,
                        help="cluster centers")
    parser.add_argument('--ACC_addr', default=Address, type=str,
                        help="Address ACC ami ARI Vmeasure")
    parser.add_argument('--Address', default=addressACC + Address, type=str,
                        help="Address ACC ami ARI Vmeasure")
    parser.add_argument('--Idec', default=Idec, type=float,
                        help="is dec or Idec?")
    parser.add_argument('--noise_sd', default=noise_sd, type=float,
                        help="noise")

    args = parser.parse_args()

    args.save_dir = 'results/' + args.dataset
    if args.pretrain:
        args.pretrain_dir = None
    else:
        args.pretrain_dir = args.save_dir + '/ae2_weights.h5'

    if args.testing:
        args.weights = args.save_dir + '/model_final.h5'
    else:
        args.weights = None

    addressACC = args.dataset + '/'
    args.Address = addressACC + args.ACC_addr

    print('+' * 30, ' Setting ', '+' * 30)
    print(args)
    print('+' * 75)
    # testing
    if args.testing:
        validate(args, Multi_view)
    else:
        ami, ari, vmi, fmi = train(args, Multi_view)
