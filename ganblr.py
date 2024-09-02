import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.activations import softmax


class softmax_weight(Constraint):
    """Constrains weight tensors to be under softmax `."""

    def __init__(self, feature_uniques):
        idxs = math_ops.cumsum([0] + feature_uniques)
        idxs = [i.numpy() for i in idxs]
        self.feature_idxs = [
            (idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)
        ]

    def __call__(self, w):
        w_new = [
            math_ops.log(softmax(w[i:j, :], axis=0))
            for i, j in self.feature_idxs
        ]
        return tf.concat(w_new, 0)

    def get_config(self):
        return {'feature_idxs': self.feature_idxs}


def sample(*arrays, size=None, frac=None):
    '''
    random sample from arrays.

    Note: arrays must be equal-length

    size = None (default) indicate that return a permutation of given arrays.
    '''
    if len(arrays) < 1:
        return None
    if frac is not None and frac <= 1 and frac > 0:
        size = int(len(arrays[0]) * frac)
    if size is None:
        size = len(arrays[0])

    random_idxs = np.random.permutation(len(arrays[0]))[:size]
    results = []
    for arr in arrays:
        results.append(arr[random_idxs])
    return results


def elr_loss(KL_LOSS):
  def loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)+ KL_LOSS
  return loss


# from tensorflow.keras.layers import Concatenate
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder


class GANBLR:
    def __init__(self, log=True):
        self.g = None
        self.d = None
        self.log = log
        self.g_history = []
        self.d_history = []

        self.batch_size = 32  # default value
        self.feature_uniques = []
        self.class_unique = 0
        self.g_input_dim = 0
        self.d_input_dim = 0
        self.y_counts = []

    def generate(self, size=None, ohe=False):
        from pgmpy.models import NaiveBayes
        from pgmpy.sampling import BayesianModelSampling
        from pgmpy.factors.discrete import TabularCPD
        # basic varibles
        weights = self.g.get_weights()[0]
        n_features = len(self.feature_uniques)
        n_classes = weights.shape[1]
        n_samples = np.sum(self.class_counts)
        # cut weights by feature uniques
        idxs = np.cumsum([0] + self.feature_uniques)
        feature_idxs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        feature_names = [str(i) for i in range(n_features)]
        # get cpd of features
        feature_probs = np.exp(weights)
        feature_cpd_probs = [feature_probs[start:end, :] for start, end in feature_idxs]
        feature_cpd_probs = [p / p.sum(axis=0, keepdims=1) for p in feature_cpd_probs]
        feature_cpds = [
            TabularCPD(name, n_unique, table, evidence=['y'], evidence_card=[n_classes])
            for name, n_unique, table in zip(feature_names, self.feature_uniques, feature_cpd_probs)
        ]
        # get cpd of label
        y_probs = (self.class_counts / n_samples).reshape(-1, 1)
        y_cpd = TabularCPD('y', n_classes, y_probs)

        # define the model
        elr = NaiveBayes(feature_names, 'y')
        elr.add_cpds(y_cpd, *feature_cpds)
        # sampling
        sample_size = n_samples if size is None else size
        result = BayesianModelSampling(elr).forward_sample(size=sample_size)
        sorted_result = result[feature_names + ['y']].values
        # return
        syn_X, syn_y = sorted_result[:, :-1], sorted_result[:, -1]

        if ohe:
            ohe_syn_X = [np.eye(b)[syn_X[:, i]] for i, b in enumerate(self.feature_uniques)]
            ohe_syn_X = np.hstack(ohe_syn_X)
            return ohe_syn_X, syn_y
        else:
            return sorted_result

    def fit(self, X, y, epochs
            , batch_size=32, warm_up_epochs=10):

        ohe = OneHotEncoder().fit(X)
        ohe_X = ohe.transform(X).toarray()
        # feature_uniques = [len(np.unique(X[:,i])) for i in range(X.shape[1])]
        self.feature_uniques = [len(c) for c in ohe.categories_]
        y_unique, y_counts = np.unique(y, return_counts=True)
        self.class_unique = len(y_unique)
        self.class_counts = y_counts
        self.g_input_dim = np.sum(self.feature_uniques)
        self.d_input_dim = X.shape[1]
        self.batch_size = batch_size
        self._build_g()
        self._build_d()

        # warm up
        self.g.fit(ohe_X, y, epochs=warm_up_epochs, batch_size=batch_size)
        syn_data = self.generate(size=len(X))
        # real_data = np.concatenate([X, y.reshape(-1,1)], axis=-1)
        for i in range(epochs):
            # prepare data
            real_label = np.ones(len(X))
            syn_label = np.zeros(len(X))
            disc_label = np.concatenate([real_label, syn_label])
            disc_X = np.vstack([X, syn_data[:, :-1]])
            disc_X, disc_label = sample(disc_X, disc_label, frac=0.8)
            # train d
            self._train_d(disc_X, disc_label)
            prob_fake = self.d.predict(X)
            ls = np.mean(-np.log(np.subtract(1, prob_fake)))
            # train g
            self._train_g(ohe_X, y, loss=ls)
            syn_data = self.generate(size=len(X))

    def _train_g(self, X, y, epochs=1, loss=None):
        if loss is not None:
            clear_session()
            self._build_g(weights=self.g.get_weights(), loss=loss)
            self._build_d(weights=self.d.get_weights())

        history = self.g.fit(X, y, epochs=epochs, batch_size=self.batch_size)
        if self.log:
            self.g_history.append(history.history)

    def _train_d(self, X, y, epochs=1):
        history = self.d.fit(X, y, batch_size=self.batch_size, epochs=epochs)
        if self.log:
            self.d_history.append(history.history)

    def _build_g(self, weights=None, loss=None):
        if loss is None:
            loss = elr_loss(0)
        else:
            loss = elr_loss(loss)
        constraint = softmax_weight(self.feature_uniques)
        g = tf.keras.Sequential()
        g.add(Dense(self.class_unique, input_dim=self.g_input_dim, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(0.01), kernel_constraint=constraint))
        g.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        self.g = g

        if weights is not None:
            g.set_weights(weights)
        return g

    def _build_d(self, weights=None):
        d = tf.keras.Sequential()
        d.add(Dense(1, input_dim=self.d_input_dim, activation='sigmoid'))
        d.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.d = d

        if weights is not None:
            d.set_weights(weights)
        return d

import os
import csv
from sklearn.utils import shuffle
def split_csv(path, total_len, per):
    # 如果train.csv和vali.csv存在就删除
    if os.path.exists('./3.csv'):
        os.remove('./3.csv')
    with open(path, 'r', newline='') as file:
        csvreader = csv.reader(file)
        i = 0
        for row in csvreader:
            if i < round(total_len * per/100):
                # vali.csv存放路径
                csv_path = os.path.join('./', '3.csv')
                print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            else:
                break
    print("成功选取一半")
    return

import operator
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('./data_generated/train_kdd.csv')
encoder1 = OrdinalEncoder()
df = encoder1.fit_transform(data).astype('int')  #用于下面ganblr生成数据
df = pd.DataFrame(df,columns=data.columns)
data = df.copy(deep=True)

len_start = len(data)

train_onehot = pd.get_dummies(data, columns = data.columns.values[0:-1], prefix_sep='_', dummy_na=False, drop_first=False)

train = train_onehot.values[:,1:]  #one_hot矩阵后续的训练数据

y_train = train_onehot.iloc[:,0]

rf = RandomForestRegressor(n_estimators=100)  #随机森林模型
rf.fit(train, y_train)#拟合

fea_importance = rf.feature_importances_ #获取重要性
column = train_onehot.columns.values[1:]  #获取onehot列名
column_importance = list(zip(column,fea_importance))  #组合
column_importance.sort(key=operator.itemgetter(1), reverse=True) #排序
print(column_importance)
first,end = zip(*column_importance)

print(first)
j = 0
import copy
import random
for i in first:
    if '_' in  i:
        name_num = i.split('_')
        name = name_num[0]
        num = name_num[1]
        if num.isdigit():
            num = int(num)
        j += 1
        #get_data = copy.deepcopy(data[data[name] == num])
        #get_data.reset_index(drop=True, inplace=True)
        #index_delete = random.sample(range(0, get_data.shape[0]), int(0.3*get_data.shape[0]))
        data.drop(data[data[name] == num].index, inplace=True)
        #get_data.drop(get_data.index[index_delete], inplace=True)
        #data = shuffle(pd.concat([data, get_data],ignore_index=False))
        if j == 2:
            break
len_end = len(data)
data = encoder1.inverse_transform(data)
data = pd.DataFrame(data,columns=df.columns)
data.to_csv('./data_generated/train_feature_kdd_cut_2.csv',index=False)

print(len_start, len_end, len_start-len_end)
gan_num = int(len_start-len_end)
X_train = df.values[:,0:-1]
y_train = df.values[:,-1]
print(X_train,type(X_train))
print(y_train,type(y_train))
clear_session()
ganblr = GANBLR()
ganblr.fit(X_train, y_train, epochs=200, warm_up_epochs=5)

data_gan = ganblr.generate(gan_num)
data_gan = encoder1.inverse_transform(data_gan)
data_gan = pd.DataFrame(data_gan,columns=data.columns)
data_new = pd.concat([data,data_gan],ignore_index=False)
data_new.to_csv('./data_generated/train_feature_kdd_gan_2.csv',index=False)