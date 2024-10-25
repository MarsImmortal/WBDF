import os
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, crosstab
import networkx as nx
from warnings import warn
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from pyitlib import discrete_random_variable as drv
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.activations import softmax
from tensorflow.keras.backend import clear_session
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Embedding,Dropout,concatenate,Flatten
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.feature_column import build_input_features, get_linear_logit, input_from_feature_columns,SparseFeat,DenseFeat,get_feature_names
from deepctr.models import xDeepFM


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


def build_graph(X_train, y_train, k=2):
    '''
    kDB algorithm

    Param:
    ----------------------

    Return:
    ----------------------
    graph edges
    '''
    # ensure data
    num_features = X_train.shape[1]
    x_nodes = list(range(num_features))
    y_node = num_features
    # X_train = X_train.to_numpy()
    # y_train = y_train.to_numpy()

    # util func
    _x = lambda i: X_train[:, i]
    _x2comb = lambda i, j: (X_train[:, i], X_train[:, j])

    # feature indexes desc sort by mutual information
    sorted_feature_idxs = np.argsort([
        drv.information_mutual(_x(i), y_train)
        for i in range(num_features)
    ])[::-1]

    # start building graph
    edges = []
    for iter, target_idx in enumerate(sorted_feature_idxs):
        target_node = x_nodes[target_idx]
        edges.append((y_node, target_node))

        parent_candidate_idxs = sorted_feature_idxs[:iter]
        if iter <= k:
            for idx in parent_candidate_idxs:
                edges.append((x_nodes[idx], target_node))
        else:
            first_k_parent_mi_idxs = np.argsort([
                drv.information_mutual_conditional(*_x2comb(i, target_idx), y_train)
                for i in parent_candidate_idxs
            ])[::-1][:k]
            first_k_parent_idxs = parent_candidate_idxs[first_k_parent_mi_idxs]

            for parent_idx in first_k_parent_idxs:
                edges.append((x_nodes[parent_idx], target_node))
    return edges


def draw_graph(edges):
    '''
    Draw the graph

    Param
    -----------------
    edges: edges of the graph

    '''
    graph = nx.DiGraph(edges)
    pos = nx.spiral_layout(graph)
    nx.draw(graph, pos, node_color='r', edge_color='b')
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")


def get_cross_table(*cols, apply_wt=False):
    '''
    author: alexland

    returns:
      (i) xt, NumPy array storing the xtab results, number of dimensions is equal to
          the len(args) passed in
      (ii) unique_vals_all_cols, a tuple of 1D NumPy array for each dimension
          in xt (for a 2D xtab, the tuple comprises the row and column headers)
      pass in:
        (i) 1 or more 1D NumPy arrays of integers
        (ii) if wts is True, then the last array in cols is an array of weights

    if return_inverse=True, then np.unique also returns an integer index
    (from 0, & of same len as array passed in) such that, uniq_vals[idx] gives the original array passed in
    higher dimensional cross tabulations are supported (eg, 2D & 3D)
    cross tabulation on two variables (columns):
    #>>> q1 = np.array([7, 8, 8, 8, 5, 6, 4, 6, 6, 8, 4, 6, 6, 6, 6, 8, 8, 5, 8, 6])
    #>>> q2 = np.array([6, 4, 6, 4, 8, 8, 4, 8, 7, 4, 4, 8, 8, 7, 5, 4, 8, 4, 4, 4])
    #>>> uv, xt = xtab(q1, q2)
    #>>> uv
      (array([4, 5, 6, 7, 8]), array([4, 5, 6, 7, 8]))
    #>>> xt
      array([[2, 0, 0, 0, 0],
             [1, 0, 0, 0, 1],
             [1, 1, 0, 2, 4],
             [0, 0, 1, 0, 0],
             [5, 0, 1, 0, 1]], dtype=uint64)
      '''
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")

    if len(cols) == 0:
        raise TypeError("xtab() requires at least one argument")

    fnx1 = lambda q: len(q.squeeze().shape)
    if not all([fnx1(col) == 1 for col in cols]):
        raise ValueError("all input arrays must be 1D")

    if apply_wt:
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = 1

    uniq_vals_all_cols, idx = zip(*(np.unique(col, return_inverse=True) for col in cols))
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    dtype_xt = 'float' if apply_wt else 'uint'
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt


def _get_dependencies_without_y(variables, y_name, kdb_edges):
    '''
    evidences of each variable without y.

    Param:
    --------------
    variables: variable names

    y_name: class name

    kdb_edges: list of tuple (source, target)
    '''
    dependencies = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t: i for i, (s, t) in enumerate(kdb_edges) if s == y_name}
    for x in variables:
        current_dependencies = [s for s, t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            sort_dict = {t: mi_desc_order[t] for t in current_dependencies}
            dependencies[x] = sorted(sort_dict)
        else:
            dependencies[x] = current_dependencies
    return dependencies


def _add_uniform(array, noise=1e-5):
    '''
    if no count on particular condition for any feature, give a uniform prob rather than leave 0
    '''
    sum_by_col = np.sum(array, axis=0)
    zero_idxs = (array == 0).astype(int)
    # zero_count_by_col = np.sum(zero_idxs,axis=0)
    nunique = array.shape[0]
    result = np.zeros_like(array, dtype='float')
    for i in range(array.shape[1]):
        if sum_by_col[i] == 0:
            result[:, i] = array[:, i] + 1. / nunique
        elif noise != 0:
            result[:, i] = array[:, i] + noise * zero_idxs[:, i]
        else:
            result[:, i] = array[:, i]
    return result


def get_kdb_embeddings(X_train, y_train, k=2, noise=1e-7, dtype='float32'):
    # assert(isinstance(X_train, DataFrame))
    # assert(isinstance(y_train, Series))

    kdb_embeddings = []

    # dependency graph generated by kdb algorithm
    edges = build_graph(X_train, y_train, k)
    n_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    if k > 0:
        dependencies = _get_dependencies_without_y(list(range(num_features)), num_features, edges)
    else:
        dependencies = {x: [] for x in range(num_features)}

    for x, evidences in dependencies.items():
        evidences = [X_train[:, e] for e in evidences] + [y_train]
        # evidences = np.hstack([X_train, y_train.reshape(-1,1)])
        # conditional probalility table of x
        normalized_cct = crosstab(X_train[:, x], evidences, dropna=False, normalize='columns').to_numpy()
        normalized_cct = _add_uniform(normalized_cct, noise)
        current_embeddings = np.log2(normalized_cct, dtype=dtype)
        kdb_embeddings.append(current_embeddings)

    return kdb_embeddings


def get_high_order_feature(X, col, evidence_cols, feature_uniques):
    '''
    encode the high order feature of X[col] given evidences X[evidence_cols].
    '''
    if evidence_cols is None or len(evidence_cols) == 0:
        return X[:, [col]]
    else:
        evidences = [X[:, _col] for _col in evidence_cols]

        # [1, variable_unique, evidence_unique]
        base = [1, feature_uniques[col]] + [feature_uniques[_col] for _col in evidence_cols[::-1][:-1]]
        cum_base = np.cumprod(base)[::-1]

        cols = evidence_cols + [col]
        high_order_feature = np.sum(X[:, cols] * cum_base, axis=1).reshape(-1, 1)
        return high_order_feature


def get_high_order_constraints(X, col, evidence_cols, feature_uniques):
    '''
    find the constraints infomation for the high order feature X[col] given evidences X[evidence_cols].

    Returns:
    ---------------------
    tuple(have_value, high_order_uniques)

    have_value: a k+1 dimensions numpy ndarray of type boolean.
        Each dimension correspond to a variable, with the order (*evidence_cols, col)
        True indicate the corresponding combination of variable values cound be found in the dataset.
        False indicate not.

    high_order_constraints: a 1d nummy ndarray of type int.
        Each number `c` indicate that there are `c` cols shound be applying the constraints since the last constrant position(or index 0),
        in sequence.

    '''
    if evidence_cols is None or len(evidence_cols) == 0:
        unique = feature_uniques[col]
        return np.ones(unique, dtype=bool), np.array([unique])
    else:
        cols = evidence_cols + [col]
        cross_table_idxs, cross_table = get_cross_table(*[X[:, i] for i in cols])
        have_value = cross_table != 0

        have_value_reshape = have_value.reshape(-1, have_value.shape[-1])
        have_value_split = np.split(have_value_reshape, have_value_reshape.shape[0], 0)
        high_order_constraints = np.sum(have_value_reshape, axis=-1)

        return have_value, high_order_constraints


def sample_synthetic_data(weights, kdb_high_order_encoder, y_counts, ohe=True, size=None):
    from pgmpy.models import BayesianModel
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.factors.discrete import TabularCPD
    # basic varibles
    feature_cards = np.array(kdb_high_order_encoder.feature_uniques_)
    n_features = len(feature_cards)
    n_classes = weights.shape[1]
    n_samples = y_counts.sum()

    # ensure sum of each constraint group equals to 1, then re concat the probs
    _idxs = np.cumsum([0] + kdb_high_order_encoder.constraints_.tolist())
    constraint_idxs = [(_idxs[i], _idxs[i + 1]) for i in range(len(_idxs) - 1)]

    probs = np.exp(weights)
    cpd_probs = [probs[start:end, :] for start, end in constraint_idxs]
    cpd_probs = np.vstack([p / p.sum(axis=0) for p in cpd_probs])

    # assign the probs to the full cpd tables
    idxs = np.cumsum([0] + kdb_high_order_encoder.high_order_feature_uniques_)
    feature_idxs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    have_value_idxs = kdb_high_order_encoder.have_value_idxs_
    full_cpd_probs = []
    for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
        # (n_high_order_feature_uniques, n_classes)
        cpd_prob_ = cpd_probs[start:end, :]
        # (n_all_combination) Note: the order is (*parent, variable)
        have_value_ravel = have_value.ravel()
        # (n_classes * n_all_combination)
        have_value_ravel_repeat = np.hstack([have_value_ravel] * n_classes)
        # (n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
        full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
        full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
        # (n_classes * n_parent_combinations, n_variable_unique)
        full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
        full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
        full_cpd_probs.append(full_cpd_prob)

    # prepare node and edge names
    node_names = [str(i) for i in range(n_features + 1)]
    edge_names = [(str(i), str(j)) for i, j in kdb_high_order_encoder.edges_]
    y_name = node_names[-1]

    # create TabularCPD objects
    evidences = kdb_high_order_encoder.dependencies_
    feature_cpds = [
        TabularCPD(str(name), feature_cards[name], table,
                   evidence=[y_name, *[str(e) for e in evidences]],
                   evidence_card=[n_classes, *feature_cards[evidences].tolist()])
        for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
    ]
    y_probs = (y_counts / n_samples).reshape(-1, 1)
    y_cpd = TabularCPD(y_name, n_classes, y_probs)

    # create kDB model, then sample data
    model = BayesianModel(edge_names)
    model.add_cpds(y_cpd, *feature_cpds)
    sample_size = n_samples if size is None else size
    result = BayesianModelSampling(model).forward_sample(size=sample_size)
    sorted_result = result[node_names].values

    # return
    syn_X, syn_y = sorted_result[:, :-1], sorted_result[:, -1]
    if ohe:
        from sklearn.preprocessing import OneHotEncoder
        ohe_syn_X = OneHotEncoder().fit_transform(syn_X)
        return ohe_syn_X, syn_y
    else:
        return syn_X, syn_y


def broad(input_dim, output_dim, constraint):
    return Dense(output_dim, input_dim=input_dim, activation='softmax', kernel_constraint=constraint)


def elr_dnn_cin(elr_constrains, broad_units, dnn_feature_columns, output_units, dnn_hidden_units=(256, 256),
                cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                use_fm=False, fm_group=None,
                l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
                dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """	Instantiates the xDeepFM architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(dnn_feature_columns)
    broad_inputs = Input((broad_units,), name='broad')

    inputs_list = list(features.values())
    inputs_list.insert(0, broad_inputs)

    broad_output = broad(broad_units, output_units, elr_constrains)(broad_inputs)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    cin_input = concat_func(sparse_embedding_list, axis=1)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_output = tf.keras.layers.Dense(
        output_units, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    ######line causing the issue########
    output = tf.concat([broad_output, dnn_output], axis=-1)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation,
                       cin_split_half, l2_reg_cin, seed)(cin_input)
        exFM_logit = tf.keras.layers.Dense(output_units, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(
            exFM_out)
        output = tf.concat([output, exFM_logit], axis=-1)

    if use_fm == True:
        from deepctr.layers.interaction import FM
        from deepctr.feature_column import DEFAULT_GROUP_NAME
        if fm_group is None:
            fm_group = [DEFAULT_GROUP_NAME]
        group_embedding_dict, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                             seed, support_group=True)
        #with open('../data/group_embedding_dict.txt', 'w') as f:
            #dump(group_embedding_dict, f)

        fm_logit = add_func([FM()(concat_func(v, axis=1))
                             for k, v in group_embedding_dict.items() if k in fm_group])

        output = tf.concat([output, tf.reshape(fm_logit, (-1, 1))], axis=-1)

    output = DNN([100, 50], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(output)
    output = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)
    output = PredictionLayer(task)(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


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
        g.add(Dense(self.class_unique, input_dim=self.g_input_dim, activation='softmax', kernel_constraint=constraint))
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

class KdbHighOrderFeatureEncoder:
    '''
    build a kdb model, use the dependency relationships to encode high order feature of given dataset.
    '''

    def __init__(self):
        self.dependencies_ = {}
        self.constraints_ = np.array([])
        self.have_value_idxs_ = []
        self.feature_uniques_ = []
        self.high_order_feature_uniques_ = []
        self.edges_ = []
        self.ohe = None

    def fit(self, X_train, y_train, k=2):
        '''
        build the kdb model, obtain the dependencies.
        '''
        edges = build_graph(X_train, y_train, k)
        n_classes = len(np.unique(y_train))
        num_features = X_train.shape[1]

        if k > 0:
            dependencies = _get_dependencies_without_y(list(range(num_features)), num_features, edges)
        else:
            dependencies = {x: [] for x in range(num_features)}

        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X_train[:, i])) for i in range(num_features)]
        self.edges_ = edges
        return self

    def transform(self, X, return_constraints=True):
        '''
        encode the high order feature,find corresbonding constraints info,
        then arrange to a proper format and return(store) it.
        '''
        high_order_features = []
        have_value_idxs = []
        constraints = []
        for k, v in self.dependencies_.items():
            hio = get_high_order_feature(X, k, v, self.feature_uniques_)
            idx, constraint = get_high_order_constraints(X, k, v, self.feature_uniques_)

            high_order_features.append(hio)
            have_value_idxs.append(idx)
            constraints.append(constraint)

        concated_constraints = np.hstack(constraints)
        concated_high_order_features = np.hstack(high_order_features)

        from sklearn.preprocessing import OneHotEncoder
        if self.ohe is None:
            self.ohe = OneHotEncoder()
            self.ohe.fit(concated_high_order_features)
        X_high_order = self.ohe.transform(concated_high_order_features)

        self.high_order_feature_uniques_ = [np.sum(constraint) for constraint in constraints]
        self.constraints_ = concated_constraints
        self.have_value_idxs_ = have_value_idxs

        if return_constraints:
            return X_high_order, concated_constraints, have_value_idxs
        else:
            return X_high_order

def data_process(data,gene_ratio=0.5,batch_size = 32,epochs = 200,warm_up_epochs=5,size_control=False):
    length = len(data)
    encoder = OrdinalEncoder()
    df = encoder.fit_transform(data).astype('int')
    X_train = df[:,0:-1]
    Y_train = df[:,-1]
    clear_session()
    ganblr = GANBLR()
    ganblr.fit(X_train,Y_train,epochs = epochs, warm_up_epochs=warm_up_epochs)
    data_gan = ganblr.generate(int(gene_ratio*len(data)))
    data_gan = pd.DataFrame(data_gan,columns=data.columns)
    print(type(data_gan))
    print(data_gan)
    data_new = pd.concat([data,data_gan],ignore_index=True)
    data_new = shuffle(data_new)
    if size_control == True:
        return data_new[0:length]
    return data_new

def data_process1(data,gene_ratio4=0.5,gene_ratio1=0.5,gene_ratio2=0.5,gene_ratio3=0.5,batch_size = 32,epochs = 200,warm_up_epochs=5):
    length = len(data)
    encoder = OrdinalEncoder()
    df = encoder.fit_transform(data).astype('int')
    X_train = df[:,0:-1]
    Y_train = df[:,-1]
    clear_session()
    ganblr = GANBLR()
    ganblr.fit(X_train,Y_train,epochs = epochs, warm_up_epochs=warm_up_epochs)
    data_gan1 = ganblr.generate(int(gene_ratio1*len(data)))
    data_gan1 = pd.DataFrame(data_gan1,columns=data.columns)
    data_gan2 = ganblr.generate(int(gene_ratio2 * len(data)))
    data_gan2 = pd.DataFrame(data_gan2, columns=data.columns)
    data_gan3 = ganblr.generate(int(gene_ratio3 * len(data)))
    data_gan3 = pd.DataFrame(data_gan3, columns=data.columns)
    data_gan4 = ganblr.generate(int(gene_ratio4 * len(data)))
    data_gan4 = pd.DataFrame(data_gan4, columns=data.columns)
    return data_gan1, data_gan2, data_gan3,data_gan4

test = pd.read_csv('./discretizedata-main/creditcard-mod-dm-encode.csv')
train = pd.read_csv('./discretizedata-main/creditcard-mod-dm-encode.csv')
data = pd.concat([test,train],ignore_index=True)
length = len(data)
sparse_features = data.columns[:-1]

data[sparse_features] = data[sparse_features].fillna('-1',)
target = data.columns[-1]

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

data[target] = LabelEncoder().fit_transform(data[target])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max()+1, embedding_dim=4)
                          for i, feat in enumerate(sparse_features)]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
accuracy = 0
accuracy_1 = 0
accuracy_2 = 0
accuracy_3 = 0
accuracy_4 = 0
f1 = 0
f1_1 = 0
f1_2 = 0
f1_3 = 0
f1_4 = 0
for i in range(5):
    train = data[1000:]
    traindata, test = train_test_split(train, test_size=0.99)
    test = data[0:1000]
    data_gan1, data_gan2, data_gan3, data_gan4= data_process1(train, gene_ratio1=0.04, gene_ratio2=0.09, gene_ratio3=0.19, gene_ratio4=0.49,epochs=2, warm_up_epochs=5, batch_size=32)
    train1 = pd.concat([traindata, data_gan1], ignore_index=True)
    train2 = pd.concat([traindata, data_gan2], ignore_index=True)
    train3 = pd.concat([traindata, data_gan3], ignore_index=True)
    train4 = pd.concat([traindata, data_gan4], ignore_index=True)
    #train = data_process(train, gene_ratio=4, epochs=200, warm_up_epochs=5, batch_size=32, size_control=False)
    #train = data_process(train, gene_ratio=1, epochs=200, warm_up_epochs=5, batch_size=32, size_control=False)
    #train = data_process(train, gene_ratio=1, epochs=200, warm_up_epochs=5, batch_size=32, size_control=False)
    train_model_input = {name: traindata[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    from ganblr.kdb import KdbHighOrderFeatureEncoder
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    enc = KdbHighOrderFeatureEncoder()

    # X_highorder_train = enc.fit_transform(train.iloc[:,:-1].values, train[target].values, k=2, return_constraints=False)
    # X_highorder_test = enc.transform(test.iloc[:,:-1].values, False)
    all = pd.concat([traindata, test], axis=0)
    X_highorder = enc.fit_transform(all.iloc[:, :-1].values, all[target].values, k=2, return_constraints=False)
    X_highorder_train = X_highorder[:traindata.shape[0]]
    X_highorder_test = X_highorder[traindata.shape[0]:]

    X_highorder_train_np = X_highorder_train.toarray()
    X_highorder_test_np = X_highorder_test.toarray()

    ohe = OneHotEncoder()
    # X_ohe_train = ohe.fit_transform(train.values[:,:-1]).toarray()
    # X_ohe_test = ohe.transform(test.values[:,:-1]).toarray()
    y_train = traindata.values[:, [-1]]
    y_test = test.values[:, [-1]]

    from copy import copy

    train_dict = copy(train_model_input)
    train_dict['broad'] = X_highorder_train_np

    test_dict = copy(test_model_input)
    test_dict['broad'] = X_highorder_test_np
    ########
    model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2)
    #model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2,use_fm=True)

    import keras_metrics
    tf.config.experimental_run_functions_eagerly(True)

    model.compile("adam", "binary_crossentropy",
                  metrics=[keras_metrics.f1_score()] )
    bdc_history = model.fit(train_dict, y_train, batch_size=256, epochs=20)
    test_pre = model.predict(test_dict)
    test_pre = test_pre[:, -1]
    for index in range(len(test_pre)):
        if test_pre[index] < 0.5:
            test_pre[index] = 0
        else:
            test_pre[index] = 1
    test_y = test[[target]].values[:, -1]
    correct = (test_pre == test_y)
    accuracy += correct.sum() / correct.size

    from sklearn.metrics import f1_score

    f1 += f1_score(test_y, test_pre, average='binary')

    #
    #
    #
    #
    #
    #
    #
    train_model_input = {name:train1[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    from ganblr.kdb import KdbHighOrderFeatureEncoder
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    enc = KdbHighOrderFeatureEncoder()

    # X_highorder_train = enc.fit_transform(train.iloc[:,:-1].values, train[target].values, k=2, return_constraints=False)
    # X_highorder_test = enc.transform(test.iloc[:,:-1].values, False)
    all = pd.concat([train1,test],axis=0)
    X_highorder = enc.fit_transform(all.iloc[:,:-1].values,  all[target].values, k=2, return_constraints=False)
    X_highorder_train = X_highorder[:train1.shape[0]]
    X_highorder_test = X_highorder[train1.shape[0]:]

    X_highorder_train_np = X_highorder_train.toarray()
    X_highorder_test_np = X_highorder_test.toarray()

    ohe = OneHotEncoder()
    #X_ohe_train = ohe.fit_transform(train.values[:,:-1]).toarray()
    #X_ohe_test = ohe.transform(test.values[:,:-1]).toarray()
    y_train = train1.values[:,[-1]]
    y_test = test.values[:,[-1]]

    from copy import copy
    train_dict = copy(train_model_input)
    train_dict['broad'] = X_highorder_train_np

    test_dict = copy(test_model_input)
    test_dict['broad'] = X_highorder_test_np
    model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns,2)
    #model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2,use_fm=True)
    model.compile("adam", "binary_crossentropy",
                  metrics=[keras_metrics.f1_score()])
    print(len(train_dict))
    print(len(y_train))
    bdc_history = model.fit(train_dict, y_train, batch_size=256, epochs=20)
    test_pre = model.predict(test_dict)
    test_pre = test_pre[:,-1]
    for index in range(len(test_pre)):
        if test_pre[index] < 0.5:
            test_pre[index] = 0
        else:
            test_pre[index] = 1
    test_y = test[[target]].values[:,-1]
    correct = (test_pre == test_y)
    accuracy_1 += correct.sum() / correct.size
    from sklearn.metrics import f1_score
    f1_1 += f1_score(test_y,test_pre,average='binary')

    #
    #
    #
    #
    #
    #
    #
    train_model_input = {name: train2[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    from ganblr.kdb import KdbHighOrderFeatureEncoder
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    enc = KdbHighOrderFeatureEncoder()

    # X_highorder_train = enc.fit_transform(train.iloc[:,:-1].values, train[target].values, k=2, return_constraints=False)
    # X_highorder_test = enc.transform(test.iloc[:,:-1].values, False)
    all = pd.concat([train2, test], axis=0)
    X_highorder = enc.fit_transform(all.iloc[:, :-1].values, all[target].values, k=2, return_constraints=False)
    X_highorder_train = X_highorder[:train2.shape[0]]
    X_highorder_test = X_highorder[train2.shape[0]:]

    X_highorder_train_np = X_highorder_train.toarray()
    X_highorder_test_np = X_highorder_test.toarray()

    ohe = OneHotEncoder()
    # X_ohe_train = ohe.fit_transform(train.values[:,:-1]).toarray()
    # X_ohe_test = ohe.transform(test.values[:,:-1]).toarray()
    y_train = train2.values[:, [-1]]
    y_test = test.values[:, [-1]]

    from copy import copy

    train_dict = copy(train_model_input)
    train_dict['broad'] = X_highorder_train_np

    test_dict = copy(test_model_input)
    test_dict['broad'] = X_highorder_test_np
    model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2)
    #model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2,use_fm=True)
    model.compile("adam", "binary_crossentropy",
                  metrics=[keras_metrics.f1_score()])
    bdc_history = model.fit(train_dict, y_train, batch_size=256, epochs=20)
    test_pre = model.predict(test_dict)
    test_pre = test_pre[:, -1]
    for index in range(len(test_pre)):
        if test_pre[index] < 0.5:
            test_pre[index] = 0
        else:
            test_pre[index] = 1
    test_y = test[[target]].values[:, -1]
    correct = (test_pre == test_y)
    accuracy_2 += correct.sum() / correct.size

    from sklearn.metrics import f1_score
    f1_2 += f1_score(test_y, test_pre, average='binary')

    #
    #
    #
    #
    #
    #
    #
    train_model_input = {name: train3[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    from ganblr.kdb import KdbHighOrderFeatureEncoder
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    enc = KdbHighOrderFeatureEncoder()

    # X_highorder_train = enc.fit_transform(train.iloc[:,:-1].values, train[target].values, k=2, return_constraints=False)
    # X_highorder_test = enc.transform(test.iloc[:,:-1].values, False)
    all = pd.concat([train3, test], axis=0)
    X_highorder = enc.fit_transform(all.iloc[:, :-1].values, all[target].values, k=2, return_constraints=False)
    X_highorder_train = X_highorder[:train3.shape[0]]
    X_highorder_test = X_highorder[train3.shape[0]:]

    X_highorder_train_np = X_highorder_train.toarray()
    X_highorder_test_np = X_highorder_test.toarray()

    ohe = OneHotEncoder()
    # X_ohe_train = ohe.fit_transform(train.values[:,:-1]).toarray()
    # X_ohe_test = ohe.transform(test.values[:,:-1]).toarray()
    y_train = train3.values[:, [-1]]
    y_test = test.values[:, [-1]]

    from copy import copy

    train_dict = copy(train_model_input)
    train_dict['broad'] = X_highorder_train_np

    test_dict = copy(test_model_input)
    test_dict['broad'] = X_highorder_test_np
    model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2)
    #model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2,use_fm=True)
    model.compile("adam", "binary_crossentropy",
                  metrics=[keras_metrics.f1_score()])
    bdc_history = model.fit(train_dict, y_train, batch_size=256, epochs=20)
    test_pre = model.predict(test_dict)
    test_pre = test_pre[:, -1]
    for index in range(len(test_pre)):
        if test_pre[index] < 0.5:
            test_pre[index] = 0
        else:
            test_pre[index] = 1
    test_y = test[[target]].values[:, -1]
    correct = (test_pre == test_y)
    accuracy_3 += correct.sum() / correct.size
    from sklearn.metrics import f1_score
    f1_3 += f1_score(test_y, test_pre, average='binary')

    #
    #
    #
    #
    #
    #
    #
    train_model_input = {name: train4[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    from ganblr.kdb import KdbHighOrderFeatureEncoder
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    enc = KdbHighOrderFeatureEncoder()

    # X_highorder_train = enc.fit_transform(train.iloc[:,:-1].values, train[target].values, k=2, return_constraints=False)
    # X_highorder_test = enc.transform(test.iloc[:,:-1].values, False)
    all = pd.concat([train4, test], axis=0)
    X_highorder = enc.fit_transform(all.iloc[:, :-1].values, all[target].values, k=2, return_constraints=False)
    X_highorder_train = X_highorder[:train4.shape[0]]
    X_highorder_test = X_highorder[train4.shape[0]:]

    X_highorder_train_np = X_highorder_train.toarray()
    X_highorder_test_np = X_highorder_test.toarray()

    ohe = OneHotEncoder()
    # X_ohe_train = ohe.fit_transform(train.values[:,:-1]).toarray()
    # X_ohe_test = ohe.transform(test.values[:,:-1]).toarray()
    y_train = train4.values[:, [-1]]
    y_test = test.values[:, [-1]]

    from copy import copy

    train_dict = copy(train_model_input)
    train_dict['broad'] = X_highorder_train_np

    test_dict = copy(test_model_input)
    test_dict['broad'] = X_highorder_test_np
    model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2)
    #model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2,use_fm=True)
    model.compile("adam", "binary_crossentropy",
                  metrics=[keras_metrics.f1_score()])
    bdc_history = model.fit(train_dict, y_train, batch_size=256, epochs=20)
    test_pre = model.predict(test_dict)
    test_pre = test_pre[:, -1]
    for index in range(len(test_pre)):
        if test_pre[index] < 0.5:
            test_pre[index] = 0
        else:
            test_pre[index] = 1
    test_y = test[[target]].values[:, -1]
    correct = (test_pre == test_y)
    accuracy_4 += correct.sum() / correct.size
    from sklearn.metrics import f1_score

    f1_4 += f1_score(test_y, test_pre, average='binary')
print(accuracy/5, accuracy_1/5, accuracy_2/5, accuracy_3/5, accuracy_4/5)
print(f1/5, f1_1/5, f1_2/5, f1_3/5, f1_4/5)