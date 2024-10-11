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
                l2_reg_embedding=0, l2_reg_dnn=0.0, l2_reg_cin=0.0, seed=1024, dnn_dropout=0.001,
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

    inputs_list = list(features.values())


    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    cin_input = concat_func(sparse_embedding_list, axis=1)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_output = tf.keras.layers.Dense(
        output_units, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    output = dnn_output

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
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
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

test = pd.read_csv('/WBDF/discretizedata-main/creditcard-mod-dm-encode.csv')
train = pd.read_csv('/WBDF/discretizedata-main/creditcard-mod-dm-encode.csv')
len_train = int(len(train))
data = pd.concat([train,test],ignore_index=True)
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
train = data.iloc[0:len_train]
test = data.iloc[len_train:]

train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

from ganblr.kdb import KdbHighOrderFeatureEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
enc = KdbHighOrderFeatureEncoder()

# X_highorder_train = enc.fit_transform(train.iloc[:,:-1].values, train[target].values, k=2, return_constraints=False)
# X_highorder_test = enc.transform(test.iloc[:,:-1].values, False)
all = pd.concat([train,test],axis=0)
X_highorder = enc.fit_transform(all.iloc[:,:-1].values,  all[target].values, k=2, return_constraints=False)
X_highorder_train = X_highorder[:train.shape[0]]
X_highorder_test = X_highorder[train.shape[0]:]

X_highorder_train_np = X_highorder_train.toarray()
X_highorder_test_np = X_highorder_test.toarray()

ohe = OneHotEncoder()
#X_ohe_train = ohe.fit_transform(train.values[:,:-1]).toarray()
#X_ohe_test = ohe.transform(test.values[:,:-1]).toarray()
y_train = train.values[:,[-1]]
y_test = test.values[:,[-1]]

from copy import copy
import keras_metrics
train_dict = copy(train_model_input)
train_dict['broad'] = X_highorder_train_np

test_dict = copy(test_model_input)
test_dict['broad'] = X_highorder_test_np

accuracy = 0
model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns,2)
#model = elr_dnn_cin(softmax_weight(enc.constraints_), X_highorder_train.shape[1], dnn_feature_columns, 2,use_fm=True)
tf.config.experimental_run_functions_eagerly(True)
model.compile("adam", "binary_crossentropy",
          metrics=['accuracy'], )
model.fit(train_dict, y_train, batch_size=256, epochs=20, validation_split=0.2)
result = model.predict(test_dict)
accuracy = model.evaluate(test_dict, test[[target]].values, batch_size=256)[1]

print(accuracy)