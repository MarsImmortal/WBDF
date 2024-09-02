import numpy as np
import pandas as pd
import networkx as nx
from pyitlib import discrete_random_variable as drv
from warnings import warn
from tqdm import tqdm

def build_graph(X_train, y_train, k=2):
    '''
    kDB algorithm

    Param:
    ----------------------

    Return:
    ----------------------
    graph edges
    '''
    #ensure data
    num_features = X_train.shape[1]
    x_nodes = list(range(num_features))
    y_node  = num_features
    #X_train = X_train.to_numpy()
    #y_train = y_train.to_numpy()

    #util func
    _x = lambda i:X_train[:,i]
    _x2comb = lambda i,j:(X_train[:,i], X_train[:,j])

    #feature indexes desc sort by mutual information
    sorted_feature_idxs = np.argsort([
    drv.information_mutual(_x(i), y_train)
    for i in range(num_features)
    ])[::-1]

    #start building graph
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
    pos=nx.spiral_layout(graph)
    nx.draw(graph, pos, node_color='r', edge_color='b')
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")

from tensorflow.keras.layers import Input,Embedding, Dense,Dropout, concatenate, Flatten
from tensorflow.python.keras.regularizers import l1_l2, l2
from tensorflow.keras.models import Model
from pandas import DataFrame, Series, crosstab
import numpy as np
import tensorflow as tf

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

from sklearn.preprocessing import OneHotEncoder

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
            dependencies = {x:[] for x in range(num_features)}

        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X_train[:,i])) for i in range(num_features)]
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
        X_high_order = OneHotEncoder().fit_transform(concated_high_order_features)

        self.high_order_feature_uniques_ = [np.sum(constraint) for constraint in constraints]
        self.constraints_ = concated_constraints
        self.have_value_idxs_ = have_value_idxs

        if return_constraints:
            return X_high_order, concated_constraints, have_value_idxs
        else:
            return X_high_order

    def fit_transform(self, X, y, k=2, return_constraints=True):
        return self.fit(X, y, k).transform(X, return_constraints)

from tensorflow.python.ops import math_ops, array_ops
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.activations import softmax
class softmax_weight(Constraint):
    """Constrains weight tensors to be under softmax `."""

    def __init__(self,feature_uniques):
        if isinstance(feature_uniques, np.ndarray):
            idxs = math_ops.cumsum(np.hstack([np.array([0]),feature_uniques]))
        else:
            idxs = math_ops.cumsum([0] + feature_uniques)
        idxs = [i.numpy() for i in idxs]
        self.feature_idxs = [
            (idxs[i],idxs[i+1]) for i in range(len(idxs)-1)
        ]

    def __call__(self, w):
        w_new = [
            math_ops.log(softmax(w[i:j,:], axis=0))
            for i,j in self.feature_idxs
        ]
        return tf.concat(w_new, 0)

    def get_config(self):
        return {'feature_idxs': self.feature_idxs}

from tensorflow.keras.layers import Dense
def get_lr(input_dim, output_dim, constraint=None):
    model = tf.keras.Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax',kernel_constraint=constraint))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def sample_synthetic_data(weights, kdb_high_order_encoder, y_counts, ohe=True,size=None):
    from pgmpy.models import BayesianModel
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.factors.discrete import TabularCPD
    #basic varibles
    feature_cards = np.array(kdb_high_order_encoder.feature_uniques_)
    n_features = len(feature_cards)
    n_classes = weights.shape[1]
    n_samples = y_counts.sum()

    #ensure sum of each constraint group equals to 1, then re concat the probs
    _idxs = np.cumsum([0] + kdb_high_order_encoder.constraints_.tolist())
    constraint_idxs = [(_idxs[i],_idxs[i+1]) for i in range(len(_idxs)-1)]

    probs = np.exp(weights)
    cpd_probs = [probs[start:end,:] for start, end in constraint_idxs]
    cpd_probs = np.vstack([p/p.sum(axis=0) for p in cpd_probs])

    #assign the probs to the full cpd tables
    idxs = np.cumsum([0] + kdb_high_order_encoder.high_order_feature_uniques_)
    feature_idxs = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    have_value_idxs = kdb_high_order_encoder.have_value_idxs_
    full_cpd_probs = []
    for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
        #(n_high_order_feature_uniques, n_classes)
        cpd_prob_ = cpd_probs[start:end,:]
        #(n_all_combination) Note: the order is (*parent, variable)
        have_value_ravel = have_value.ravel()
        #(n_classes * n_all_combination)
        have_value_ravel_repeat = np.hstack([have_value_ravel] * n_classes)
        #(n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
        full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
        full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
        #(n_classes * n_parent_combinations, n_variable_unique)
        full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
        full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
        full_cpd_probs.append(full_cpd_prob)

    #prepare node and edge names
    node_names = [str(i) for i in range(n_features + 1)]
    edge_names = [(str(i), str(j)) for i,j in kdb_high_order_encoder.edges_]
    y_name = node_names[-1]

    #create TabularCPD objects
    evidences = kdb_high_order_encoder.dependencies_
    feature_cpds = [
        TabularCPD(str(name), feature_cards[name], table,
                   evidence=[y_name, *[str(e) for e in evidences]],
                   evidence_card=[n_classes, *feature_cards[evidences].tolist()])
        for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
    ]
    y_probs = (y_counts/n_samples).reshape(-1,1)
    y_cpd = TabularCPD(y_name, n_classes, y_probs)

    #create kDB model, then sample data
    model = BayesianModel(edge_names)
    model.add_cpds(y_cpd, *feature_cpds)
    sample_size = n_samples if size is None else size
    result = BayesianModelSampling(model).forward_sample(size=sample_size)
    sorted_result = result[node_names].values

    #return
    syn_X, syn_y = sorted_result[:,:-1], sorted_result[:,-1]
    if ohe:
        from sklearn.preprocessing import OneHotEncoder
        ohe_syn_X = OneHotEncoder().fit_transform(syn_X)
        return ohe_syn_X, syn_y
    else:
        return syn_X, syn_y


from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from tensorflow.keras.backend import clear_session
from itertools import islice
from json import dumps
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#datasets_path = Path('../../uci-datasets/mdl')
#outputs_path = Path('../data/outputs')
#outputs_path.mkdir(exist_ok=True)
#outputs_path = outputs_path/'2020331/'
#outputs_path.mkdir(exist_ok=True)
epochs = 100
batch_size = 32

def run(X, y, train_idxs, test_idxs):

    kdb_high_order_encoder = KdbHighOrderFeatureEncoder()
    X_high_order = kdb_high_order_encoder.fit_transform(X, y, return_constraints=False)
    train_data = X_high_order[train_idxs], y[train_idxs]
    test_data  = X_high_order[test_idxs] , y[test_idxs]
    feature_uniques = [len(np.unique(X[:,i])) for i in range(X.shape[1])]
    class_unique = len(np.unique(y))
    logs = []

    clear_session()
    constraint = softmax_weight(kdb_high_order_encoder.constraints_)
    elr = get_lr(X_high_order.shape[1], class_unique, constraint)
    log_elr = elr.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=epochs)
    logs.append(dict(
        batch_size=batch_size,
        epochs=epochs,
        model='elr',
        data=log_elr.history
    ))

    ohe_X = OneHotEncoder().fit_transform(X)
    ohe_train_data = ohe_X[train_idxs], y[train_idxs]
    ohe_test_data  = ohe_X[test_idxs] , y[test_idxs]
    # clear_session()
    # lr = get_lr(ohe_X.shape[1], class_unique)
    # log_lr = lr.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=epochs)
    # logs.append(dict(
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     model='lr',
    #     data=log_lr.history
    # ))

    clear_session()
    weights = elr.get_weights()[0]
    _, y_counts = np.unique(y[train_idxs], return_counts=True)
    syn_data = sample_synthetic_data(weights, kdb_high_order_encoder, y_counts)

    lr_syn = get_lr(ohe_X.shape[1], class_unique)
    log_lr_syn = lr_syn.fit(*syn_data, validation_data=ohe_test_data, batch_size=batch_size,epochs=epochs)
    logs.append(dict(
        batch_size=batch_size,
        epochs=epochs,
        model='lr(synthetic data)',
        data=log_lr_syn.history
    ))
    return syn_data


df = pd.read_csv('./data_generated/train_magic.csv')
df = OrdinalEncoder().fit_transform(df).astype('int')
print(df)
print(df.shape)
X = df[:,0:-1]
y = df[:,-1]
for i, (train_idxs, test_idxs) in enumerate(KFold(n_splits=2, random_state=42, shuffle=True).split(X)):
    print('-----------%d of 3 Fold Cross Validation--------------' % (i+1))
    if i > 0:
        break
    syn_data = run(X, y, train_idxs, test_idxs)