# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer as glorot_normal

from tensorflow.python.keras.layers import Layer, Dropout

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization
from tensorflow.python.keras.regularizers import l2

from .activation import activation_layer


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")
        self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)

        super(LocalActivationUnit, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = self.dnn(att_input, training=training)

        attention_score = tf.nn.bias_add(tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias)

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import activations

class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, 
                 hidden_units, 
                 activation='relu', 
                 l2_reg=0, 
                 dropout_rate=0, 
                 use_bn=False, 
                 output_activation=None,
                 seed=1024, 
                 **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        # Placeholders for layer instances
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []

    def build(self, input_shape):
        for i, units in enumerate(self.hidden_units):
            # Define Dense layers
            self.dense_layers.append(
                Dense(units,
                      activation=None,  # Activation applied later
                      kernel_initializer=glorot_normal(seed=self.seed),
                      kernel_regularizer=l2(self.l2_reg))
            )

            # Optional BatchNormalization
            if self.use_bn:
                self.bn_layers.append(BatchNormalization())

            # Optional Dropout
            if self.dropout_rate > 0:
                self.dropout_layers.append(Dropout(self.dropout_rate, seed=self.seed + i))

        super(DNN, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = inputs

        for i, dense in enumerate(self.dense_layers):
            x = dense(x)  # Linear transformation

            # Batch Normalization if enabled
            if self.use_bn:
                x = self.bn_layers[i](x, training=training)

            # Activation function
            act_func = self.activation if (i < len(self.dense_layers) - 1 or self.output_activation is None) else self.output_activation
            x = activations.get(act_func)(x)

            # Dropout if enabled
            if self.dropout_rate > 0:
                x = self.dropout_layers[i](x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
        return shape

    def get_config(self):
        config = super(DNN, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            'use_bn': self.use_bn,
            'output_activation': self.output_activation,
            'seed': self.seed
        })
        return config

class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RegulationModule(Layer):
    """Regulation module used in EDCN.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size,field_size * embedding_size)``.

      Arguments
        - **tau** : Positive float, the temperature coefficient to control
        distribution of field-wise gating unit.

      References
        - [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models.](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)
    """

    def __init__(self, tau=1.0, **kwargs):
        if tau == 0:
            raise ValueError("RegulationModule tau can not be zero.")
        self.tau = 1.0 / tau
        super(RegulationModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_size = int(input_shape[1])
        self.embedding_size = int(input_shape[2])
        self.g = self.add_weight(
            shape=(1, self.field_size, 1),
            initializer=Ones(),
            name=self.name + '_field_weight')

        # Be sure to call this somewhere!
        super(RegulationModule, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        feild_gating_score = tf.nn.softmax(self.g * self.tau, 1)
        E = inputs * feild_gating_score
        return tf.reshape(E, [-1, self.field_size * self.embedding_size])

    def compute_output_shape(self, input_shape):
        return (None, self.field_size * self.embedding_size)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(RegulationModule, self).get_config()
        base_config.update(config)
        return base_config
