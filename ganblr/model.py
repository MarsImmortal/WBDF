from .kdb import *
import numpy as np
import tensorflow as tf
class GANBLR_G:
    def __init__(self, input_dim, output_dim, constraint) -> None:
        
        KL_LOSS = 0
        _loss = lambda y_true, y_pred: tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)+ KL_LOSS

        model = tf.keras.Sequential()
        model.add(Dense(2, input_dim=167, activation='softmax',kernel_constraint=constraint))
        model.compile(loss=_loss, optimizer='adam', metrics=['accuracy'])
        #model.set_weights(weights)

    def train_on_batch(self) -> float:
        
        pass
    
    def warmup_run(self):
        pass

    def sample_syenthetic(self) -> tuple(np.ndarray, np.ndarray):
        pass


class GANBLR_D:
    def __init__(self, input_dim, output_dim) -> None:
        model = tf.keras.Sequential()
        model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train_on_batch(self):
        pass

    

class GANBLR:
    def __init__(self) -> None:
        pass

    def fit():
        pass