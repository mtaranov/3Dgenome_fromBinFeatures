import numpy as np
import os
from abc import abstractmethod, ABCMeta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l1

from metrics import ClassificationResult 
#from deeplift import keras_conversion as kc
#from deeplift.blobs import MxtsMode

from sklearn.svm import SVC as scikit_SVC
from sklearn.tree import DecisionTreeClassifier as scikit_DecisionTree
from sklearn.ensemble import RandomForestClassifier

class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **hyperparameters):
        pass

    @abstractmethod
    def train(self, X, y, validation_data):
        pass
    @abstractmethod
    def predict(self, X):
        pass

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)[metric]

class LongRangeDNN(Model):

    class PrintMetrics(Callback):

        def __init__(self, validation_data, sequence_DNN):
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN

        def on_epoch_end(self, epoch, logs={}):
            print('Epoch {}: validation loss: {:.3f}\n{}\n'.format(
                epoch,
                logs['val_loss'],
                self.sequence_DNN.test(self.X_valid, self.y_valid)))

    class LossHistory(Callback):

        def __init__(self, X_train, y_train, validation_data, sequence_DNN):
            self.X_train = X_train
            self.y_train = y_train
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN
            self.train_losses = []
            self.valid_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(self.sequence_DNN.model.evaluate(
                self.X_train, self.y_train, verbose=False))
            self.valid_losses.append(self.sequence_DNN.model.evaluate(
                self.X_valid, self.y_valid, verbose=False))


    def __init__(self, num_features=11, num_nodes=2, use_deep_CNN=False,
                  num_tasks=1, num_filters=25,
                  num_filters_2=25, num_filters_3=25,
                  L1=0, dropout=0.0, verbose=2):
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.input_shape = (1, num_features, num_nodes)
        self.num_tasks = num_tasks
        self.verbose = verbose
        self.model = Sequential()
        self.model.add(Convolution2D(
            nb_filter=num_filters, nb_row=num_features,
            nb_col=1, activation='linear',
            init='he_normal', input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        if use_deep_CNN:
            self.model.add(Convolution2D(
                nb_filter=num_filters_2, nb_row=1,
                nb_col=1, activation='relu',
                init='he_normal', W_regularizer=l1(L1)))
            self.model.add(Dropout(dropout))
            self.model.add(Convolution2D(
                nb_filter=num_filters_3, nb_row=1,
                nb_col=1, activation='relu',
                init='he_normal', W_regularizer=l1(L1)))
            self.model.add(Dropout(dropout))
        self.model.add(Flatten())
        self.model.add(Dense(output_dim=25, activation='relu'))
        self.model.add(Dense(output_dim=25, activation='relu'))
        self.model.add(Dense(output_dim=25, activation='relu'))
        self.model.add(Dense(output_dim=self.num_tasks))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.train_losses = None
        self.valid_losses = None

    def train(self, X, y, validation_data):
        if y.dtype != bool:
            assert len(np.unique(y)) == 2
            y = y.astype(bool)
        multitask = y.shape[1] > 1
        if not multitask:
            num_positives = y.sum()
            num_sequences = len(y)
            num_negatives = num_sequences - num_positives
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        if self.verbose >= 1:
            self.callbacks.append(self.PrintMetrics(validation_data, self))
            print('Training model...')
        self.callbacks.append(self.LossHistory(X, y, validation_data, self))
        self.model.fit(
            X, y, batch_size=250, nb_epoch=100,
            validation_data=validation_data,
            class_weight={True: num_sequences / num_positives,
                          False: num_sequences / num_negatives}
            if not multitask else None,
            callbacks=self.callbacks, verbose=self.verbose >= 2)
        self.train_losses = self.callbacks[-1].train_losses
        self.valid_losses = self.callbacks[-1].valid_losses
           
    def predict(self, X):
        return self.model.predict(X, batch_size=128, verbose=False) 

class DecisionTree(Model):

    def __init__(self):
        self.classifier = scikit_DecisionTree()

    def train(self, X, y, validation_data=None):
        self.classifier.fit(X, y)

    def predict(self, X):
        predictions = np.asarray(self.classifier.predict_proba(X))[..., 1]
        if len(predictions.shape) == 2:  # multitask
            predictions = predictions.T
        else:  # single-task
            predictions = np.expand_dims(predictions, 1)
        return predictions


class RandomForest(DecisionTree):

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)

class SVC(Model):

    def __init__(self):
        #self.classifier = scikit_SVC(probability=True, kernel='linear')
        self.classifier = scikit_SVC(probability=True, kernel='rbf')

    def train(self, X, y, validation_data=None):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1:]
