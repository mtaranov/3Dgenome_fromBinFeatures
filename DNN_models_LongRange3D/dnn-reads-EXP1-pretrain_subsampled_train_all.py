import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import copy
from models import LongRangeDNN
from utils import get_features, get_labels, subsample_data, normalize_features, reconstruct_2d, printMatrix, binarize, zscore, get_2D, plot_prediction
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from dragonn.models import Model, SequenceDNN
#from keras.models import Sequential
#from keras.callbacks import Callback, EarlyStopping
#from keras.layers.core import (
#    Activation, Dense, Dropout, Flatten,
#    Permute, Reshape, TimeDistributedDense
#)
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.layers.recurrent import GRU
#from keras.regularizers import l1

#from deeplift import keras_conversion as kc
#from deeplift.blobs import MxtsMode


# In[2]:

data_path='/users/mtaranov/NN_all_data/'

X_train = get_features('/users/mtaranov/NN_all_data/train_set_upperTriangle_noDiag_reads.npy')
y_train = get_labels('/users/mtaranov/NN_all_data/labels_train_upperTriangle_noDiag.npy')
X_valid = get_features('/users/mtaranov/NN_all_data/vali_set_upperTriangle_noDiag_reads.npy')
y_valid = get_labels('/users/mtaranov/NN_all_data/labels_vali_upperTriangle_noDiag.npy')
X_test = get_features('/users/mtaranov/NN_all_data/test_set_upperTriangle_noDiag_reads.npy')
y_test = get_labels('/users/mtaranov/NN_all_data/labels_test_upperTriangle_noDiag.npy')

"""
X_train = get_features(data_path+'train_set_upperTriangle_noDiag_reads.npy')
y_train = get_labels(data_path+'labels_train_upperTriangle_noDiag.npy')
X_valid = get_features(data_path+'vali_set_upperTriangle_noDiag_reads.npy')
y_valid = get_labels(data_path+'labels_vali_upperTriangle_noDiag.npy')
X_test = get_features(data_path+'test_set_upperTriangle_noDiag_reads.npy')
y_test = get_labels(data_path+'labels_test_upperTriangle_noDiag.npy')

X_train = get_features('NN_datasets/train_set_all_reads.npy')
y_train = get_labels('NN_datasets/labels_train_all.npy')
X_valid = get_features('NN_datasets/vali_set_all_reads.npy')
y_valid = get_labels('NN_datasets/labels_vali_all.npy')
X_test = get_features('NN_datasets/test_set_all_reads.npy')
y_test = get_labels('NN_datasets/labels_test_all.npy')

X_train_subsampled, y_train_subsampled = subsample_data(X_train, y_train)
X_valid_subsampled, y_valid_subsampled = subsample_data(X_valid, y_valid)
X_test_subsampled, y_test_subsampled = subsample_data(X_test, y_test)
"""


# In[3]:


#X_train_scaled, X_valid_scaled, X_test_scaled = normalize_features(X_train, X_valid, X_test)
#X_train_normalized, X_valid_normalized, X_test_normalized = normalize_features(X_train, X_valid, X_test,
#                                                                             normalizer=StandardScaler)
                                                                         


# In[4]:

"""
X_train_normalized_subsampled, y_train_subsampled = subsample_data(X_train_normalized, y_train)
X_valid_normalized_subsampled, y_valid_subsampled = subsample_data(X_valid_normalized, y_valid)
X_test_normalized_subsampled, y_test_subsampled = subsample_data(X_test_normalized, y_test)

X_train_scaled_subsampled, y_train_subsampled = subsample_data(X_train_scaled, y_train)
X_valid_scaled_subsampled, y_valid_subsampled = subsample_data(X_valid_scaled, y_valid)
X_test_scaled_subsampled, y_test_subsampled = subsample_data(X_test_scaled, y_test)
"""


# In[5]:

X_train_normalized, X_valid_normalized, X_test_normalized = normalize_features(X_train, X_valid, X_test)

X_train_normalized_subsampled, y_train_subsampled = subsample_data(X_train_normalized, y_train)
X_valid_normalized_subsampled, y_valid_subsampled = subsample_data(X_valid_normalized, y_valid)
X_test_normalized_subsampled, y_test_subsampled = subsample_data(X_test_normalized, y_test)


#X_train_scaled_subsampled, y_train_subsampled = subsample_data(X_train_scaled, y_train)
#X_valid_scaled_subsampled, y_valid_subsampled = subsample_data(X_valid_scaled, y_valid)
#X_test_scaled_subsampled, y_test_subsampled = subsample_data(X_test_scaled, y_test)


# # pretrain on subsampled

# In[6]:

dnn = LongRangeDNN(num_features=11, use_deep_CNN=True)


# In[10]:

validation_data = (X_valid_normalized_subsampled[:, :, :11, :], y_valid_subsampled)
#validation_data = (X_test_normalized_subsampled[:, :, :10, :], y_test_subsampled)
#validation_data = (X_test_subsampled[:, :, :10, :], y_test_subsampled)

dnn.train(X_train_normalized_subsampled[:, :, :11, :], y_train_subsampled, validation_data)
#dnn_normalized.train(X_test_normalized_subsampled[:, :, :10, :], y_test_subsampled, validation_data)
#dnn_normalized.train(X_test_subsampled[:, :, :10, :], y_test_subsampled, validation_data)


# In[12]:

print "Performance on Test: ", (dnn.test(X_test_normalized[:, :, :11, :], y_test))

dnn.train(X_train_normalized[:, :, :11, :], y_train, (X_valid_normalized, y_valid))

print "Performance on Test: ", (dnn.test(X_test_normalized[:, :, :11, :], y_test))

pred_probs_test = dnn.predict(X_test_normalized[:, :, :11, :])
np.save("model_predictions/test_set_upperTriangle_noDiag_reads_with_distances_pretr_subs_train_all.npy", pred_probs_test)
pred_probs_valid = dnn.predict(X_valid_normalized[:, :, :11, :])
np.save("model_predictions/valid_set_upperTriangle_noDiag_reads_with_distances_pretr_subs_train_all.npy", pred_probs_valid)
pred_probs_train = dnn.predict(X_train_normalized[:, :, :11, :])
np.save("model_predictions/train_set_upperTriangle_noDiag_reads_with_distances_pretr_subs_train_all.npy", pred_probs_train)
