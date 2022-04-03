import tensorflow as tf
from tensorflow import keras
from keras import layers
print("TensorFlow version:", tf.__version__)

from sklearn.model_selection import KFold

import numpy as np
print('Numpy version:', np.__version__)

from matplotlib import pyplot as plt

from agilenusrc.blazardnn import BlazarDNN

import os
current_dir = os.getcwd()
if os.path.isfile('nn_data_v3.npz'):
    filename = 'nn_data_v3.npz'
    print ("Dataset loaded in filename")
else:
    raise SystemError(f'Archive with dataset not found in current diretory {current_dir}!')




version = 4 #versioning
save_dir = os.path.join(current_dir,f'model_v{version}/')

run = BlazarDnn()
device_name = run.checkgpu
data, label = run.loadData(filename)
data_norm = run.rescaleData(data)

permutation = np.random.permutation(data_norm.shape[0])
data=data[permutation]
label=label[permutation]

model = keras.Sequential()
model.add(keras.Input(shape = (529,2))) # input
model.add(keras.layers.Conv1D(filters = 64, kernel_size = 3, strides=1, padding='same', dilation_rate=1,
                              activation=None, use_bias=True))
model.add(layers.Activation('tanh'))
model.add(keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid'))
model.add(keras.layers.Conv1D(filters = 128, kernel_size = 3, strides=1, padding='same', dilation_rate=1,
                              activation=None, use_bias=True))
model.add(layers.Activation('tanh'))
model.add(keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid'))
model.add(keras.layers.Conv1D(filters = 256, kernel_size = 3, strides=1, padding='same', dilation_rate=1,
                              activation=None, use_bias=True))
model.add(layers.Activation('tanh'))
model.add(keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid'))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(304)))
model.add(keras.layers.Dense(152, activation='relu', use_bias=True))
model.add(keras.layers.Dense(152, activation='relu', use_bias=True))
model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))
#opt = tf.keras.optimizers.Adam(learning_rate=0.0001) # ridotto learning rate di un fattore 10
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy','binary_crossentropy'])

model.summary()

n_splits = 10 # number of splits for Kfold cross validation

history_list = run.trainingModel(data_norm,n_splits,save_dir)

run.plotandsaveHistory(history_list)

# ROC Curve and AUC value with error for both Train e Test
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
i=0
for train_idx, test_idx in kf.split(data_norm):
    i = i+1
    probs = model.predict(data[test_idx])
    y_preds_probs = np.argmax(probs,axis=1)
    fpr, tpr, _ = metrics.roc_curve(label[test_idx], y_preds_probs)
    roc_auc = metrics.auc(fpr, tpr)

    probs_train = model.predict(data[train_idx])
    y_preds_probs_train = np.argmax(probs_train,axis=1)
    fpr_1, tpr_1, _= metrics.roc_curve(label[train_idx], y_preds_probs_train)
    roc_auc_1 = metrics.auc(fpr_1, tpr_1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Receiver Operating Characteristic and AUC value for both Train e Test (fold_{i})')

    ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax1.legend(loc = 'lower right')
    ax1.plot([0, 1], [0, 1],'r--')
    ax1.xlim([0, 1])
    ax1.ylim([0, 1])
    ax1.ylabel('True Positive Rate')
    ax1.xlabel('False Positive Rate')

    ax2.plot(fpr_1, tpr_1, 'b', label = 'AUC-train = %0.2f' % roc_auc_1)
    ax2.legend(loc = 'lower right')
    ax2.plot([0, 1], [0, 1],'r--')
    ax2.xlim([0, 1])
    ax2.ylim([0, 1])
    ax2.ylabel('True Positive Rate')
    ax2.xlabel('False Positive Rate')

    fig.savefig(os.path.join(save_dir,f'auc_{i}'),bbox_inches='tight')
    fig.show()
