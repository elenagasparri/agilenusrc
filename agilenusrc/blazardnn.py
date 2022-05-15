import tensorflow as tf
from tensorflow import keras
from keras import layers
print("TensorFlow version:", tf.__version__)

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import numpy as np
print('Numpy version:', np.__version__)

from matplotlib import pyplot as plt

from agilenusrc.blazardnnfunc import BlazarDNN

import os
current_dir = os.getcwd()
if os.path.isfile('nn_data.npz'):
    filename = 'nn_data.npz'
    print ("Dataset present in current directory")
else:
    raise SystemError(f'Archive with dataset not found in current diretory {current_dir}!')




version = 6 #versioning
save_dir = os.path.join(current_dir,f'model_v{version}/')

run = BlazarDnn(filename)
device_name = run.checkgpu
data, label = run.loadData()
data_norm = run.rescaleData(data)

# balancing the split of dataset
bl=4627                      #number of blazar sources
agn=4059					 #number og agns sources
bl_data = data_norm[0:bl-1]
agn_data = data_norm[bl:]
bl_label = label[0:bl-1]
agn_label = label[bl:]

data_train_bl, data_test_bl, l_train_bl, l_test_bl = train_test_split(bl_data, bl_label, test_size=0.1, random_state=1)
data_train_agn, data_test_agn, l_train_agn, l_test_agn = train_test_split(agn_data, agn_label, test_size=0.1, random_state=1)

balance=len(data_train_agn)

data_train_bl_tmp = data_train_bl[0:balance-1]
l_train_bl_tmp = l_train_bl[0:balance-1]
data_train = np.concatenate((data_train_bl_tmp,data_train_agn))
data_test_bl = np.concatenate((data_test_bl, data_train_bl[balance:]))
l_test_bl = np.concatenate((l_test_bl,l_train_bl[balance:]))
l_train_bl = l_train_bl_tmp

label = np.concatenate((l_train_bl, l_train_agn))

l_test=np.concatenate((l_test_bl,l_test_agn))
data_test=np.concatenate((data_test_bl,data_test_agn))

#Permutation of the training dataset
permutation = np.random.permutation(data_train.shape[0])
data_train=data_train[permutation]
label=label[permutation]

# NN input shape
ishape=data.shape[1:]

#Bidirectional LSTM neural network model
init = tf.keras.initializers.LecunNormal(5)

init = tf.keras.initializers.LecunNormal(5)

model = keras.Sequential()
model.add(keras.Input(shape = ishape)) # input
model.add(keras.layers.Conv1D(filters = 128,kernel_initializer=init, kernel_size = 3, strides=1, padding='same',
    activation=None))
model.add(layers.Activation('tanh'))
model.add(keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Conv1D(filters = 256,kernel_initializer=init, kernel_size = 3, strides=1, padding='same',
    activation=None))
model.add(layers.Activation('tanh'))
model.add(keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Conv1D(filters = 512,kernel_initializer=init, kernel_size = 3, strides=1, padding='same',
    activation=None))
model.add(layers.Activation('tanh'))
model.add(keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(1024)))
model.add(keras.layers.Dense(512, activation='relu',kernel_initializer=init, use_bias=True))
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(256, activation='relu', kernel_initializer=init, use_bias=True))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation='relu', kernel_initializer=init, use_bias=True))

opt = tf.keras.optimizers.Adadelta(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


# Training
n_splits = 9 # number of splits for Kfold cross validation

history_list = run.trainingModel(data_train,n_splits,save_dir)

run.plotandsaveHistory(history_list)

# ROC Curve and AUC value with error for both Train e Test
import sklearn.metrics as metrics
from keras import models

plt.figure()
for i in range(9):
  model=models.load_model(f'model_v6/model_{i+1}.h5')
  # calculate the fpr and tpr for all thresholds of the classification
  probs = model.predict(data_test)
  fpr, tpr, threshold = metrics.roc_curve(l_test, probs)
  roc_auc = metrics.auc(fpr, tpr)

  plt.plot(fpr, tpr, label = f'AUC{i+1} = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')

plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(save_dir,'plot_auc')
plt.show()
