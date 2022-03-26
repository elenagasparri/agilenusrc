class BlazarDnn:
    ''' Class describing the buildt and training of a deep neural network designed
    to identify blazars among agns.
    '''
        
    def checkgpu(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))
        return device_name
        
    def loadData(self,filename):
        '''Funzione che riceve come input un archivio numpy con il dataset 'filename.npz' e ritorna un ndarray
        unico con il dataset completo e le label. Si suppone che l'archivio sia composto da tre array nominati
        'bl_data','agn_data' e 'nn_freq_data'.
        '''
        nn_data = np.load(filename)
        bl_data = nn_data['bl_data']
        agn_data = nn_data['agn_data']
        set_freq = nn_data['nn_freq_data']
        data = np.concatenate((bl_data, agn_data), axis =0)
        label = np.concatenate((np.ones(bl_data.shape[0]),np.zeros(agn_data.shape[0])))
        return data, label
        
    def rescaleData(self,data):
        ''' Funzione che normalizza il dataset tra [0,1].
        '''
        ptp=np.ptp(data[:,:,0])
        min = np.min(data[:,:,0])
        max = np.max(data[:,:,0])
        data[:,:,0]= (data[:,:,0]-min)/ptp
        return data
    
    def get_model_name(self,k):
        return 'model_'+str(k)+'.h5'

    def trainingModel(self,data,n_splits,save_dir):
        kf = KFold(n_splits=n_splits, shuffle = True, random_state = 1)
        save_dir = save_dir
        fold_var = 0
        history_list = []
        with tf.device('/device:GPU:0'):
            for train_idx, test_idx in kf.split(data):
                fold_var = fold_var+1
                # CREATE CALLBACKS
                checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+run.get_model_name(fold_var), 
							        monitor='val_accuracy', verbose=1, 
							        save_best_only=True, mode='max')
                callbacks_list = [checkpoint]
                history=model.fit(x=data[train_idx],y=label[train_idx],validation_data=(data[test_idx],label[test_idx]),epochs=20, callbacks=callbacks_list)
                np.save(os.path.join(save_dir,f'my_history_{fold_var}.npy'),history.history)
                history_list.append(history)
        return hystory_list

    def plotandsaveHistory(self,history_list):
        '''Function that plot and save the graph of Loss and Accuracy both for training
        and validation of each fold.
        '''
        fold_var = 0
        for history in history_list:
            fold_var = fold_var+1
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f'Loss and Accuracy plot for both Training and Validation (fold_{i})')

            ax1.plot(history.history["val_loss"], label='validation')
            ax1.plot(history.history["loss"], label='training')
            ax1.legend(loc="upper right")
            ax1.xlabel('Epochs')
            ax1.ylabel('Loss')
            ax1.xlim(-0.5,20)
            ax1.ylim(0.1,0.7)
  
            ax2.plot(history.history["val_accuracy"], label='validation')
            ax2.plot(history.history["accuracy"], label='training')
            ax2.legend(loc="lower right")
            ax2.xlabel('Epochs')
            ax2.ylabel('Accuracy')
            ax2.xlim(-0.5,20)
            ax2.ylim(0.5,0.9)
            fig.savefig(os.path.join(save_dir,f'plot_{i}'),bbox_inches='tight')
            fig.show()




import tensorflow as tf
from tensorflow import keras
from keras import layers
print("TensorFlow version:", tf.__version__)

from sklearn.model_selection import KFold

import numpy as np
print('Numpy version:', np.__version__)

from matplotlib import pyplot as plt

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
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) # ridotto learning rate di un fattore 10
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy','binary_crossentropy'])

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
