import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.model_selection import KFold

import numpy as np

from matplotlib import pyplot as plt





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
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.set_xlim(-0.5,20)
            ax1.set_ylim(0.1,0.7)
  
            ax2.plot(history.history["val_accuracy"], label='validation')
            ax2.plot(history.history["accuracy"], label='training')
            ax2.legend(loc="lower right")
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlim(-0.5,20)
            ax2.set_ylim(0.5,0.9)
            fig.savefig(os.path.join(save_dir,f'plot_{i}'),bbox_inches='tight')
            fig.show()




