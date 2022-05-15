import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt





class BlazarDnn:
	''' Class describing functions for training a deep neural network designed
    to identify blazars among agns.
    '''
    
    def __init__(self, filename):
		""" Class constructor:
		    
		Parameters
		----------
		filename: string (.npz file object)
				Inside the archive is supposed to found three N-D array named 'bl_data','agn_data' e 'nn_freq_data'.
		"""
		    
		self.filename = os.path.join(os.path.abspath(os.getcwd()),filename)
	  
    def checkgpu(self):
		'''Return the name of the Device and check if the GPU has been found.
        '''
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))
        return device_name
        
    def loadData(self):
        '''Return data and categorical labels for the training, validation and testing of the dnn.
        '''
        nn_data = np.load(self.filename)
        bl_data = nn_data['bl_data']
        agn_data = nn_data['agn_data']
        set_freq = nn_data['nn_freq_data']
        data = np.concatenate((bl_data, agn_data), axis =0)
        label = np.concatenate((np.ones(bl_data.shape[0]),np.zeros(agn_data.shape[0])))
        return data, label
        
    def rescaleData(self,data):
        '''Return the N-D array of data with the flux of each source normalized in [0,1].
	
		Parameters
		----------
		data: array-like
	      	The complete dataset with shape (N_source,len(nn_freq_data),2).
        '''
        ptp=np.ptp(data[:,:,0])
        min = np.min(data[:,:,0])
        max = np.max(data[:,:,0])
        data[:,:,0]= (data[:,:,0]-min)/ptp
        return data
    
    def get_model_name(self,k):
		'''Return a string to save the trained model with the name corrispondent to the fold of K-fold.
	
		Parameters
		----------
		k: int
	   		The number of the fold of a specific run of k-fold.
		'''
        return 'model_'+str(k)+'.h5'

    def trainingModel(self,data,n_splits,save_dir):
		'''After training the model of the dnn, return a list with all the history value and save the model and the history of each run to file.
	
		Parameters
		----------
		data: array-like
	      	The complete rescaled dataset with shape (N_source,len(nn_freq_data),2).
	      
		n-splits: int
	          Number of splits for the k-fold routine.
		  
		save-dir: str
		 	 Path to the directory where to save the output of the dnn training.
		'''
        kf = KFold(n_splits=n_splits, shuffle = True, random_state = 1)
        save_dir = save_dir
        fold_var = 0
        history_list = []
        for train_idx, val_idx in kf.split(data_train):
			fold_var = fold_var+1
			print(train_idx.shape, val_idx.shape)
			# CREATE CALLBACKS
			checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
								monitor='val_accuracy', verbose=1, 
								save_best_only=True, mode='max')
			earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=4)
			callbacks_list = [checkpoint,earlystopping]
			history=model.fit(x=data_train[train_idx],y=l_train[train_idx],validation_data=(data_train[val_idx],l_train[val_idx]),epochs=20, callbacks=callbacks_list)
            np.save(os.path.join(save_dir,f'my_history_{fold_var}.npy'),history.history)
            history_list.append(history)
        return history_list

    def plotandsaveHistory(self,history_list):
        '''Function that plot and save the graph of Loss and Accuracy both for training and validation of each fold.
	
		Parameters
		----------
		history_list = list
		      	 List of history from the fit of the model of the dnn.
       	'''
        fold_var = 0
        for history in history_list:
            fold_var = fold_var+1
            fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
            fig.suptitle(f'Loss and Accuracy plot for both Training and Validation (fold_{i})')

            ax1.plot(history.history["val_loss"], label='validation')
            ax1.plot(history.history["loss"], label='training')
            ax1.legend(loc="upper right")
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
  
            ax2.plot(history.history["val_accuracy"], label='validation')
            ax2.plot(history.history["accuracy"], label='training')
            ax2.legend(loc="lower right")
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')

            fig.savefig(os.path.join(save_dir,f'plot_{i}'),bbox_inches='tight')
            fig.show()




