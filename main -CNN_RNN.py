import glob
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
#import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Reshape
from keras.optimizers import SGD
data_dim = 60
timesteps = 41
nb_classes = 3
features1 =[]
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            #label =  fn.split('\\')[2].split('-')[3].split('.')[0]
            label = sub_dir
            for (start,end) in windows(sound_clip,window_size):
                #print(start)
                #print(end)
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                   
                    logspec = librosa.power_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                   
                    log_specgrams.append(logspec)
                    labels.append(label)
            
    
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels,dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = 'Dataset1'
sub_dirs= ['0','1','2']
features,labels = extract_features(parent_dir,sub_dirs)
labels = one_hot_encode(labels)


#p = np.random.choice(features.shape[0], features.shape[0], replace=False)
indices = np.random.permutation(features.shape[0])
features = features[indices,:,:]
labels = labels[indices,:]

x_train = features[:round(.8*features.shape[0]),:,:,:]
y_train = labels[:round(.8*features.shape[0]),:]
x_val = features[round(.8*features.shape[0]):,:,:,:]
y_val = labels[round(.8*features.shape[0]):,:]

num_lstm_units =64
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(x_train.shape[1] ,x_train.shape[2] , x_train.shape[3] )))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Reshape((model.output_shape[1],
                   model.output_shape[2] * model.output_shape[3])))
model.add(Dropout(0.5))
model.add(LSTM(num_lstm_units, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(num_lstm_units))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
history=model.fit(x_train, y_train,
          batch_size=64, epochs=10,
          validation_data=(x_val, y_val))
model.summary()
print(history.history.keys())

plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()
