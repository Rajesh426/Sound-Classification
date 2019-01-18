import glob
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
# Data_Dim is the dimensions of the each timestep
data_dim = 60
# timesteps is the total number of timesteps in the each RNN layer
timesteps = 41
nb_classes = 3
# features is the null matrix for storing the mfcc values
features1 =[]
# window function for sliding the window in a given speech signal
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size)
# Extract function for extracting the features from the speech signals and labesl for the given speech signal
def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            label =  sub_dir
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
            
    
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames)

    features = log_specgrams
    features1 = np.zeros((features.shape[0], features.shape[2], features.shape[1]))
    # Transposing featuers and saving the features1
    for i in range(len(features)):
        features1[i, :, :] = features[i, :, :].T
      
    
    return np.array(features1), np.array(labels,dtype = np.int)
# Generating the one hot vector for the labels
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
# Loading the speech data from the Dataset2 folder
parent_dir = 'Dataset2'
sub_dirs= ['0','1','2']
features,labels = extract_features(parent_dir,sub_dirs)
labels = one_hot_encode(labels)


# Shuffling the features and labels for better training
indices = np.random.permutation(features.shape[0])
features = features[indices,:,:]
labels = labels[indices,:]

# Dividing the data into training[Training data is around 80 %] and validation data{Validation data is 20 %]
x_train = features[:round(.8*features.shape[0]),:,:]
y_train = labels[:round(.8*features.shape[0]),:]
x_val = features[round(.8*features.shape[0]):,:,:]
y_val = labels[round(.8*features.shape[0]):,:]


# expected input data shape: (batch_size, timesteps, data_dim)
# Generating the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 64
model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 64
model.add(Dropout(0.5))
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.5))
model.add(LSTM(32))  # return a single vector of dimension 32
# Output layer
model.add(Dense(3, activation='softmax'))
# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
history=model.fit(x_train, y_train,
          batch_size=64, epochs=20,
          validation_data=(x_val, y_val))
# Model summary for number of parameters use in the algorithm 
model.summary()
# Plotting the error and accuracy of the model 
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()
