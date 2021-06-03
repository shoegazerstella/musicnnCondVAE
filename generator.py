import os
import numpy as np
import operator
from tensorflow import keras

from musicnn_tags import musicnn_tags

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, le, min_max_scaler, batch_size=10, shuffle=True):
        'Initialization'
        self.df = df
        self.list_IDs = list(self.df.index)
        self.le = le
        self.min_max_scaler = min_max_scaler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return [X, y]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        
        # Initialization
        df_batch = self.df.iloc[list_IDs_temp]

        X = []
        y = []

        # Generate data
        for index, row in df_batch.iterrows():
            
            spec_file = row['spectrograms']
            tags_file = row['tags']

            if os.path.exists(spec_file) and os.path.exists(tags_file):

                arr = np.load(spec_file)
                v = np.load(tags_file)

                # take top N tags only
                d = dict(zip(musicnn_tags, list(v)))
                D = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True))
                tags = list(D.keys())[:1]

                if arr.shape == (96, 188):
                    X.append(arr)
                    y.append(tags[0])

        X = np.array(X)

        X = self.min_max_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        X = X.reshape(X.shape[0], -1).astype('float32')
        
        y_t = self.le.transform(y)

        y = np.array(y_t)

        return X, y