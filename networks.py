from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential

import numpy as np

import modules

import pickle

balanced_labels, tokenizer, data = modules.tokenize() 

def prototype():
    model = Sequential()
    model.add(Embedding(200000, 128, input_length=300))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    with open("keras_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)
    model.save("yelp_sentiment_model.hdf5")
    
    return model

#print(prototype())
