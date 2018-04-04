from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential

import numpy as np

from modules import tokenize

import time

balanced_labels, tokenizer, data = tokenize() 

def prototype():
    model = Sequential()
    model.add(Embedding(200000, 128, input_length=300))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save("yelp_sentiment_model.hdf5")
    
    return model

#print(prototype())

if __name__ == '__main__':
    start_time = time.time()
    print("Prototype make start", flush=True)
    prototype()
    print("Prototype make done", flush=True)
