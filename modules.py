from collections import Counter

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import time
import pickle

import data_load
reviews = data_load.load_data()

def tokenize():
    # Get a balanced sample of positive and negative reviews
    texts = [review['text'] for review in reviews]

    # Convert our 5 classes into 2 (negative or positive)
    binstars = [0 if review['stars'] <= 3 else 1 for review in reviews]
    balanced_texts = []
    balanced_labels = []
    limit = 200000  # Change this to grow/shrink the dataset
    neg_pos_counts = [0, 0]
    for i in range(len(texts)):
        polarity = binstars[i]
        if neg_pos_counts[polarity] < limit:
            balanced_texts.append(texts[i])
            balanced_labels.append(binstars[i])
            neg_pos_counts[polarity] += 1

    Counter(balanced_labels)
    # >>> Counter({0: 100000, 1: 100000})

    # tokenizer = Tokenizer(num_words=5)
    # toytexts = ["Is is a common word", "So is the", "the is common", "discombobulation is not common"]
    # tokenizer.fit_on_texts(toytexts)
    # sequences = tokenizer.texts_to_sequences(toytexts)


    # >>> [[1, 1, 4, 2], [1, 3], [3, 1, 2], [1, 2]]

    # print(sequences)
    # >>> [[1, 1, 4, 2], [1, 3], [3, 1, 2], [1, 2]]

    # print(tokenizer.word_index)

    # padded_sequences = pad_sequences(sequences)

    # print(padded_sequences)

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(balanced_texts)
    sequences = tokenizer.texts_to_sequences(balanced_texts)
    data = pad_sequences(sequences, maxlen=300)
    
    with open("keras_tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)

    return balanced_labels, tokenizer, data

#print(tokenize())

if __name__ == '__main__':
    start_time = time.time()
    print("Tokensize start", flush=True)
    tokenize()
    print("Tokenize done.", flush=True)
