from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# load the tokenizer and the model
with open("keras_tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("yelp_sentiment_model.hdf5")

# replace with the data you want to classify
newtexts = ["Clean, nice and professional service. Friendly team. Also my order was very delicious. Strongly recommend you this place", "Very organized and efficient. And people working behind the counter are actually friendly and don't seem like they hate their jobs and hate you. Food did take a few minutes to come out since they were so busy. Overall, a good experience","Only came in here to use the bathroom because the 9/11 Memorial doesn't have Any! Separate lines for men & women - each only has 1 toilet/1 sink!","fuck this shitty place. workers dont give a damn fuck about this place. slow as fuck. esp the main indian women working takes her slow ass time when people are wait ing like wow service is so damn bad"]

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(newtexts)
data = pad_sequences(sequences, maxlen=300)

# get predictions for each of your new texts
predictions = model.predict(data)
print(predictions)
