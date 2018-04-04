import numpy as np
from modules import tokenize
from networks import prototype
import time

balanced_labels, tokenizer, data = tokenize() 


if __name__ == '__main__':
    start_time = time.time()
    model = prototype()
    print("prototyep network loaded", flush=True)
    model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=3)
    print("Done.", flush=True)
