import numpy as np
import modules
import networks

balanced_labels, tokenizer, data = modules.tokenize() 

model = networks.prototype()
model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=3)

