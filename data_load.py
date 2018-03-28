from datetime import datetime

import json

def load_data():

    # Load the reviews and parse JSON
    t1 = datetime.now()
    with open("/root/yelp_review/dataset/review.json") as f:
        reviews = f.readlines()
    reviews = [json.loads(review, encoding='utf-8') for review in reviews]
    #print(datetime.now() - t1)
    return reviews

#print(load_data())
