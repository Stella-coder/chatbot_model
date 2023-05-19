import random 
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask,request
from flask_cors import CORS,cross_origin

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD



app = Flask('__name__')
CORS(app, resources={r"/user/*": {"origins": "*"}})
app.config['CORS_HEADERS']="Content-Type"




cred = credentials.Certificate('adminSDK.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred)
db = firestore.client()

intent=[]
intents={"intents":intent}

docs = db.collection('intents').get()
for doc in docs:
    data = doc.to_dict()
    intent.append(data)
    
# print(intent)  

# print(intents)

lemmatizer = WordNetLemmatizer()
ignore_letters = ["?", "!", ",", ".", "'"]

def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    for intent in intents:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))
    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))
    training = []
    output_empty = [0] * len(classes)
    for document in documents:
        bag =[]
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        output_row = list(output_empty)    
        output_row[classes.index(document[1])]  = 1
        training.append([bag,output_row])
    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x, train_y, classes

def train_model(train_x, train_y, classes):
    model = Sequential()
    model.add(Dense(128, input_shape= (len(train_x[0]),), activation= "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]),activation = "softmax"))
    sgd  = tensorflow.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size =5, verbose=1)
    model.save("chatbotmodel.h5", hist)
    print("done")



def retrain_model(snapshot, changes, read_time):
    intents=[]
    docs = db.collection('intents').get()
    for doc in docs:
        data = doc.to_dict()
        intents.append(data)
    train_x, train_y, classes = preprocess_data(intents)
    train_model(train_x, train_y, classes)

# Create a reference to the intents collection
intents_ref = db.collection('intents')

# Watch for document changes in the collection
query_watch = intents_ref.on_snapshot(lambda snapshot, changes, read_time: retrain_model(snapshot, changes, read_time))

 

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort( key = lambda x: x[1], reverse = True)
    return_list=[]
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability":str(r[1])})
    return return_list

def get_responses(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result
    
print("Go, your bot is running")

@app.route("/user", methods=["POST"])
@cross_origin()
def user():
    jsony = request.json
    data = jsony["msg"]
    ints = predict_class(data)
    return str(get_responses(ints,intents))
    
app.run()   

# while True:
#     message = input("")
#     ints = predict_class(message)
#     res= get_responses(ints,intents)
#     print(res)
#     if message == "Bye":
#         break