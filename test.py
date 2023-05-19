import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer 

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


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
    
print(intent)  

print(intents)

lemmatizer = WordNetLemmatizer()

# intents = json.loads(open("intent.json").read())
# print(intents)
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ",", ".", "'"]

# def preprocess_data(intents):
#     words = []
#     classes = []
#     documents = []
#     for intent in intents:
#         for pattern in intent["patterns"]:
#             word_list = nltk.word_tokenize(pattern)
#             words.extend(word_list)
#             documents.append((word_list, intent["tag"]))
#             if intent["tag"] not in classes:
#                 classes.append(intent["tag"])
#     words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
#     words = sorted(set(words))
#     classes = sorted(set(classes))
#     pickle.dump(words, open("words.pkl", "wb"))
#     pickle.dump(classes, open("classes.pkl", "wb"))
#     training = []
#     output_empty = [0] * len(classes)
#     for document in documents:
#         bag =[]
#         word_patterns = document[0]
#         word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#         for word in words:
#             bag.append(1) if word in word_patterns else bag.append(0)
#         output_row = list(output_empty)    
#         output_row[classes.index(document[1])]  = 1
#         training.append([bag,output_row])
#     random.shuffle(training)
#     training = np.array(training)
#     train_x = list(training[:,0])
#     train_y = list(training[:,1])
#     return train_x, train_y, classes

# def train_model(train_x, train_y, classes):
#     model = Sequential()
#     model.add(Dense(128, input_shape= (len(train_x[0]),), activation= "relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(64,activation = "relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(len(train_y[0]),activation = "softmax"))
#     sgd  = tensorflow.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
#     hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size =5, verbose=1)
#     model.save("chatbotmodel.h5", hist)
#     print("done")




for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words)


classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training =[]
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

model = Sequential()
model.add(Dense(128, input_shape= (len(train_x[0]),), activation= "relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation = "softmax"))

sgd  = tensorflow.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size =5, verbose=1)
model.save("chatbotmodel.h5", hist)
print("done")

# def retrain_model(event, context):
#     intents=[]
#     docs = db.collection('intents').get()
#     for doc in docs:
#         data = doc.to_dict()
#         intents.append(data)
#     train_x, train_y, classes = preprocess_data(intents)
#     train_model(train_x, train_y, classes)