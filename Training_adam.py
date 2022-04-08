# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:25:22 2022

@author: Anjali
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:56:11 2022

@author: ankita
"""

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import Adam - gives error
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data040422.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
print("words: ",words)
print("classes: ",classes)
print("docs:",documents)



# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
print(training)
# =============================================================================
# train_x
# train_y
# training
# bag
# doc[2]
# =============================================================================
# Create modelA - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
modelA = Sequential()
modelA.add(Dense(120, input_shape=(len(train_x[0]),), activation='relu'))
modelA.add(Dropout(0.02))
modelA.add(Dense(64, activation='relu'))
modelA.add(Dropout(0.02))
modelA.add(Dense(len(train_y[0]), activation='softmax'))


# Compile modelA. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this modelA
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adm = Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")
#modelA.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
modelA.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
#fitting and saving the modelA 
hist = modelA.fit(np.array(train_x), np.array(train_y), epochs=800, batch_size=5, verbose=1, validation_split=0.3)
modelA.summary()
modelA.save('modelA.h5', hist)


print("modelA created")
'''
#plt.plot(loss_history,label='Training Loss')  
#plt.plot(val_loss_history,label='Validation Loss')  
plt.legend()  
plt.show()  
plt.plot(accuracy,label='Training accuracy')  
plt.plot(val_accuracy,label='Validation accuracy')  
plt.legend()  
plt.show()  

'''







