'''MNIST is short for Modified National Institute of Standards and Technology database. MNIST contains a collection of 70,000, 28 x 28 images of handwritten digits from 0 to 9.
The dataset is already divided into training and testing sets.

Classify MNIST digits with different Neural Network Architectures
1. Load data (Available at keras.datasets)
2. Preprocess data
3. Define and Train model
4. Evaluate model
5. Augment data and test model again.
6. Use More layers (i.e. increase depth of Network) and Evaluate
model again.
7. Use more Neuron (i.e. increase width of Network) in each layer
and Evaluate model again.

'''

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten


#load data
(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

X_test.shape

y_train

import matplotlib.pyplot as plt
plt.imshow(X_train[2])


#preprocess data
X_train = X_train/255
X_test = X_test/255
X_train[0]


#define the model
model = Sequential()  #a Sequential model is a linear stack of layers used to build a neural network and  used for building and training various types of neural networks

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))  #Dense (fully connected) layers: These are common in feedforward neural networks.
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

#model train
history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)  #epochs specifies the number of times the model will iterate over the entire training dataset during training.  0.2, which means 20% of the training data will be reserved for validation.


#model evaluate

y_prob = model.predict(X_test)   #it predict probability of each number presnce
y_pred = y_prob.argmax(axis=1)    #max prob of a digit 

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#Augment Data and Test Model Again--Data augmentation can improve model generalization
#1-Use More Layers (Increase Depth of Network) and Evaluate:
#2-Use More Neurons (Increase Width of Network) and Evaluate


#visulization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])



'''
NOTES--
RELU-f(x)=max(0,x)     In other words, it returns the input if it's positive and zero otherwise , can led to dying relu
SOFTMAX-- The Softmax activation function is often used in the output layer of a neural network, particularly for multi-class classification problems.
It transforms the raw output scores (logits) of the network into a probability distribution over multiple classes.

#model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
loss=sparse_categorical_crossentropy-- used in multiclassification
ADAM-that combines the advantages of two other optimization methods, AdaGrad and RMSprop'''

