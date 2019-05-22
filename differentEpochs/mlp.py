'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# for one hidden layer 
def mlpLayer1(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs):
	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(784,)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))		# only one hidden layer
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
	history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)

	return history,score

