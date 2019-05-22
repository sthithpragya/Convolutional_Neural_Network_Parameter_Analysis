'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# imports for plotting og graphs for output
# from keras.utils import plot_model
from matplotlib import pyplot as plt
from CNN_layers import *
import numpy as np


# requirement 
# sudo apt-get install python3-tk (for matplotlib)
# sudo pip install matplotlib

def computeAndPlot(epochs,folderName):
	batch_size = 128
	num_classes = 10

	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	history1, score1 = cnnOriginal(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs,input_shape)
	history2, score2 = cnn3ConvLayer32(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs,input_shape)
	history3, score3 = cnn3ConvLayer64(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs,input_shape)


	# code for plots 

	# Plot training & validation accuracy values
	fig = plt.figure()
	# accPlot = plt.subplot(111)
	plt.plot(history1.history['acc'])
	plt.plot(history1.history['val_acc'])
	plt.plot(history2.history['acc'])
	plt.plot(history2.history['val_acc'])
	plt.plot(history3.history['acc'])
	plt.plot(history3.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train1', 'Test1','Train32', 'Test32','Train64', 'Test64'],loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=6)
	fileNameAccuracy = folderName + 'MLP_Conv32andConv64_accuracy_e'+str(epochs)+'.png'
	plt.savefig(fileNameAccuracy)

	# Plot training & validation loss values
	fig = plt.figure()
	plt.plot(history1.history['loss'])
	plt.plot(history1.history['val_loss'])
	plt.plot(history2.history['loss'])
	plt.plot(history2.history['val_loss'])
	plt.plot(history3.history['loss'])
	plt.plot(history3.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train1', 'Test1','Train32', 'Test32','Train64', 'Test64'], loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=6)
	fileNameLoss = folderName +  'MLP_Conv32andConv64_loss_e'+str(epochs)+'.png'
	plt.savefig(fileNameLoss)

	names = ['CNN original', 'CNN with extra Conv32', 'CNN with extra Conv64']
	score_loss = [score1[0], score2[0], score3[0]]
	score_acc = [score1[1], score2[1], score3[1]]

	N = 3
	ind = np.arange(N)  # the x locations for the groups
	width = 0.27       # the width of the bars

	fig = plt.figure()
	ax = fig.add_subplot(111)

	rects1 = ax.bar(ind, score_loss, width, color='r')
	rects2 = ax.bar(ind+width, score_acc, width, color='g')

	ax.set_ylabel('Test Scores')
	ax.set_xticks(ind+width)
	ax.set_xticklabels( (names[0], names[1], names[2]) )
	ax.legend( (rects1[0], rects2[0]), ('loss', 'accuracy'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	# Add counts above the two bar graphs
	for rect in rects1 + rects2:
	    height = rect.get_height()
	    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % height, ha='center', va='bottom')

	fileNameTestScores = folderName + 'MLP_Conv32andConv64_testScores_e'+str(epochs)+'.png'
	plt.savefig(fileNameTestScores)

#epochs = 20
epochs_list = [5,10,15,20]
folderName = "MLP_Conv32andConv64/"

for epochs in epochs_list:
	computeAndPlot(epochs,folderName)

