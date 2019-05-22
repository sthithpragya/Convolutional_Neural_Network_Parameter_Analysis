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

# imports for plotting og graphs for output
# from keras.utils import plot_model
from matplotlib import pyplot as plt
from mlp_layers import *
import numpy as np


# requirement 
# sudo apt-get install python3-tk (for matplotlib)
# sudo pip install matplotlib

def computeAndPlot(epochs,folderName):
	batch_size = 128
	num_classes = 10
	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes) 
	history1, score1 = mlpLayer1(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs)
	history2, score2 = mlpLayer2(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs)
	history3, score3 = mlpLayer3(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs)


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
	plt.legend(['Train1', 'Test1','Train2', 'Test2','Train3', 'Test3'],loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=6)
	fileNameAccuracy = folderName + 'MLP_upto3layers_accuracy_e'+str(epochs)+'.png'
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
	plt.legend(['Train1', 'Test1','Train2', 'Test2','Train3', 'Test3'], loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=6)
	fileNameLoss = folderName +  'MLP_upto3layers_loss_e'+str(epochs)+'.png'
	plt.savefig(fileNameLoss)

	names = ['no of layers = 1', 'no of layers = 2', 'no of layers = 3']
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

	fileNameTestScores = folderName + 'MLP_upto3layers_testScores_e'+str(epochs)+'.png'
	plt.savefig(fileNameTestScores)

#epochs = 20
epochs_list = [5,10,15,20]
folderName = "MLP_upto3layers/"

for epochs in epochs_list:
	computeAndPlot(epochs,folderName)

