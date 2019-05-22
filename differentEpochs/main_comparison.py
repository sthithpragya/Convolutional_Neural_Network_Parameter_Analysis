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
from mlp import *
from CNN import *
import numpy as np


# requirement 
# sudo apt-get install python3-tk (for matplotlib)
# sudo pip install matplotlib

def computeAndPlotFCN(epochs,folderName):
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
	return mlpLayer1(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs)

def computeAndPlotCNN(epochs,folderName):
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

	return cnnOriginal(x_train,y_train,x_test,y_test,batch_size,num_classes,epochs,input_shape)

#epochs = 20
epochs_list = [5,10,15,20]
folderName = "difference/"

fcn_acc = []
fcn_val_acc = []
fcn_score_loss = []
fcn_score_acc  = []

cnn_acc = []
cnn_val_acc = []
cnn_score_loss = []
cnn_score_acc  = []


for epochs in epochs_list:
	history1, score1 = computeAndPlotFCN(epochs,folderName)
	history2, score2 = computeAndPlotFCN(epochs,folderName)

	fcn_acc = fcn_acc + [history1.history['acc'][-1]]
	fcn_val_acc = fcn_val_acc + [history1.history['val_acc'][-1]]
	fcn_score_loss = fcn_score_loss + [score1[0]]
	fcn_score_acc  = fcn_score_acc + [score1[1]]

	cnn_acc = cnn_acc + [history2.history['acc'][-1]]
	cnn_val_acc = cnn_val_acc + [history2.history['val_acc'][-1]]
	cnn_score_loss = cnn_score_loss + [score2[0]]
	cnn_score_acc  = cnn_score_acc + [score2[1]]

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.33       # the width of the bars

#########################################################3
fig = plt.figure()
ax = fig.add_subplot(111)

rects1_ac = ax.bar(ind, fcn_acc, width, color='r')
rects2_ac = ax.bar(ind+width, cnn_acc, width, color='g')


# ax.set_ylabel('Test Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( (str(epochs_list[0]), str(epochs_list[1]), str(epochs_list[2]),str(epochs_list[3])) )
ax.legend( (rects1_ac[0], rects2_ac[0]), ('train_accuracy_mlp', 'train_accuracy_cnn'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Add counts above the two bar graphs
for rect in rects1_ac + rects2_ac:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % height, ha='center', va='bottom')

fileNameTestScores = folderName + 'train_accuracy_difference'+'.png'
plt.savefig(fileNameTestScores)

############################################################
fig = plt.figure()
bx = fig.add_subplot(111)

rects1_vac = bx.bar(ind, fcn_val_acc, width, color='r')
rects2_vac = bx.bar(ind+width, cnn_val_acc, width, color='g')


# ax.set_ylabel('Test Scores')
bx.set_xticks(ind+width)
bx.set_xticklabels( (str(epochs_list[0]), str(epochs_list[1]), str(epochs_list[2]),str(epochs_list[3])) )
bx.legend( (rects1_vac[0], rects2_vac[0]), ('validation_accuracy_mlp', 'validation_accuracy_cnn'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Add counts above the two bar graphs
for rect in rects1_vac + rects2_vac:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % height, ha='center', va='bottom')

fileNameTestScores = folderName + 'validation_accuracy_difference'+'.png'
plt.savefig(fileNameTestScores)


############################################################
fig = plt.figure()
cx = fig.add_subplot(111)

rects1_sl = cx.bar(ind, fcn_score_loss, width, color='r')
rects2_sl = cx.bar(ind+width, cnn_score_loss, width, color='g')


# ax.set_ylabel('Test Scores')
cx.set_xticks(ind+width)
cx.set_xticklabels( (str(epochs_list[0]), str(epochs_list[1]), str(epochs_list[2]),str(epochs_list[3])) )
cx.legend( (rects1_sl[0], rects2_sl[0]), ('score_loss_mlp', 'score_loss_cnn'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Add counts above the two bar graphs
for rect in rects1_sl + rects2_sl:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % height, ha='center', va='bottom')

fileNameTestScores = folderName + 'score_loss_difference'+'.png'
plt.savefig(fileNameTestScores)

############################################################
fig = plt.figure()
dx = fig.add_subplot(111)

rects1_sa = dx.bar(ind, fcn_score_acc, width, color='r')
rects2_sa = dx.bar(ind+width, cnn_score_acc , width, color='g')


# ax.set_ylabel('Test Scores')
dx.set_xticks(ind+width)
dx.set_xticklabels( (str(epochs_list[0]), str(epochs_list[1]), str(epochs_list[2]),str(epochs_list[3])) )
dx.legend( (rects1_sa[0], rects2_sa[0]), ('score_acc _mlp', 'score_acc _cnn'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Add counts above the two bar graphs
for rect in rects1_sa + rects2_sa:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % height, ha='center', va='bottom')

fileNameTestScores = folderName + 'score_acc _difference'+'.png'
plt.savefig(fileNameTestScores)

























# N = 4
# ind = np.arange(N)  # the x locations for the groups
# width = 0.2       # the width of the bars

# fig = plt.figure()
# ax = fig.add_subplot(111)

# rects1 = ax.bar(ind, fcn_acc, width, color='r')
# rects2 = ax.bar(ind+width, fcn_val_acc, width, color='g')
# rects3 = ax.bar(ind+2*width, fcn_score_loss, width, color='b')
# rects4 = ax.bar(ind+3*width, fcn_score_acc, width, color='y')

# # ax.set_ylabel('Test Scores')
# ax.set_xticks(ind+width)
# ax.set_xticklabels( (str(epochs_list[0]), str(epochs_list[1]), str(epochs_list[2]),str(epochs_list[3])) )
# ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0]), ('train_accuracy', 'validation_accuracy','test_loss', 'test_accuracy'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
# # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# # Add counts above the two bar graphs
# for rect in rects1 + rects2 + rects3 + rects4:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % height, ha='center', va='bottom')

# fig = plt.figure()
# bx = fig.add_subplot(111)


# rects_c1 = bx.bar(ind, cnn_acc, width, color='r')
# rects_c2 = bx.bar(ind+width, cnn_val_acc, width, color='g')
# rects_c3 = bx.bar(ind+2*width, cnn_score_loss, width, color='b')
# rects_c4 = bx.bar(ind+3*width, cnn_score_acc, width, color='y')

# # ax.set_ylabel('Test Scores')
# bx.set_xticks(ind+width)
# bx.set_xticklabels( (str(epochs_list[0]), str(epochs_list[1]), str(epochs_list[2]),str(epochs_list[3])) )
# bx.legend( (rects_c1[0], rects_c2[0],rects_c3[0],rects_c4[0]), ('train_accuracy', 'validation_accuracy','test_loss', 'test_accuracy'),loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=2 )
# # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# # Add counts above the two bar graphs
# for rect in rects_c1 + rects_c2 + rects_c3 + rects_c4:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % height, ha='center', va='bottom')


# fileNameTestScores = folderName + 'difference'+'.png'
# plt.savefig(fileNameTestScores)

