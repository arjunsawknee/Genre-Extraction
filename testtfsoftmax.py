import numpy as np
import tensorflow as tf
import pandas as pd
import math
from random import shuffle
import os
import sox
import librosa
from tfsoftmax import load_data, get_one_hot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
import itertools


input_dim = 20000
num_classes = 10
alpha = 0.0001
num_epochs = 200
batch_size = 512
classificationweight = 0.0001

# Borrowed from scikit-learn implementation of plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	#classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	classes = ['classical', 'jazz', 'metal', 'pop']

	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def runPCA2D(sess, inputs_batch, labels_batch, X, Y, y_hat):
	# set up
	allgenres = ['classical', 'jazz', 'metal', 'pop']
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = plt.axes()
	plt.cla()

	# run PCA on raw data
	'''truelabels = tf.math.argmax(Y, axis=1).eval()
	pca = decomposition.PCA(n_components = 3)
	pca.fit(X)
	latents = pca.transform(X)'''

	# uncomment out for labels of clusters (may overlap)
	'''for name, label in [('Classical', 0), ('Jazz', 1), ('Metal', 2), ('Pop', 3)]:
		ax.text(
			latents[truelabels == label, 0].mean(),
			latents[truelabels == label, 1].mean(),
			name,
			horizontalalignment = 'center',
			bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
	ax.scatter(latents[:, 0], latents[:, 1], c=truelabels, cmap=plt.cm.nipy_spectral,
		   edgecolor='k')
	plt.show()'''

	# run PCA on encoder results
	logits = sess.run([y_hat], feed_dict={inputs_batch : X, labels_batch : Y})
	truelabels = tf.math.argmax(Y, axis=1).eval()

	pca = decomposition.PCA(n_components = 2)
	logits = logits[0]
	pca.fit(logits)
	logits = pca.transform(logits)

	colors = [('purple', 'Pop', 3), ('darkorange', 'Metal', 2), ('green', 'Jazz', 1), ('navy', 'Classical', 0)]
	for color, name, label in colors:
		plt.scatter(logits[truelabels == label, 0], logits[truelabels == label, 1], color=color, alpha=.5,
                label=name, edgecolor = 'k')
	plt.title('PCA of 4-Dim Softmax Regression Logits')
	plt.legend(loc='upper right', shadow=False, scatterpoints=1)
	#for name, label in [('Classical', 0), ('Jazz', 1), ('Metal', 2), ('Pop', 3)]:
	#	ax.text(
	#		latents[truelabels == label, 0].mean(),
	#		latents[truelabels == label, 1].mean(),
	#		name,
	#		horizontalalignment = 'center',
	#		bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

	#ax.scatter(latents[:, 0], latents[:, 1], c=truelabels, cmap=plt.cm.nipy_spectral,
	#	   edgecolor='k')

	plt.show()

def runPCA3D(sess, inputs_batch, labels_batch, X, Y, y_hat):
	allgenres = ['classical', 'jazz', 'metal', 'pop']

	logits = sess.run([y_hat], feed_dict={inputs_batch : X, labels_batch : Y})
	truelabels = tf.math.argmax(Y, axis=1).eval()

	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = plt.axes(projection = '3d')
	plt.cla()

	pca = decomposition.PCA(n_components = 3)
	logits = logits[0]
	pca.fit(logits)
	logits = pca.transform(logits)

	#colors = [('red', 'Classical', 0), ('turquoise', 'Jazz', 1), ('darkorange', 'Metal', 2), ('magenta', 'Pop', 3)]
	#colors = reversed(colors)
	#for color, name, label in colors:
	#	plt.scatter(latents[truelabels == label, 0], latents[truelabels == label, 1], 
	#		latents[truelabels == label, 2], color=color, alpha=.5, label=name, edgecolor = 'k')
	#plt.title('PCA of 128-Dim Softmax Autoencoder Bottleneck')
	#plt.legend(loc='best', shadow=False)
	for name, label in [('Classical', 0), ('Jazz', 1), ('Metal', 2), ('Pop', 3)]:
		ax.text(
			logits[truelabels == label, 0].mean(),
			logits[truelabels == label, 1].mean(),
			logits[truelabels == label, 2].mean(),
			name,
			horizontalalignment = 'center',
			bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

	ax.scatter(logits[:, 0], logits[:, 1], logits[:, 2], c=truelabels, cmap=plt.cm.nipy_spectral,
		   edgecolor='k')

	plt.show()


def test(X, Y):
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('./modelWeights/tfsoftmax/tfsoftmax-200.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./modelWeights/tfsoftmax'))
		graph = tf.get_default_graph()
		inputs_batch = tf.get_collection('inputs_placeholder')[0]
		labels_batch = tf.get_collection('labels_placeholder')[0]
		#encoding = tf.get_collection('encoding')[0]
		#decoding = tf.get_collection('decoding')[0]
		y_hat = tf.get_collection('y_hat')[0]
		loss = tf.get_collection('loss')[0]

		logits = sess.run([y_hat], feed_dict={inputs_batch : X, labels_batch : Y})
		logits = np.asarray(logits[0])
		predictions = tf.math.argmax(tf.nn.softmax(logits), axis=1)
		truelabels = tf.math.argmax(Y, axis=1)
		numequal = tf.math.equal(predictions, truelabels)
		numcorrect = tf.math.count_nonzero(numequal)
		devaccuracy = numcorrect.eval() / float(X.shape[0])
		print("Dev Accuracy: " + str(devaccuracy))

		#Computes confusion matrix
		#class_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]
		#class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		class_names = [0, 1, 2, 3]
		pred = predictions.eval()
		correct_labels = truelabels.eval()
		confmat = confusion_matrix(correct_labels, pred, class_names)
		np.set_printoptions(precision=2)
		#plt.figure()
		#plot_confusion_matrix(confmat, classes=class_names, normalize=False, title='Confusion matrix, without normalization')
		#plt.figure()
		#plot_confusion_matrix(confmat, classes=class_names, normalize=True, title='Normalized confusion matrix')
		# Commented out to prevent showing the matrix everytime code is run
		#plt.show()


		runPCA2D(sess, inputs_batch, labels_batch, X, Y, y_hat)
		#runPCA3D(sess, inputs_batch, labels_batch, X, Y, y_hat)


def main():
	#songs = pd.read_csv('songs_dev.csv')
	#labels = pd.read_csv('labels_dev.csv')
	songs, labels = load_data()
	test(songs, labels)

if __name__== "__main__":
	main()