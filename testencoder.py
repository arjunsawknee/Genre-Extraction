import numpy as np
import tensorflow as tf
import pandas as pd
import math
from random import shuffle
import os
import sox
import librosa
from softmaxautoencoder import load_data, get_one_hot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
import itertools


input_dim = 20000
num_classes = 10
alpha = 0.00001
num_epochs = 400
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

def runPCA2D(sess, decoding, encoding, inputs_batch, labels_batch, X, Y):
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
	_, latents = sess.run([decoding, encoding], feed_dict={inputs_batch : X, labels_batch : Y})
	truelabels = tf.math.argmax(Y, axis=1).eval()

	pca = decomposition.PCA(n_components = 3)
	#latents = latents[0]
	pca.fit(latents)
	latents = pca.transform(latents)

	for name, label in [('Classical', 0), ('Jazz', 1), ('Metal', 2), ('Pop', 3)]:
		ax.text(
			latents[truelabels == label, 0].mean(),
			latents[truelabels == label, 1].mean(),
			name,
			horizontalalignment = 'center',
			bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

	ax.scatter(latents[:, 0], latents[:, 1], c=truelabels, cmap=plt.cm.nipy_spectral,
		   edgecolor='k')

	plt.show()

def runPCA3D(sess, decoding, encoding, inputs_batch, labels_batch, X, Y):
	allgenres = ['classical', 'jazz', 'metal', 'pop']

	_, latents = sess.run([decoding, encoding], feed_dict={inputs_batch : X, labels_batch : Y})
	truelabels = tf.math.argmax(Y, axis=1).eval()

	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	ax = plt.axes(projection = '3d')
	plt.cla()

	pca = decomposition.PCA(n_components = 3)
	#latents = latents[0]
	pca.fit(latents)
	latents = pca.transform(latents)

	for name, label in [('Classical', 0), ('Jazz', 1), ('Metal', 2), ('Pop', 3)]:
		ax.text(
			latents[truelabels == label, 0].mean(),
			latents[truelabels == label, 1].mean(),
			latents[truelabels == label, 2].mean(),
			name,
			horizontalalignment = 'center',
			bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

	ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=truelabels, cmap=plt.cm.nipy_spectral,
		   edgecolor='k')
	plt.show()


def test(X, Y):
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('./modelWeights/autoencoder/autoencoder-100.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./modelWeights/autoencoder'))
		graph = tf.get_default_graph()
		inputs_batch = tf.get_collection('inputs_placeholder')[0]
		labels_batch = tf.get_collection('labels_placeholder')[0]
		keep_prob = tf.get_collection('keep_prob')[0]
		encoding = tf.get_collection('encoding')[0]
		decoding = tf.get_collection('decoding')[0]
		loss = tf.get_collection('loss')[0]

		#runPCA2D(sess, decoding, encoding, inputs_batch, labels_batch, X, Y)
		runPCA3D(sess, decoding, encoding, inputs_batch, labels_batch, X, Y)


def main():
	#songs = pd.read_csv('songs_dev.csv')
	#labels = pd.read_csv('labels_dev.csv')
	songs, labels = load_data()
	test(songs, labels)

if __name__== "__main__":
	main()