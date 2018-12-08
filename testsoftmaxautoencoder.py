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
from sklearn.metrics import confusion_matrix
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
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
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

def test(X, Y):
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('./modelWeights/softmaxautoencoder-120.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./modelWeights'))
		graph = tf.get_default_graph()
		inputs_batch = tf.get_collection('inputs_placeholder')[0]
		labels_batch = tf.get_collection('labels_placeholder')[0]
		encoding = tf.get_collection('encoding')[0]
		decoding = tf.get_collection('decoding')[0]
		y_hat = tf.get_collection('y_hat')[0]
		loss = tf.get_collection('loss')[0]
		_, preds = sess.run([encoding, y_hat], feed_dict={inputs_batch : X, labels_batch : Y})
		predictions = tf.math.argmax(preds, axis=1)
		truelabels = tf.math.argmax(Y, axis=1)
		numequal = tf.math.equal(predictions, truelabels)
		numcorrect = tf.math.count_nonzero(numequal)
		devaccuracy = numcorrect.eval() / float(X.shape[0])
		print("Dev Accuracy: " + str(devaccuracy))

		#Computes confusion matrix
		#class_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]
		class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		pred = predictions.eval()
		correct_labels = truelabels.eval()
		confmat = confusion_matrix(correct_labels, pred, class_names)
		np.set_printoptions(precision=2)
		plt.figure()
		plot_confusion_matrix(confmat, classes=class_names, normalize=False, title='Confusion matrix, without normalization')
		plt.figure()
		plot_confusion_matrix(confmat, classes=class_names, normalize=True, title='Normalized confusion matrix')
		# Commented out to prevent showing the matrix everytime code is run
		plt.show()

def main():
	songs = pd.read_csv('songs_dev.csv')
	labels = pd.read_csv('labels_dev.csv')
	test(songs, labels)

if __name__== "__main__":
	main()