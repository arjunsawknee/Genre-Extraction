import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sox
import librosa
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from random import shuffle

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



def load_data():
	"""
		Converts all files of the GTZAN dataset
		to the WAV (uncompressed) format.
	"""

	print('Reading data...')
	tfm = sox.Transformer()
	songs = np.zeros((10000, 40000))
	labels = np.zeros((10000, 1))
	counter = 0

	allgenres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	# Splits each song into 10 examples of shape (40000, 1) ~ 2 seconds each
	numsplit = 10
	sizesplit = 40000
	# generalized loop to process data
	for index in range(len(allgenres)):
		for filename in os.listdir('./genres/' + allgenres[index]):
			if filename.endswith(".wav"):
				audio, sr = librosa.core.load('./genres/' + allgenres[index] + '/' + filename)
				for j in range(numsplit):
					songs[counter] = audio[(sizesplit * j) : (sizesplit * (j + 1))]
					labels[counter] = index
					counter += 1
	songs = pd.DataFrame(songs)
	labels = pd.DataFrame(labels)
	print('Data reading done :)')
	return songs, labels


def get_one_hot(label_num, num_classes = 10):
    one_hot = np.zeros((num_classes,1))
    one_hot[int(label_num),0] = 1
    return one_hot

def initialize_parameters(vector_dim = 40000, num_classes = 10):
	w = np.random.randn(num_classes, vector_dim)
	b = np.zeros((num_classes, 1))
	return w, b

def softmax(logits):
	e_x = np.exp(logits - np.max(logits))
	return e_x / e_x.sum()

def forward_propagate(inputs, weights, bias):
	score = np.matmul(weights, inputs) + bias
	y_hat = softmax(score)
	pred = np.argmax(y_hat)
	return y_hat, pred

def get_loss(y_hat, y):
	return -1*(np.sum(y*np.log(y_hat)))

def back_propagate(weights, bias, y_hat, y, inputs, learning_rate = 0.001):
	db = y_hat-y
	dw = db.dot(inputs.T)
	new_weights = weights - learning_rate*dw
	new_bias = bias - learning_rate*db
	return new_weights, new_bias


def train(songs, labels, songs_dev, labels_dev):
	w, b = initialize_parameters()
	for i in range(100):
		# Shuffling training set
		ind_list=[i for i in range(songs.shape[0])]
		shuffle(ind_list)
		songs = songs.iloc[ind_list]
		labels = labels.iloc[ind_list]

		smoothed_cost_list = []
		correct_class = 0
		attempts = 0
		for j, row in songs.iterrows():
			label = labels.loc[j]
			label = int(label[0])
			#print(type(row.values))
			inputs = row.values
			inputs = inputs.reshape((40000, 1))
			y_hat, pred = forward_propagate(inputs, w, b)
			if pred == label:
				correct_class += 1
			attempts += 1
			label = get_one_hot(label, 10)
			curr_loss = get_loss(y_hat, label)
			smoothed_cost_list.append(curr_loss)
			w, b = back_propagate(w, b, y_hat, label, inputs)
		smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
		print("Epoch " + str(i) + ": Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost))
		if (i % 5 == 0 and i):
			test(w, b, songs_dev, labels_dev)
	return w, b

def test(w, b, songs, labels):
	smoothed_cost_list = []
	preds = []
	correct_class = 0
	attempts = 0
	for j, row in songs.iterrows():
		label = labels.loc[j]
		label = int(label[0])
		#print(type(row.values))
		inputs = row.values
		inputs = inputs.reshape((40000, 1))
		y_hat, pred = forward_propagate(inputs, w, b)
		preds += pred
		if pred == label:
			correct_class += 1
		attempts += 1
		label = get_one_hot(label, 10)
		curr_loss = get_loss(y_hat, label)
		smoothed_cost_list.append(curr_loss)
	smoothed_cost = float(sum(smoothed_cost_list))/len(smoothed_cost_list)
	print("Test Accuracy = " + str(float(correct_class)*100/attempts) + ", Smoothed Cost : "  + str(smoothed_cost))

	class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	confmat = confusion_matrix(labels, preds, class_names)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(confmat, classes=class_names, normalize=False, title='Confusion matrix, without normalization')
	plt.figure()
	plot_confusion_matrix(confmat, classes=class_names, normalize=True, title='Normalized confusion matrix')
	plt.show()

def main():
	songs, labels = load_data()

	ind_list=[i for i in range(songs.shape[0])]
	shuffle(ind_list)
	songs = songs.iloc[ind_list]
	labels = labels.iloc[ind_list]
	songs_train = songs.iloc[0:8000]
	songs_dev = songs.iloc[8000:]
	labels_train = labels.iloc[0:8000]
	labels_dev = labels.iloc[8000:]
	w, b = train(songs_train, labels_train, songs_dev, labels_dev)
	test(w, b, songs_dev, labels_dev)

  
if __name__== "__main__":
	main()
