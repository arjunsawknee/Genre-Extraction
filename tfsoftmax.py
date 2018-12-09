import numpy as np
import tensorflow as tf
import pandas as pd
import math
from random import shuffle
import os
import sox
import librosa

#input_dim = 40000
input_dim = 500
num_classes = 4
alpha = 0.0001
num_epochs = 200
batch_size = 512

def get_one_hot(label_num, num_classes = 4):
    one_hot = np.zeros((1,num_classes))
    one_hot[0, int(label_num)] = 1
    return one_hot

def load_data():
	"""
		Converts all files of the GTZAN dataset
		to the WAV (uncompressed) format.
	"""

	print('Reading data...')
	tfm = sox.Transformer()
	songs = np.zeros((8000, input_dim))
	onehotlabels = np.zeros((8000, num_classes))
	counter = 0

	allgenres = ['classical', 'jazz', 'metal', 'pop']

	#allgenres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	# Splits each song into 10 examples of shape (40000, 1) ~ 2 seconds each
	'''numsplit = 10
	sizesplit = 40000'''

	# splits of 1 second each:
	numsplit = 20
	sizesplit = input_dim
	# generalized loop to process data
	for index in range(len(allgenres)):
		for filename in os.listdir('./genres/' + allgenres[index]):
			if filename.endswith(".wav"):
				audio, sr = librosa.core.load('./genres/' + allgenres[index] + '/' + filename)

				audio = audio[:600000]
				audio = audio.reshape(15000, 40)
				audio = np.mean(audio, axis=1)

				for j in range(numsplit):
					songs[counter] = audio[(sizesplit * j) : (sizesplit * (j + 1))]
					onehotlabels[counter] = get_one_hot(index)
					counter += 1
	songs = pd.DataFrame(songs)
	onehotlabels = pd.DataFrame(onehotlabels)
	print('Data reading done :)')
	return songs, onehotlabels

def get_placeholders():
	inputs_placeholder = tf.placeholder(tf.float32, (None, input_dim))
	labels_placeholder = tf.placeholder(tf.float32, (None, num_classes))
	tf.add_to_collection('inputs_placeholder', inputs_placeholder)
	tf.add_to_collection('labels_placeholder', labels_placeholder)
	return inputs_placeholder, labels_placeholder

def add_parameters():
	weights = {}

	# Softmax classifier weights
	weights["W1_softmax"] = tf.get_variable(name="W1_softmax", shape = (input_dim, 128), initializer = tf.contrib.layers.xavier_initializer())
	weights["b1_softmax"] = tf.get_variable(name="b1_softmax", initializer = tf.zeros((1,128)))
	weights["W2_softmax"] = tf.get_variable(name="W2_softmax", shape = (128, num_classes), initializer = tf.contrib.layers.xavier_initializer())
	weights["b2_softmax"] = tf.get_variable(name="b2_softmax", initializer = tf.zeros((1,num_classes)))
	return weights

# Convention of h for hidden layers of classifier
def softmaxclassifier(inputs_batch, weights):
	h_1  = tf.nn.tanh(tf.add(tf.matmul(inputs_batch, weights["W1_softmax"]), weights["b1_softmax"]))
	h_2 = tf.add(tf.matmul(h_1, weights["W2_softmax"]), weights["b2_softmax"])
	return h_2

def get_batches(seq, size=batch_size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def train(X, Y, X_dev, Y_dev):
	tf.reset_default_graph()
	inputs_batch, labels_batch = get_placeholders()
	weights = add_parameters()

	y_hat = softmaxclassifier(inputs_batch, weights)
	tf.add_to_collection("y_hat", y_hat)

	# Check shape of labels: need to be shape (batch_size, num_classes) according to documentation
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels_batch), logits=y_hat))
	tf.add_to_collection("loss", loss)
	# Just reconstruction loss: order of magnitude 0.005-0.01
	#loss = tf.reduce_mean(tf.pow(decoding - inputs_batch, 2))
	# Just cross entropy loss: order of magnitude 1.5 - 2.5
	#loss = (classificationweight * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_batch, logits=y_hat))
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=5)
	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter('./tensorboardlogs/tfsoftmax', sess.graph)
		sess.run(init)
		# Shuffling training set
		ind_list=[i for i in range(X.shape[0])]
		shuffle(ind_list)
		X = X.iloc[ind_list]
		Y = Y.iloc[ind_list]

		train_accuracies = []
		dev_accuracies = []
		loss_per_epoch = []
		for iteration in range(num_epochs):
			inputs_batches = get_batches(X)
			labels_batches = get_batches(Y)

			cost_list = []
			currnumcorrect = 0
			for i in range(len(inputs_batches)):
				batch = inputs_batches[i]

				#should be (batch_size, num_classes)
				batchlabel = labels_batches[i]
				logits, _, curr_loss = sess.run([y_hat, optimizer, loss], feed_dict={inputs_batch: batch, labels_batch: batchlabel})
				#logits = sess.run([y_hat], feed_dict={inputs_batch: batch, labels_batch: batchlabel})
				#print(np.asarray(logits[0]).shape)
				#print(type(logits[0]))

				#Check if predictions are correct and increment counter/accuracy
				predictions = tf.math.argmax(tf.nn.softmax(logits), axis=1)
				truelabels = tf.math.argmax(batchlabel, axis=1)
				numequal = tf.math.equal(predictions, truelabels)
				numcorrect = tf.math.count_nonzero(numequal)
				currnumcorrect += numcorrect.eval()

				#_, _, preds, total_loss = sess.run([encoding, decoding, y_hat, loss], feed_dict={inputs_batch : X, labels_batch : Y})
				#print ("Epoch " + str(iteration+1) + ", Update Number " + str(i)+ ", Summed Cost : "  + str(total_loss))
				cost_list.append(curr_loss)
				#print(curr_loss)
			accuracy = currnumcorrect / float(X.shape[0])
			print("Epoch " + str(iteration+1) + ", Train Accuracy: " + str(accuracy))

			logits = sess.run([y_hat], feed_dict={inputs_batch : X_dev, labels_batch : Y_dev})
			logits = np.asarray(logits[0])
			predictions = tf.math.argmax(tf.nn.softmax(logits), axis=1)

			truelabels = tf.math.argmax(Y_dev, axis=1)
			numequal = tf.math.equal(predictions, truelabels)
			numcorrect = tf.math.count_nonzero(numequal)
			devaccuracy = numcorrect.eval() / float(X_dev.shape[0])
			print("Epoch " + str(iteration+1) + ", Dev Accuracy: " + str(devaccuracy))
			train_accuracies.append(accuracy)
			dev_accuracies.append(devaccuracy)
			train_smoothed_cost = float(sum(cost_list)) / len(cost_list)
			loss_per_epoch.append(train_smoothed_cost)
			#print("Lists for plotting: ")
			#print("Train accuracies: ", train_accuracies)
			#print("Dev accuracies: ", dev_accuracies)
			#print("Train smoothed cost: ", loss_per_epoch)

			saver.save(sess, './modelWeights/tfsoftmax/tfsoftmax', global_step = (iteration+1))
			objectives_summary = tf.Summary()
			objectives_summary.value.add(tag='train_accuracy', simple_value=accuracy)
			objectives_summary.value.add(tag='dev_accuracy', simple_value=devaccuracy)
			objectives_summary.value.add(tag='train_smoothed_cost', simple_value=train_smoothed_cost)
			summary_writer.add_summary(objectives_summary, iteration+1)
			summary_writer.flush()

def main():
	songs, labels = load_data()

	# Shuffling training set
	ind_list=[i for i in range(songs.shape[0])]
	shuffle(ind_list)
	songs = songs.iloc[ind_list]
	labels = labels.iloc[ind_list]
	songs_train = songs.iloc[0:6000]
	songs_dev = songs.iloc[6000:]
	labels_train = labels.iloc[0:6000]
	labels_dev = labels.iloc[6000:]

	# Write dev values to a csv for testing:
	'''songs_dev.to_csv('songs_dev.csv', index = False)
	labels_dev.to_csv('labels_dev.csv', index = False)'''

	train(songs_train, labels_train, songs_dev, labels_dev)

if __name__== "__main__":
	main()
