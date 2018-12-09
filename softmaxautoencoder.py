import numpy as np
import tensorflow as tf
import pandas as pd
import math
from random import shuffle
import os
import sox
import librosa

# OPTIMAL HYPERPARAMETERS: alpha: 0.0001, classificationweight = 0.5, input_dim = 500, (artificial sr = 500 Hz), 
#input_dim = 40000
#input_dim = 20000
input_dim = 500

numexamples = 8000

num_classes = 4
alpha = 0.0001
num_epochs = 200
batch_size = 512
classificationweight = 0.1

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
	songs = np.zeros((numexamples, input_dim))
	onehotlabels = np.zeros((numexamples, num_classes))
	counter = 0

	#allgenres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	# Use four classes only
	allgenres = ['classical', 'jazz', 'metal', 'pop']
	# Splits each song into 10 examples of shape (40000, 1) ~ 2 seconds each
	'''numsplit = 10
	sizesplit = 40000'''

	# splits of 1 second each:
	# numexamples should be numsplit * 1000, since we have 1000 songs
	numsplit = 20
	sizesplit = input_dim
	# generalized loop to process data
	for index in range(len(allgenres)):
		for filename in os.listdir('./genres/' + allgenres[index]):
			if filename.endswith(".wav"):
				audio, sr = librosa.core.load('./genres/' + allgenres[index] + '/' + filename)

				# Takes mean of values to form artificial sampling rate of 500Hz
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
	keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
	return inputs_placeholder, labels_placeholder, keep_prob

def add_parameters():
	weights = {}
	weights["W1_encoder"] = tf.get_variable(name="W1_encoder", shape = (input_dim, 256), initializer = tf.contrib.layers.xavier_initializer())
	weights["W2_encoder"] = tf.get_variable(name="W2_encoder", shape = (256, 192), initializer = tf.contrib.layers.xavier_initializer())
	weights["W3_encoder"] = tf.get_variable(name="W3_encoder", shape = (192, 128), initializer = tf.contrib.layers.xavier_initializer())
	weights["W4_encoder"] = tf.get_variable(name="W4_encoder", shape = (128, 64), initializer = tf.contrib.layers.xavier_initializer())
	
	weights["W1_decoder"] = tf.get_variable(name="W1_decoder", shape = (64, 128), initializer = tf.contrib.layers.xavier_initializer())
	weights["W2_decoder"] = tf.get_variable(name="W2_decoder", shape = (128, 192), initializer = tf.contrib.layers.xavier_initializer())
	weights["W3_decoder"] = tf.get_variable(name="W3_decoder", shape = (192, 256), initializer = tf.contrib.layers.xavier_initializer())
	weights["W4_decoder"] = tf.get_variable(name="W4_decoder", shape = (256, input_dim), initializer = tf.contrib.layers.xavier_initializer())

	weights["b1_encoder"] = tf.get_variable(name="b1_encoder", initializer = tf.zeros((1,256)))
	weights["b2_encoder"] = tf.get_variable(name="b2_encoder", initializer = tf.zeros((1,192)))
	weights["b3_encoder"] = tf.get_variable(name="b3_encoder", initializer = tf.zeros((1,128)))
	weights["b4_encoder"] = tf.get_variable(name="b4_encoder", initializer = tf.zeros((1,64)))
	
	weights["b1_decoder"] = tf.get_variable(name="b1_decoder", initializer = tf.zeros((1,128)))
	weights["b2_decoder"] = tf.get_variable(name="b2_decoder", initializer = tf.zeros((1,192)))
	weights["b3_decoder"] = tf.get_variable(name="b3_decoder", initializer = tf.zeros((1,256)))
	weights["b4_decoder"] = tf.get_variable(name="b4_decoder", initializer = tf.zeros((1, input_dim)))

	# Softmax classifier weights
	weights["W1_softmax"] = tf.get_variable(name="W1_softmax", shape = (64, 32), initializer = tf.contrib.layers.xavier_initializer())
	weights["b1_softmax"] = tf.get_variable(name="b1_softmax", initializer = tf.zeros((1,32)))
	weights["W2_softmax"] = tf.get_variable(name="W2_softmax", shape = (32, 16), initializer = tf.contrib.layers.xavier_initializer())
	weights["b2_softmax"] = tf.get_variable(name="b2_softmax", initializer = tf.zeros((1,16)))
	weights["W3_softmax"] = tf.get_variable(name="W3_softmax", shape = (16, num_classes), initializer = tf.contrib.layers.xavier_initializer())
	weights["b3_softmax"] = tf.get_variable(name="b3_softmax", initializer = tf.zeros((1,num_classes)))
	return weights

def encoder(inputs_batch, weights, keep_prob):
	a_1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_batch, weights["W1_encoder"]),weights["b1_encoder"]))
	#a_1 = tf.nn.dropout(a_1, keep_prob)
	a_2 = tf.nn.tanh(tf.add(tf.matmul(a_1, weights["W2_encoder"]),weights["b2_encoder"]))
	a_2 = tf.nn.dropout(a_2, keep_prob)
	a_3 = tf.nn.relu(tf.add(tf.matmul(a_2, weights["W3_encoder"]),weights["b3_encoder"]))
	a_4 = tf.nn.relu(tf.add(tf.matmul(a_3, weights["W4_encoder"]),weights["b4_encoder"]))
	return a_4

def decoder(inputs_batch, weights, keep_prob):
	a_5 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_batch, weights["W1_decoder"]),weights["b1_decoder"]))
	#a_4 = tf.nn.dropout(a_4, keep_prob)
	a_6 = tf.nn.sigmoid(tf.add(tf.matmul(a_5, weights["W2_decoder"]),weights["b2_decoder"]))
	a_6 = tf.nn.dropout(a_6, keep_prob)
	a_7 = tf.nn.relu(tf.add(tf.matmul(a_6, weights["W3_decoder"]),weights["b3_decoder"]))
	a_8 = tf.nn.relu(tf.add(tf.matmul(a_7, weights["W4_decoder"]),weights["b4_decoder"]))
	return a_8

# Convention of h for hidden layers of classifier
def softmaxclassifier(inputs_batch, weights, keep_prob):
	h_1  = tf.nn.tanh(tf.add(tf.matmul(inputs_batch, weights["W1_softmax"]), weights["b1_softmax"]))
	h_2  = tf.nn.tanh(tf.add(tf.matmul(h_1, weights["W2_softmax"]), weights["b2_softmax"]))

	# Remove softmax from here
	h_3 = tf.add(tf.matmul(h_2, weights["W3_softmax"]), weights["b3_softmax"])
	return h_3

def get_batches(seq, size=batch_size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def train(X, Y, X_dev, Y_dev):
	tf.reset_default_graph()
	inputs_batch, labels_batch, keep_prob = get_placeholders()
	weights = add_parameters()
	encoding = encoder(inputs_batch, weights, keep_prob)
	decoding = decoder(encoding, weights, keep_prob)
	tf.add_to_collection("encoding", encoding)
	tf.add_to_collection("decoding", decoding)

	y_hat = softmaxclassifier(encoding, weights, keep_prob)
	tf.add_to_collection("y_hat", y_hat)

	# Check shape of labels: need to be shape (batch_size, num_classes) according to documentation
	loss = tf.reduce_mean(tf.pow(decoding - inputs_batch, 2)) + tf.reduce_mean((classificationweight * tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels_batch), logits=y_hat)))
	
	# Loss without autnencoder portion
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels_batch), logits=y_hat))
	tf.add_to_collection("loss", loss)

	# Just reconstruction loss: order of magnitude 0.005-0.01
	#loss = tf.reduce_mean(tf.pow(decoding - inputs_batch, 2))
	# Just cross entropy loss: order of magnitude 1.5 - 2.5
	#loss = (classificationweight * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_batch, logits=y_hat))
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=5)
	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter('./tensorboardlogs/softmaxautoencoder', sess.graph)
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


				bottleneck, reconstruction, preds, _, curr_loss = sess.run([encoding, decoding, y_hat, optimizer, loss], feed_dict={inputs_batch: batch, labels_batch: batchlabel, keep_prob : 0.8})
				#Check if predictions are correct and increment counter/accuracy
				predictions = tf.math.argmax(preds, axis=1)
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
			_, preds = sess.run([encoding, y_hat], feed_dict={inputs_batch : X_dev, labels_batch : Y_dev, keep_prob : 1.0})
			predictions = tf.math.argmax(preds, axis=1)
			truelabels = tf.math.argmax(Y_dev, axis=1)
			numequal = tf.math.equal(predictions, truelabels)
			numcorrect = tf.math.count_nonzero(numequal)
			devaccuracy = numcorrect.eval() / float(X_dev.shape[0])
			print("Epoch " + str(iteration+1) + ", Dev Accuracy: " + str(devaccuracy))
			train_accuracies.append(accuracy)
			dev_accuracies.append(devaccuracy)
			train_smoothed_cost = float(sum(cost_list)) / len(cost_list)
			loss_per_epoch.append(train_smoothed_cost)
			'''print("Lists for plotting: ")
			print("Train accuracies: ", train_accuracies)
			print("Dev accuracies: ", dev_accuracies)
			print("Train smoothed cost: ", loss_per_epoch)'''

			saver.save(sess, './modelWeights/softmaxautoencoder', global_step = (iteration+1))
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
	'''songs_train = songs.iloc[0:18000]
	songs_dev = songs.iloc[18000:]
	labels_train = labels.iloc[0:18000]
	labels_dev = labels.iloc[18000:]'''

	# In the case of 8000 examples
	songs_train = songs.iloc[0:6000]
	songs_dev = songs.iloc[6000:]
	labels_train = labels.iloc[0:6000]
	labels_dev = labels.iloc[6000:]

	# Write dev values to a csv for testing:
	songs_dev.to_csv('songs_dev.csv', index = False)
	labels_dev.to_csv('labels_dev.csv', index = False)

	train(songs_train, labels_train, songs_dev, labels_dev)

if __name__== "__main__":
	main()
