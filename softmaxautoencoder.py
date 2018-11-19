import numpy as np
import tensorflow as tf
import pandas as pd
import math
from random import shuffle

input_dim = 40000
num_classes = 10
alpha = 0.0001
num_epochs = 200
batch_size = 16
classificationweight = 0.000001

def load_data(songs_path, labels_path):
	print("Loading data...")
	songs = pd.read_csv(songs_path)
	labels = pd.read_csv(labels_path)
	print("Congrats mate.")
	return songs, labels

def get_placeholders():
	inputs_placeholder = tf.placeholder(tf.float32, (None, input_dim))
	labels_placeholder = tf.placeholder(tf.float32, (None, num_classes))
	return inputs_placeholder, labels_placeholder

def add_parameters():
	weights = {}
	weights["W1_encoder"] = tf.get_variable(name="W1_encoder", shape = (input_dim, 1024), initializer = tf.contrib.layers.xavier_initializer())
	weights["W2_encoder"] = tf.get_variable(name="W2_encoder", shape = (1024, 512), initializer = tf.contrib.layers.xavier_initializer())
	weights["W1_decoder"] = tf.get_variable(name="W1_decoder", shape = (512, 1024), initializer = tf.contrib.layers.xavier_initializer())
	weights["W2_decoder"] = tf.get_variable(name="W2_decoder", shape = (1024, input_dim), initializer = tf.contrib.layers.xavier_initializer())
	weights["b1_encoder"] = tf.get_variable(name="b1_encoder", initializer = tf.zeros((1,1024)))
	weights["b2_encoder"] = tf.get_variable(name="b2_encoder", initializer = tf.zeros((1,512)))
	weights["b1_decoder"] = tf.get_variable(name="b1_decoder", initializer = tf.zeros((1,1024)))
	weights["b2_decoder"] = tf.get_variable(name="b2_decoder", initializer = tf.zeros((1, input_dim)))

	# Softmax classifier weights
	weights["W1_softmax"] = tf.get_variable(name="W1_softmax", shape = (512, 128), initializer = tf.contrib.layers.xavier_initializer())
	weights["b1_softmax"] = tf.get_variable(name="b1_softmax", initializer = tf.zeros((1,128)))
	weights["W2_softmax"] = tf.get_variable(name="W2_softmax", shape = (128, num_classes), initializer = tf.contrib.layers.xavier_initializer())
	weights["b2_softmax"] = tf.get_variable(name="b2_softmax", initializer = tf.zeros((1,num_classes)))
	return weights

def encoder(inputs_batch, weights):
	a_1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_batch, weights["W1_encoder"]),weights["b1_encoder"]))
	a_2 = tf.nn.relu(tf.add(tf.matmul(a_1, weights["W2_encoder"]),weights["b2_encoder"]))
	return a_2

def decoder(inputs_batch, weights):
	a_3 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_batch, weights["W1_decoder"]),weights["b1_decoder"]))
	a_4 = tf.nn.relu(tf.add(tf.matmul(a_3, weights["W2_decoder"]),weights["b2_decoder"]))
	return a_4

# Convention of h for hidden layers of classifier
def softmaxclassifier(inputs_batch, weights):
	h_1  = tf.nn.tanh(tf.add(tf.matmul(inputs_batch, weights["W1_softmax"]), weights["b1_softmax"]))
	h_2 = tf.nn.softmax(tf.add(tf.matmul(h_1, weights["W2_softmax"]), weights["b2_softmax"]))
	return h_2

def get_batches(seq, size=batch_size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def train(X, Y, X_dev, Y_dev):
	inputs_batch, labels_batch = get_placeholders()
	weights = add_parameters()
	encoding = encoder(inputs_batch, weights)
	decoding = decoder(encoding, weights)

	y_hat = softmaxclassifier(encoding, weights)

	# Check shape of labels: need to be shape (batch_size, num_classes) according to documentation
	loss = tf.reduce_mean(tf.pow(decoding - inputs_batch, 2)) + tf.reduce_mean((classificationweight * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_batch, logits=y_hat)))
	# Just reconstruction loss: order of magnitude 0.005-0.01
	#loss = tf.reduce_mean(tf.pow(decoding - inputs_batch, 2))
	# Just cross entropy loss: order of magnitude 1.5 - 2.5
	#loss = (classificationweight * tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_batch, logits=y_hat))
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		# Shuffling training set
		ind_list=[i for i in range(X.shape[0])]
		shuffle(ind_list)
		X = X.iloc[ind_list]
		Y = Y.iloc[ind_list]
		for iteration in range(num_epochs):
			inputs_batches = get_batches(X)

			labels_batches = get_batches(Y)

			cost_list = []
			currnumcorrect = 0
			for i in range(len(inputs_batches)):
				batch = inputs_batches[i]

				#should be (batch_size, num_classes)
				batchlabel = labels_batches[i]


				bottleneck, reconstruction, preds, _, curr_loss = sess.run([encoding, decoding, y_hat, optimizer, loss], feed_dict={inputs_batch: batch, labels_batch: batchlabel})
				#Check if predictions are correct and increment counter/accuracy
				predictions = tf.math.argmax(preds, axis=1)
				truelabels = tf.math.argmax(batchlabel, axis=1)
				numequal = tf.math.equal(predictions, truelabels)
				numcorrect = tf.math.count_nonzero(numequal)
				currnumcorrect += numcorrect.eval()

				#_, _, preds, total_loss = sess.run([encoding, decoding, y_hat, loss], feed_dict={inputs_batch : X, labels_batch : Y})
				#print ("Epoch " + str(iteration+1) + ", Update Number " + str(i)+ ", Summed Cost : "  + str(total_loss))
				#cost_list.append(curr_loss)
				#print(curr_loss)
			accuracy = currnumcorrect / float(X.shape[0])
			print("Epoch " + str(iteration+1) + ", Train Accuracy: " + str(accuracy))
			_, preds = sess.run([encoding, y_hat], feed_dict={inputs_batch : X_dev, labels_batch : Y_dev})
			predictions = tf.math.argmax(preds, axis=1)
			truelabels = tf.math.argmax(Y_dev, axis=1)
			numequal = tf.math.equal(predictions, truelabels)
			numcorrect = tf.math.count_nonzero(numequal)
			devaccuracy = numcorrect.eval() / float(X_dev.shape[0])
			print("Epoch " + str(iteration+1) + ", Dev Accuracy: " + str(devaccuracy))

def main():
	songs, labels = load_data('songs.csv','onehotlabels.csv')

	# Shuffling training set
	ind_list=[i for i in range(songs.shape[0])]
	shuffle(ind_list)
	songs = songs.iloc[ind_list]
	labels = labels.iloc[ind_list]
	songs_train = songs.iloc[0:800]
	songs_dev = songs.iloc[800:]
	labels_train = labels.iloc[0:800]
	labels_dev = labels.iloc[800:]
	train(songs_train, labels_train, songs_dev, labels_dev)

if __name__== "__main__":
	main()
