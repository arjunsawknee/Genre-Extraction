import numpy as np
import tensorflow as tf
import pandas as pd
import math

input_dim = 40000
alpha = 0.005
num_epochs = 200
batch_size = 16
classificationweight = 1

def load_data(songs_path, labels_path):
	print("Loading data......")
	songs = pd.read_csv(songs_path)
	labels = pd.read_csv(labels_path)
	print("Loaded data.")
	return songs, labels

def get_placeholders():
	inputs_placeholder = tf.placeholder(tf.float32, (None, input_dim))
	labels_placeholder = tf.placeholder(tf.float32, (None, input_dim))
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
	return weights

def encoder(inputs_batch, weights):
	a_1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_batch, weights["W1_encoder"]),weights["b1_encoder"]))
	a_2 = tf.nn.relu(tf.add(tf.matmul(a_1, weights["W2_encoder"]),weights["b2_encoder"]))
	return a_2

def decoder(inputs_batch, weights):
	a_3 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_batch, weights["W1_decoder"]),weights["b1_decoder"]))
	a_4 = tf.nn.relu(tf.add(tf.matmul(a_3, weights["W2_decoder"]),weights["b2_decoder"]))
	return a_4

def get_batches(seq, size=batch_size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def train(X, Y):
	inputs_batch, labels_batch = get_placeholders()
	weights = add_parameters()
	encoding = encoder(inputs_batch, weights)
	y_hat = decoder(encoding, weights)
	#loss = tf.reduce_mean(tf.pow(y_hat - labels_batch, 2))
	loss = tf.reduce_mean(tf.pow(y_hat - inputs_batch, 2))
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for iteration in range(num_epochs):
			inputs_batches = get_batches(X)
			cost_list = []
			for i in range(len(inputs_batches)):
				batch = inputs_batches[i]
				bottleneck, preds, _, curr_loss = sess.run([encoding, y_hat, optimizer, loss], feed_dict={inputs_batch: batch, labels_batch: batch})
				_, _, total_loss = sess.run([encoding, y_hat, loss], feed_dict={inputs_batch : X, labels_batch : X})
				print ("Epoch " + str(iteration+1) + ", Update Number " + str(i)+ ", Summed Cost : "  + str(total_loss))
				cost_list.append(curr_loss)

def main():
	songs, labels = load_data('songs.csv','labels.csv')
	train(songs, labels)

if __name__== "__main__":
	main()
