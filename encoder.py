import numpy as np
import tensorflow as tf
import pandas as pd
import math
from softmaxautoencoder import load_data
from random import shuffle

input_dim = 500
alpha = 0.0001
num_epochs = 400
batch_size = 512
num_classes = 4

def get_placeholders():
	inputs_placeholder = tf.placeholder(tf.float32, (None, input_dim))
	labels_placeholder = tf.placeholder(tf.float32, (None, num_classes))
	keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
	tf.add_to_collection('inputs_placeholder', inputs_placeholder)
	tf.add_to_collection('labels_placeholder', labels_placeholder)
	tf.add_to_collection('keep_prob', keep_prob)
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

def get_batches(seq, size=batch_size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def train(X, Y):
	tf.reset_default_graph()
	inputs_batch, labels_batch, keep_prob = get_placeholders()
	weights = add_parameters()
	encoding = encoder(inputs_batch, weights, keep_prob)
	decoding = decoder(encoding, weights, keep_prob)
	tf.add_to_collection("encoding", encoding)
	tf.add_to_collection("decoding", decoding)
	loss = tf.reduce_mean(tf.pow(decoding - inputs_batch, 2))
	tf.add_to_collection("loss", loss)
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep=5)
	with tf.Session() as sess:
		sess.run(init)
		for iteration in range(num_epochs):
			inputs_batches = get_batches(X)
			labels_batches = get_batches(Y)
			cost_list = []
			for i in range(len(inputs_batches)):
				batch = inputs_batches[i]
				batchlabel = labels_batches[i]
				bottleneck, preds, _, curr_loss = sess.run([encoding, decoding, optimizer, loss], feed_dict={inputs_batch: batch, labels_batch: batchlabel, keep_prob : 0.8})
				#_, _, total_loss = sess.run([encoding, decoding, loss], feed_dict={inputs_batch : X, labels_batch : X, keep_prob : 1.0})
				print ("Epoch " + str(iteration+1) + ", Update Number " + str(i)+ ", Summed Cost : "  + str(curr_loss))
				cost_list.append(curr_loss)
			saver.save(sess, './modelWeights/autoencoder/autoencoder', global_step = (iteration+1))

def main():
	songs, labels = load_data()
	# Shuffling training set
	ind_list=[i for i in range(songs.shape[0])]
	shuffle(ind_list)
	songs = songs.iloc[ind_list]
	labels = labels.iloc[ind_list]
	train(songs, labels)

if __name__== "__main__":
	main()
