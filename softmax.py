import numpy as np
import tensorflow as tf
import pandas as pd
import math

def get_one_hot(label_num, num_classes = 10):
    one_hot = np.zeros((num_classes,1))
    one_hot[int(label_num),0] = 1
    return one_hot

def initialize_parameters(vector_dim = 40000, num_classes = 10):
	w = np.random.randn(num_classes, vector_dim)
	b = np.zeros((num_classes, 1))
	return w, b

#def softmax(logits):
	#return np.exp(logits) / np.sum(np.exp(logits), axis=0)
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

def back_propagate(weights, bias, y_hat, y, inputs, learning_rate = 0.01):
	db = y_hat-y
	dw = db.dot(inputs.T)
	new_weights = weights - learning_rate*dw
	new_bias = bias - learning_rate*db
	return new_weights, new_bias


def train(songs, labels):
	w, b = initialize_parameters()
	for i in range(1000):
		smoothed_cost_list = []
		correct_class = 0
		attempts = 0
		for j, row in songs.iterrows():
			label = labels.iloc[j]
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

def main():
	songs = pd.read_csv('songs.csv')
	print("Loaded songs csv")
	labels = pd.read_csv('labels.csv')
	print("Loaded labels csv")
	train(songs, labels)
  
if __name__== "__main__":
	main()
