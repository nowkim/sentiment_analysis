import numpy as np
import os
import re
import math


def read_data(config):
	sentiments = ['neg', 'pos']
	train_data = []
	test_data = []
	max_len_context = 0
	train_x_len = []
	test_x_len = []


	###  read   ###
	for sentiment in sentiments:
		data_idx = 0
		for roots, dirs, files in os.walk(os.path.join(config.data_dir, sentiment)):
			for file_name in files:
				with open(os.path.join(roots, file_name), "r", encoding='utf-8', errors='ignore') as f:
					data_idx += 1
					context = tokenize(str(f.readlines()))
					if len(context) > max_len_context:
						max_len_context = len(context)
					if data_idx <= len(files) * config.train_data_ratio:
						train_data.append({"context" : context, "sentiment" : sentiment})
						train_x_len.append(len(context))
					elif data_idx <= len(files):
						test_data.append({"context" : context, "sentiment" : sentiment})
						test_x_len.append(len(context))
					else:
						break

	print("Loading {} train data".format(len(train_data)))
	print("Loading {} test data".format(len(test_data)))
	print("max length of context :  {}".format(max_len_context))
	
	
	###  padding  ###
	for data_idx, data in enumerate(train_data + test_data):
		if len(data["context"]) < max_len_context:
			for _ in range(len(data["context"]), max_len_context):
				if data_idx < len(train_data+test_data) * config.train_data_ratio:
					train_data[data_idx]["context"].append("$PAD$")
				else:
					test_data[data_idx - len(train_data)]["context"].append("$PAD$")
	
	np.random.shuffle(train_data)
	np.random.shuffle(test_data)	
			
	print("the 'padding' process has been completed")

	return train_data, test_data, max_len_context, train_x_len, test_x_len


def tokenize(context):
	context = re.sub('[=#&+_$*/,.\-;:()\\[\]?!"\']+', ' ', context)
	context = re.sub('[0-9]+', '', context)
	context = context.replace('\\n', ' ').replace('\\','')
	context = context.split()
	return context


def write_wordic(config, train_data):
	print('writing a dictionary . . . ')
	wordic = ['$UNK$','$PAD$']
	word_freq = {}

	for _, contents in enumerate(train_data):
		for word in contents["context"]:
			if not word in word_freq.keys():
				word_freq[word] = 1
			else:
				word_freq[word] += 1

	for w,freq in word_freq.items():
		if not w in wordic and freq >= config.word_freq_limit:
			wordic.append(w)

	print("There are {} words in the dictionary".format(len(wordic)))

	return wordic



def one_hot(data_X, data_Y, max_len_context, wordic):
	input_X = np.array([[[]]])
	input_Y = np.array([[]])

	for loop, data in enumerate(data_X):
		one_hot = np.zeros((max_len_context, len(wordic)))
		one_hot[np.arange(max_len_context), 
				np.array([wordic.index(data[i]) if data[i] in wordic else wordic.index('$UNK$') for i in range(max_len_context)])] = 1
		one_hot = one_hot.reshape(1, max_len_context, len(wordic))

			#one_hot = np.array([[[int(i == wordic.index(contents['context'][j])) for i in range(len(wordic))] if contents['context'][j] in wordic else [int(i == wordic.index('$UNK$')) for i in range(len(wordic))] for j in range(max_len_context)]])
		if loop == 0:
			input_X = one_hot
			input_Y = np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])
			continue
		else:
			input_X = np.concatenate((input_X, one_hot))
			input_Y = np.concatenate((input_Y, np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])))

	print(input_X.shape, input_Y.shape)

	return input_X, input_Y


def word2index(data_X, data_Y, wordic):
	input_X = data_X
	input_Y = np.array([[]])
	for i, para in enumerate(input_X):
		for j, word in enumerate(para):
			if word in wordic:
				input_X[i][j] = wordic.index(word)
			else:
				input_X[i][j] = wordic.index('$UNK$')
	for loop, senti in enumerate(data_Y):
		if loop == 0:
			input_Y = np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])
		else:
			input_Y = np.concatenate((input_Y, np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])))

	return input_X, input_Y


