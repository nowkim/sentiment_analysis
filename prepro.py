import numpy as np
import os
import sys
import re
import math
import gensim


def gensim_word2vec(config):
	print("word embedding by gensim")
	sentiments = ['neg', 'pos']
	sentences = []
	max_len_context = 0

	###  read   ###
	for sentiment in sentiments:
		data_idx = 0
		for roots, dirs, files in os.walk(os.path.join(config.data_dir, sentiment)):
			for file_name in files:
				with open(os.path.join(roots, file_name), "r", encoding='utf-8', errors='ignore') as f:
					while True:
						line = f.readline()
						if not line: break
						words = tokenize(line)
						if len(words) > max_len_context:
							max_len_context = len(words)
						sentences.append(words)
				paras.append(sentences)
	
	print("the number of sentences : {}".format(len(sentences)))
	print("max len context : {}".format(max_len_context))


	model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)




def read_data(config):
	sentiments = ['neg', 'pos']
	all_data = []
	max_len_context = 0
	train_data = []
	valid_data = []
	test_data = []
	data = {}

	'''
	###  read   ###
	for sentiment in sentiments:
		for roots, dirs, files in os.walk(os.path.join(config.data_dir, sentiment)):
			for file_name in files:
				with open(os.path.join(roots, file_name), "r", encoding='utf-8', errors='ignore') as f:
					context = tokenize(str(f.readlines()))
					if len(context) > max_len_context:
						max_len_context = len(context)
					all_data.append({'context' : context, 'sentiment' : sentiment, 'seqlen' : len(context)})
	'''
	###  read   ###
	for sentiment in sentiments:
		with open(os.path.join(config.data_dir, "rt-polarity.{}".format(sentiment)), "r", encoding='utf-8', errors='ignore') as f:
			while True:
				context = f.readline().split()
				if not context: break
				if len(context) > max_len_context:
					max_len_context = len(context)
				all_data.append({'context' : context, 'sentiment' : sentiment, 'seqlen' : len(context)})
	# shuffle all data #
	np.random.shuffle(all_data)
	# padding #
	all_data = padding(config, max_len_context, all_data)

	train_data = all_data[:int(len(all_data)*config.train_data_ratio)]
	valid_data = all_data[int(len(all_data)*config.train_data_ratio):int(len(all_data)*config.valid_data_ratio)]
	test_data = all_data[int(len(all_data)*config.valid_data_ratio):]

	print("Loading {} all data".format(len(all_data)))
	print("Loading {} train data".format(len(train_data)))
	print("Loading {} valid data".format(len(valid_data)))
	print("Loading {} test data".format(len(test_data)))
	
	print("max length of context :  {}".format(max_len_context))
		
	data['train_data'] = train_data
	data['valid_data'] = valid_data
	data['test_data'] = test_data

	return data, max_len_context



def padding(config, max_len_context, data):
	###  padding  ###
	for data_idx, contents in enumerate(data):
		if len(contents['context']) < max_len_context:
			for _ in range(len(contents['context']), max_len_context):
				data[data_idx]['context'].append("$PAD$")
	print("the 'padding' process has been completed")
	
	return data



def cross_validation(config, all_data):
	num_of_folds = config.cross_valid_k
	
	print("cross-validation : {}".format(config.cross_validation))
	test_data = all_data[int(len(all_data)*config.train_data_ratio):len(all_data)]
	print('test data len', len(test_data))
	cross_data = []
	data = {}
	fold_size = int(len(all_data)*config.train_data_ratio) // config.cross_valid_k
	for fold_idx in range(config.cross_valid_k) :
		print("fold {}".format(fold_idx+1))
		train_data = []
		valid_data = []
		cross_data_tmp = {}
		train_start = fold_idx * fold_size
		valid_start = int((fold_size*(fold_idx+num_of_folds*config.train_data_ratio))%len(all_data))
		train_end = int(train_start+fold_size*num_of_folds*config.train_data_ratio)
		valid_end = int(valid_start+fold_size)
		print(train_start, train_end, valid_start, valid_end)
		
		if train_end <= len(all_data):
			train_data = all_data[train_start:train_end]
		else:
			train_tmp1 = all_data[0:train_end-len(all_data)]
			train_tmp2 = all_data[train_start:len(all_data)]
			train_data.extend(train_tmp1)
			train_data.extend(train_tmp2)
			
		valid_data = all_data[valid_start:valid_end]
		print(len(train_data), len(valid_data), len(test_data))
		cross_data_tmp['train_data'] = train_data
		cross_data_tmp['valid_data'] = valid_data
		cross_data.append(cross_data_tmp)
	
	data['cross_data'] = cross_data
	data['test_data'] = test_data

	return data


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


