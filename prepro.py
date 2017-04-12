import numpy as np
import os
import re


def read_data(config):
	sentiments = ['neg', 'pos']
	train_data = []
	test_data = []
	max_len_context = 0

	for sentiment in sentiments:
		data_cnt = 0
		for roots, dirs, files in os.walk(os.path.join(config.data_dir, sentiment)):
			for file_name in files:
				with open(os.path.join(roots, file_name), "r", encoding='utf-8', errors='ignore') as f:
					data_cnt += 1
					context = tokenize(str(f.readlines()))
					if len(context) > max_len_context:
						max_len_context = len(context)
					if data_cnt <= len(files) * config.train_data_ratio * 0.02:
						train_data.append({"context" : context, "sentiment" : sentiment})
					elif data_cnt <= len(files) * 0.02:
						test_data.append({"context" : context, "sentiment" : sentiment})
					else :
						break
	
	print("Loading {} train data".format(len(train_data)))
	print("Loading {} test data".format(len(test_data)))
	print("max length of context :  {}".format(max_len_context))
	
	return train_data, test_data, max_len_context


def tokenize(context):
	context = re.sub('[=#&+_$*/,.\-;:()\\[\]?!"\']+', ' ', context)
	context = re.sub('[0-9]+', '', context)
	context = context.replace('\\n', ' ').replace('\\','')
	context = context.split()
	return context


def write_wordic(config, train_data, test_data):
	print('writing a dictionary . . . ')
	wordic = ['$UNK$']
	word_freq = {}
	all_data = train_data + test_data
	for _, contents in enumerate(all_data):
		for word in contents["context"]:
			if not word in word_freq.keys():
				word_freq[word] = 1
			else:
				word_freq[word] += 1
	for w,freq in word_freq.items():
		if freq	< config.word_freq_limit:
			for i, contents in enumerate(all_data):
				for j,word in enumerate(contents['context']):
					all_data[i]['context'][j] = '$UNK$'
			continue
		else:
			if not w in wordic:
				wordic.append(w)

	print("There are {} words in the dictionary".format(len(wordic)))

	return all_data[0:len(train_data)-1], all_data[len(train_data):len(all_data)-1], wordic


def one_hot(train_data, test_data, max_len_context, wordic):
	print("--- one hot encoding ---")
	train_X = np.array([[[]]])
	train_Y = np.array([[]])

	test_X = np.array([[[]]])
	test_Y = np.array([[]])

	loop = 0
	for data in [train_data, test_data]:
		for _, contents in enumerate(data):
			loop += 1
			one_hot = np.array([[[int(i == wordic.index(contents['context'][j])) if j<len(contents['context']) else 0 for i in range(len(wordic))] for j in range(max_len_context)]])
			print('onehot shape',one_hot.shape)
			print('x shape',train_X.shape)
			print('y shape',train_Y.shape)
			if loop % 10 == 0:
				print("{}% of the process has been completed".format(loop/len(train_data+test_data)*100))
			if loop == 1:
				train_X = one_hot
				train_Y = np.array([[int(contents['sentiment']=="neg"), int(contents['sentiment']=="pos")]])
				continue
			elif loop <= len(train_data):
				train_X = np.concatenate((train_X, one_hot))
				train_Y = np.concatenate((train_Y, np.array([[int(contents['sentiment']=="neg"), int(contents['sentiment']=="pos")]])))
			elif loop == len(train_data)+1:
				test_X = one_hot
				test_Y = np.array([[int(contents['sentiment']=="neg"), int(contents['sentiment']=="pos")]])
				continue	
			else:
				test_X = np.concatenate((test_X, one_hot))
				test_Y = np.concatenate((test_Y, np.array([[int(contents['sentiment']=="neg"), int(contents['sentiment']=="pos")]])))

	
	return train_X, train_Y, test_X, test_Y




