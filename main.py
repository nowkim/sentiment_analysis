import numpy as np
import tensorflow as tf

from config import get_config
from prepro import read_data, write_wordic, one_hot
from model import basicLSTM


def main():
	config = get_config()
	
	# read data
	raw_train_data, raw_test_data, max_len_context = read_data(config)
	
	# write a word dictionary
	unk_train_data, unk_test_data, wordic = write_wordic(config, raw_train_data, raw_test_data)
	
	# one hot encoding
	train_X, train_Y, test_X, test_Y = one_hot(unk_train_data, unk_test_data, max_len_context, wordic)
	

	basicLSTM(config, len(wordic), max_len_context, train_X, train_Y)



if __name__ == "__main__":
	main()

