import numpy as np
import tensorflow as tf

from config import get_config
from prepro import read_data, write_wordic, one_hot
from model import basicLSTM


def main():
	config = get_config()
	
	# read data
	train_data, test_data, max_len_context, train_x_len, test_x_len = read_data(config)
	
	# write a word dictionary
	wordic = write_wordic(config, train_data)
	
	

	basicLSTM(config, wordic, max_len_context, train_x_len, test_x_len, train_data, test_data)

	'''
	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
		
		_build(config)
	
		_feed(config, train_X, train_Y, test_X, test_Y)
	'''


if __name__ == "__main__":
	main()

