import tensorflow as tf


def get_config():
	flags = tf.app.flags

	flags.DEFINE_string("data_dir", "data/txt_sentoken", "Data dir")
	
	flags.DEFINE_string("word_freq_limit", 10, "Data dir")

	flags.DEFINE_string("mode", "train", "mode")
	flags.DEFINE_string("initial_scale", 0.05, "initial_scale")

	flags.DEFINE_string("train_data_ratio", 0.8, "train data")
	
	# MODEL
	flags.DEFINE_string("lstm_hidden_size", 128, "hidden size of lstm")
	flags.DEFINE_string("batch_size", 5, "Batch size")
	flags.DEFINE_string("keep_prob", 0.5, "drop out")
	flags.DEFINE_string("num_lstm_layers", 2, "the number of lstm layers")
	flags.DEFINE_string("lr", 0.001, "learning rate")


	config = flags.FLAGS

	print("word freq limitation", config.word_freq_limit)
	print("batch size : ", config.batch_size)

	return config

