import tensorflow as tf


def get_config():
	flags = tf.app.flags

	flags.DEFINE_string("data_dir", "data/rt-polaritydata", "Data dir")
	
	flags.DEFINE_string("word_freq_limit", 5, "Data dir")
	flags.DEFINE_string("save_path", "/tmp/model.ckpt", "saver.save path")

	flags.DEFINE_string("init_scale", 0.05, "initial_scale")
	flags.DEFINE_string("pre_trained", True, "if you want to use a pre-trained word embedding, you should set this value to 'true'")

	flags.DEFINE_string("train_data_ratio", 0.6, "train data")
	flags.DEFINE_string("valid_data_ratio", 0.8, "train ratio + valid ratio")
	
	flags.DEFINE_string("cross_validation", False, "if you want to use cross-validation, you should set this value to 'true'")
	flags.DEFINE_string("cross_valid_k", 5, "k-folds cross validation")

	# Word Embedding
	flags.DEFINE_string("embedding_size", 300, "50, 100, 200, 300 only")

	# LSTM
	flags.DEFINE_string("lstm_hidden_size", 200, "hidden size of lstm")
	flags.DEFINE_string("keep_prob", 0.5, "drop out")
	flags.DEFINE_string("num_lstm_layer", 1, "the number of lstm layers")
	flags.DEFINE_string("lr", 1e-3, "learning rate")
	flags.DEFINE_string("lr_decay", 0.8, "the decay of the learning rate for each epoch after 'lr_init_epoch'")
	flags.DEFINE_string("lr_init_epoch", 1, "the number of epochs trained with the initial learning rate")
	flags.DEFINE_string("max_grad_norm", 5, "gradient clipping")

	# CNN
	flags.DEFINE_string("filter_sizes", [3,4,5], "Comma-separated filter sizes (default: '3,4,5')")
	flags.DEFINE_string("num_filters", 128, "The number fo filters per filter size")

	# Train
	flags.DEFINE_string("batch_size", 40, "Batch size")
	flags.DEFINE_string("max_epoch", 100, "epoch")


	config = flags.FLAGS

	print("word freq limitation", config.word_freq_limit)
	print("pre-trained word embedding : ", str(config.pre_trained))
	print("batch size : ", config.batch_size)
	print("max epochs : ", config.max_epoch)
	print("drop out(keep) : {}%".format(int(config.keep_prob*100)))

	return config

