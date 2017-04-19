import tensorflow as tf
import numpy as np
from prepro import one_hot, word2index
import sys


def _build(config):
	with tf.name_scope("Train"):
		with tf.variable_scope("Model", reuse=None, initializer=initializer):
			m = lstm_model(config)
	return


def _feed(config, train_X, train_Y, test_X, test_Y):
	sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	sessConfig.gpu_options.allow_growth = True
	sv = tf.train.Supervisor(logdir=FLAGS.save_path)
	with sv.managed_session(config=sessConfig) as session:
		for i in range(config.max_max_epoch):
			_train()

		_test()
	
	return

'''
class lstm_model(config):
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
	stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers, state_is_tuple=False)

	words = tf.placeholder(tf.int32, [batch_size, num_steps])
y
y


	initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
	for i in range(num_steps):
		output, state = stacked_lstm(words[:, i], state)

	final_state = state
'''


def _train():
	numpy_state = initial_state.eval()
	total_loss = 0.0
	for current_batch_of_words in words_in_dataset:
		numpy_state, current_loss = session.run([final_state, loss],
		# Initialize the LSTM state from the previous iteration.
			feed_dict={initial_state: numpy_state, words: current_batch_of_words})
		total_loss += current_loss


def _test():
	numpy_state = initial_state.eval()
	total_loss = 0.0
	for current_batch_of_words in words_in_dataset:
		numpy_state, current_loss = session.run([final_state, loss],
		# Initialize the LSTM state from the previous iteration.
			feed_dict={initial_state: numpy_state, words: current_batch_of_words})
		total_loss += current_loss






def LSTM(config, train_X, train_Y):
	lstm = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_size, state_is_tuple=False)
	state = tf.zeros([config.batch_size, lstm.state_size])
	probabilities = []
	loss = 0.0

	attn_cell = lstm
	if is_training:
		attn_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=config.keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell([attn_cell for _ in range(config.num_lstm_layers)], state_is_tuple=False)



			

def basicLSTM(config, wordic, max_len_context, train_data, test_data):
	
	train_X = []
	train_Y = []
	test_X = []
	test_Y = []
	for data in train_data:
		train_X.append(data['context'])
		train_Y.append(data['sentiment'])
	for data in test_data:
		test_X.append(data['context'])
		test_Y.append(data['sentiment'])


	batch_size = config.batch_size
	lstm_size = config.lstm_hidden_size
	time_step_size = max_len_context
	vocab_size = len(wordic)
	lr = config.lr

	#x = tf.placeholder(tf.float32, [None, time_step_size, vocab_size])
	x = tf.placeholder(tf.int32, [None, time_step_size])
	y = tf.placeholder(tf.int32, [None, 2])
	
	w = tf.Variable(tf.random_normal([lstm_size, 2]))
	b = tf.Variable(tf.random_normal([2]))

	# embedding
	embeddings = tf.Variable(
			tf.random_uniform([vocab_size, config.embedding_size], -1.0, 1.0))
	embed = tf.nn.embedding_lookup(embeddings, x)

	print(embed)

	lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

	state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

	outputs = []
	with tf.variable_scope('rnn') as scope:
		for time_step in range(time_step_size):
			if time_step != 0:
				scope.reuse_variables()
			cell_output, state = lstm_cell(embed[:,time_step,:], state)
			outputs.append(cell_output)

	pred = tf.matmul(outputs[-1], w) + b

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
	tf.summary.scalar("cost_summary", cost)
	train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	tf.summary.scalar("learning rate", lr)

	merged = tf.summary.merge_all()

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()

	sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	sessConfig.gpu_options.allow_growth = True

	print('before sess.run')
	with tf.Session(config=sessConfig) as sess:
		sess.run(init)

		print("batch size : {}\n{} iterations".format(batch_size, len(train_X)//batch_size))
		for iteration in range(len(train_X)//batch_size):
			batch_x = []
			batch_y = []
			for data_idx in range(batch_size):
				batch_x.append(train_X[data_idx])
				batch_y.append(train_Y[data_idx])
			batch_x, batch_y = word2index(batch_x, batch_y, wordic)
			#batch_x, batch_y = one_hot(batch_x, batch_y, max_len_context, wordic)
			sess.run(train, feed_dict={x: batch_x, y: batch_y})
			summary, loss = sess.run([merged, cost], feed_dict={x: batch_x, y: batch_y})
			if (iteration+1) != 1 and (iteration+1) % 5 == 0:
				print("{} iterations, loss : {}".format(iteration+1, loss))
		print("train complete!")
		#print(len(test_X[0]), len(test_X[1]), len(test_X[2]))
		#print(test_X[0].shape, test_X[1].shape, test_X[2].shape)

		test_acc = 0
		for iteration in range(len(test_X)//batch_size):
			tbatch_x = []
			tbatch_y = []
			for data_idx in range(batch_size):
				tbatch_x.append(test_X[data_idx])
				tbatch_y.append(test_Y[data_idx])
			tbatch_x, tbatch_y = one_hot(tbatch_x, tbatch_y, max_len_context, wordic)
			test_acc += accuracy.eval(feed_dict={x: tbatch_x, y: tbatch_y})
			if (iteration+1) != 1 and (iteration+1) % 5 == 0:
				print("{} iterations".format(iteration+1))
		
		print("test accuracy: ", test_acc / (len(test_X)//batch_size))


	'''
	outputs = []
	lstm_cell = tf.contrib.rnn.BasicRNNCell(lstm_size)
	state = tf.zeros([batch_size, lstm_cell.state_size])
	for time_step in range(time_step_size):
		cell_output, state = lstm_cell(train_X[:, time_step], state)
		outputs.append(cell_output)

	output = tf.reshape(tf.concat(axis=1, values=ouputs), [-1, lstm_size])
	softmax_w = tf.get_variable(
		[lstm_size, vocab_size],dtype=tf.float32)
	softmax_b = tf.get_variable(
		[vocab_size], dtype=tf.float32)
	logits = tf.matmul(output, softmax_w) + softmax_b

	loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
		[logits],
		[tf.reshape(train_Y, [-1])],
		[tf.ones([batch_size * time_step_size], dtype=tf.float32)])

	cost = tf.reduce_sum(loss) / batch_size
	'''


