import tensorflow as tf
import numpy as np


def _build():
	return


def _feed():
	sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	sessConfig.gpu_options.allow_growth = True
	sv = tf.train.Supervisor(logdir=FLAGS.save_path)
	with sv.managed_session(config=sessConfig) as session:
		for i in range(config.max_max_epoch):
			_train()

		_test()
	
	return


def LSTM(config, train_X, train_Y):
	lstm = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_size, state_is_tuple=False)
	state = tf.zeros([config.batch_size, lstm.state_size])
	probabilities = []
	loss = 0.0

	attn_cell = lstm
	if is_training:
		attn_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=config.keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell([attn_cell for _ in range(config.num_lstm_layers)], state_is_tuple=False)


def CNN(config):
	with tf.Graph().as_default():
		sess_config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=True)
		sess_config.gpu_options.allow_growth = True
		sess = tf.Session(config=sess_config)

			

def basicLSTM(config, vocab_size, max_len_context, train_X, train_Y):
	
	batch_size = config.batch_size
	lstm_size = config.lstm_hidden_size
	time_step_size = max_len_context
	lr = config.lr

	x = tf.placeholder(tf.float32, [batch_size, time_step_size, vocab_size])
	y = tf.placeholder(tf.float32, [None, 2])
	
	w = tf.Variable(tf.random_normal([lstm_size, 2]))
	b = tf.Variable(tf.random_normal([2]))

	print(x)
	x = tf.convert_to_tensor(x, dtype=tf.float32)

	lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

	state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
	print(x)	
	for time_step in range(time_step_size):
		outputs, states = lstm_cell(x[:,time_step,:], state)

	pred = tf.matmul(outputs[-1], w) + b

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.initialize_all_variables()

	sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	sessConfig.gpu_options.allow_growth = True


	with tf.Session(config=sessConfig) as sess:
		sess.run(init)
		step = 1

	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape((batch_size, time_step_size, vocab_size))

		sess.run(train, feed_dict={x: batch_x, y: batch_y})
		if step % display_step == 0:
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print("step : %d, acc: %f" % ( step, acc ))
		step += 1
	print("train complete!")

	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	print("test accuracy: ", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



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


