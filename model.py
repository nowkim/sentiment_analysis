import tensorflow as tf
import numpy as np
from prepro import one_hot, word2index
import sys
from tensorflow.contrib.tensorboard.plugins import projector



			

def basicLSTM(config, wordic, max_len_context, train_x_len, test_x_len, train_data, test_data):
	
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
	seqlen = tf.placeholder(tf.int32, [None])
	
	w = tf.Variable(tf.random_normal([lstm_size, 2]))
	b = tf.Variable(tf.random_normal([2]))

	# embedding
	embeddings = tf.Variable(
			tf.random_uniform([vocab_size, config.embedding_size], -1.0, 1.0), name="word_embedding")
	embed = tf.nn.embedding_lookup(embeddings, x)
	embed = tf.unstack(embed, max_len_context, 1)

	# embedding visualizaion
	embed_config = projector.ProjectorConfig()
	embedding = embed_config.embeddings.add()
	embedding.tensor_name = embeddings.name
	embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
	summary_writer = tf.summary.FileWriter(LOG_DIR)
	projector.visualize_embeddings(summary_writer, embed_config)


	lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, embed, sequence_length=seqlen, dtype=tf.float32)
	
	outputs = tf.stack(outputs)  
		# outputs : shape=(max_len_context, batch_size, lstm_size)
	outputs = tf.transpose(outputs, [1,0,2])  
		# outputs : shape=(batch_size, max_len_context, lstm_size)

	tmp = tf.shape(outputs)[0]

	index = tf.range(0, tmp) * max_len_context + (seqlen - 1) 
		# index : shape=(batch_size)
	outputs = tf.gather(tf.reshape(outputs, [-1, lstm_size]), index)
		# outputs : shape=(batch_size, lstm_size)
	pred = tf.matmul(outputs, w) + b  
		# pred : shape=(batch_size, 2)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
	tf.summary.scalar("cost_summary", cost)
	train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	tf.summary.scalar("learning rate", lr)

	merged = tf.summary.merge_all()

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	print(tf.trainable_variables)
	init = tf.global_variables_initializer()

	sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	sessConfig.gpu_options.allow_growth = True

	with tf.Session(config=sessConfig) as sess:
		sess.run(init)
		data_idx = 0
		print("batch size : {}\n{} iterations".format(batch_size, len(train_X)//batch_size))
		for iteration in range(len(train_X)//batch_size):
			batch_x = []
			batch_y = []
			batch_x_len = []
			for _ in range(batch_size):
				batch_x.append(train_X[data_idx])
				batch_y.append(train_Y[data_idx])
				batch_x_len.append(train_x_len[data_idx])
				data_idx = (data_idx + 1) % ((len(train_X)//batch_size) * batch_size)
			batch_x, batch_y = word2index(batch_x, batch_y, wordic)
			sess.run(train, feed_dict={x: batch_x, y: batch_y, seqlen: batch_x_len})
			if (iteration+1) != 1 and (iteration+1) % 5 == 0:
				summary, loss, A,B,C = sess.run([merged, cost, index, outputs, pred], 
						feed_dict={x: batch_x, y: batch_y, seqlen: batch_x_len})
				print("{} iterations, loss : {}".format(iteration+1, loss))
				print('index : ', A)
				print('outputs : ', B)
				print('pred : ', C)
		print("train complete!")

		test_acc = 0
		data_idx = 0
		for iteration in range(len(test_X)//batch_size):
			tbatch_x = []
			tbatch_y = []
			tbatch_x_len = []
			for data_idx in range(batch_size):
				tbatch_x.append(test_X[data_idx])
				tbatch_y.append(test_Y[data_idx])
				tbatch_x_len.append(test_x_len[data_idx])
				data_idx = data_idx + 1
			tbatch_x, tbatch_y = word2index(tbatch_x, tbatch_y, wordic)
			test_acc += accuracy.eval(
					feed_dict={x: tbatch_x, y: tbatch_y, seqlen: tbatch_x_len})
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


