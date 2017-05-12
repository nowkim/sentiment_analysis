import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.tensorboard.plugins import projector
import os
import time
import datetime


class senti_anal_model(object):
	
	def __init__(self, config, seqlen, wordic):
		sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
		sessConfig.gpu_options.allow_growth = True
		self.session = tf.Session(config=sessConfig)
		self.batch_size = config.batch_size
		self.lstm_size = config.lstm_hidden_size
		self.wordic = wordic
		self.vocab_size = len(wordic)
		self.num_lstm_layer = config.num_lstm_layer
		self.pre_trained = config.pre_trained
		self.time_step_size = seqlen
		self.sequence_length = seqlen
		self.embedding_size = config.embedding_size
		self.lr = tf.Variable(0.0, trainable = False)
		self.max_grad_norm = config.max_grad_norm

		# cnn
		self.filter_sizes = config.filter_sizes
		self.num_filters = config.num_filters

		self.lstm()
		self.session.run(tf.global_variables_initializer())

	def lstm(self):
		now = time.localtime()
		print("== lstm layer == build start at {}:{}:{}".format(now.tm_hour, now.tm_min, now.tm_sec))
		self.x = tf.placeholder(tf.int32, [None, self.time_step_size])
		self.y = tf.placeholder(tf.int32, [None, 2])
		self.seqlen = tf.placeholder(tf.int32, [None])
		self.keep_prob = tf.placeholder(tf.float32)
	
		w = tf.Variable(tf.random_normal([self.lstm_size, 2]))
		b = tf.Variable(tf.constant(0.0, shape=[2]))
		
		
		# embedding
		if not self.pre_trained:
			embeddings = tf.Variable(
							tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="word_embedding")
		else:
			glove = {}
			embedding_table = []
			with open('./glove/glove.6B.'+str(self.embedding_size)+'d.txt', 'r', encoding='utf-8', errors='ignore') as f:
				while True:
					line = f.readline()
					if not line: break
					line = line.split()
					for idx in range(len(line[1:])):
						line[idx+1] = float(line[idx+1])
					glove[line[0]] = line[1:]

			for _, word in enumerate(self.wordic):
				if not word in glove.keys():
					embedding_table.append([0.0 for _ in range(self.embedding_size)])
				else:
					embedding_table.append(glove[word])

			embeddings = tf.Variable(embedding_table, trainable=False, name='word_embedding')
		
		embed = tf.nn.embedding_lookup(embeddings, self.x)
				# embed : shape = (batch_size, time_step_size, embedding_size)
		embed = tf.unstack(embed, self.time_step_size, 1)
				# embed : shape = (batch_size, embedding_size) * time_step_size
		
		# embedding visualizaion
		embed_config = projector.ProjectorConfig()
		embedding = embed_config.embeddings.add()
		embedding.tensor_name = embeddings.name
		embedding.metadata_path = os.path.join('/tmp', 'metadata.tsv')
		summary_writer = tf.summary.FileWriter('/tmp')
		projector.visualize_embeddings(summary_writer, embed_config)


		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

		lstm_cell_with_drop = tf.contrib.rnn.DropoutWrapper(
															lstm_cell, output_keep_prob=self.keep_prob)
		multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(
															[lstm_cell_with_drop for _ in range(self.num_lstm_layer)])
	
		outputs, states = tf.contrib.rnn.static_rnn(multi_lstm_cell, embed, sequence_length=self.seqlen, dtype=tf.float32)
	
		outputs = tf.stack(outputs)  
			# outputs : shape=(max_len_context, batch_size, lstm_size)
		outputs = tf.transpose(outputs, [1,0,2])  
			# outputs : shape=(batch_size, max_len_context, lstm_size)

		tmp = tf.shape(outputs)[0]

		index = tf.range(0, tmp) * self.time_step_size + (self.seqlen - 1) 
			# index : shape=(batch_size)
		outputs = tf.gather(tf.reshape(outputs, [-1, self.lstm_size]), index)
			# outputs : shape=(batch_size, lstm_size)
		pred = tf.matmul(outputs, w) + b  
			# pred : shape=(batch_size, 2)
		
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=pred))
		tf.summary.scalar("cost_summary", self.cost)
		
		correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	

		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)


		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train = optimizer.apply_gradients(
											zip(grads, tvars),
											global_step = self.global_step)
		self.new_lr = tf.placeholder(tf.float32, shape=[])
		
		self.lr_update = tf.assign(self.lr, self.new_lr)

		tf.summary.scalar("learning rate", self.lr)

		self.merged = tf.summary.merge_all()


		end = time.localtime()
		build_time = end.tm_hour*2400 + end.tm_min*60 + end.tm_sec - (now.tm_hour*2400 + now.tm_min*60 + now.tm_sec)
		print("== lstm build time : {}".format(build_time))


	def assign_lr(self, lr_value):
		sess = self.session
		sess.run(self.lr_update, feed_dict={self.new_lr : lr_value})


	def cnn(self):
		self.x = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.y = tf.placeholder(tf.int32, [None, 2])
	
		embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
		embedded = tf.nn.embedding_lookup(embedding_table, self.x)
		embeddings = tf.expand_dims(embedded, -1)
		print("embeddings", embeddings)

		pooled_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.name_scope("conv-maxpool-{}".format(filter_size)):
				filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
				W = tf.Variable(tf.random_normal(filter_shape), name="W")
				B = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="B")
				conv = tf.nn.conv2d(
								embeddings, W, strides=[1,1,1,1], padding='VALID', name='conv')
				print("conv", conv)
				h = tf.nn.relu(tf.nn.bias_add(conv, B), name='relu')
				print("h", h)
				pooled = tf.nn.max_pool(
									h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
									strides=[1,1,1,1], padding='VALID', name='pool')
				pooled_outputs.append(pooled)
				print("pooled_outputs", pooled_outputs)

		num_filters_total = self.num_filters * len(self.filter_sizes)
		self.h_pool = tf.concat(pooled_outputs, 3)
		print(self.h_pool)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
		print(self.h_pool_flat)

		self.keep_prob = tf.placeholder(tf.float32)

		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

		
		with tf.name_scope("output"):
			W = tf.get_variable("W", shape=[num_filters_total, 2],
													initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			print(self.scores)
			self.predictions = tf.argmax(self.scores, 1, name="predictions")
			print(self.predictions)

		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
			self.cost = tf.reduce_mean(losses)

		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(self.lr)
		grads_and_vars = optimizer.compute_gradients(self.cost)
		self.train = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

		


