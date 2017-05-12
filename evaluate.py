import tensorflow as tf
import numpy as np
from prepro import word2index
import datetime


def run_epoch(config, model, wordic, step, data):

	sess = model.session
	batch_size = config.batch_size

	input_X = []
	input_Y = []
	input_len = []
	for contents in data:
		input_X.append(contents['context'])
		input_Y.append(contents['sentiment'])
		input_len.append(contents['seqlen'])
	
	data_idx = 0
	acc_sum = 0
	acc = 0
	print("batch size : {}\n{} iterations per epoch".format(batch_size, len(input_X)//batch_size))
	for iteration in range(len(input_X)//batch_size):
		batch_x = []
		batch_y = []
		batch_x_len = []
		for _ in range(batch_size):
			batch_x.append(input_X[data_idx])
			batch_y.append(input_Y[data_idx])
			batch_x_len.append(input_len[data_idx])
			data_idx = (data_idx + 1) % ((len(input_X)//batch_size) * batch_size)
		batch_x, batch_y = word2index(batch_x, batch_y, wordic)	
		
		if step == "train":
			keep_prob = config.keep_prob
		else:
			keep_prob = 1.0
		feed_dict = {
					model.x : batch_x,
					model.y : batch_y,
					model.seqlen : batch_x_len,
					model.keep_prob : keep_prob
			}
		
		if step == 'train':
			_, global_step, loss, accuracy, lr = sess.run(
										[model.train, model.global_step, model.cost, model.accuracy, model.lr], feed_dict=feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}, lr {:g}".format(time_str, global_step, loss, accuracy, lr))
			#if (iteration+1) != 1 and (iteration+1) % 5 == 0:
			#	loss, accuracy = sess.run([model.cost, model.accuracy], feed_dict=feed_dict)
			#	print(step + "#Iter : %d, Loss : %.3f, Accuracy : %.3f" % (iteration+1, loss, accuracy))
		else:
			global_step, loss, accuracy, lr = sess.run(
													[model.global_step, model.cost, model.accuracy, model.lr], feed_dict=feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}, lr {:g}".format(time_str, global_step, loss, accuracy, lr))
			
			acc_sum += accuracy
		
			if iteration + 1 >= len(input_X)//batch_size:
				acc = acc_sum / iteration
				print("valid acc : {}".format(acc))
				
				return acc



