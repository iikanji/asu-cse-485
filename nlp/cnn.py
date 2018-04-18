import tensorflow as tf
import os
import helpers
import librosa
from random import shuffle
import numpy as np

labels=[
'zero',
'one',
'two',
'three',
'four',
'five',
'six',
'seven',
'eight',
'nine'
]

n_classes = len(labels)
BATCH_SIZE = 64
LEARNING_RATES = [0.001, 0.0001]
HEIGHT = 20 # Num of features
WIDTH = 100
TRAINING_ITERS = [10000, 3000]
eval_every = 15
LOGDIR = "/tmp/tboard_logs/"
VAL_LOGDIR = "/tmp/tboard_val_logs/"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_size(using_set, labels=[]):
	speech_cmd_zip = "{}/data_speech.tar.gz".format(os.path.dirname(os.path.realpath(__file__)))
	speech_data_dir = "{}/data_speech/".format(os.path.dirname(os.path.realpath(__file__)))
	if not os.path.isdir(speech_data_dir):
		helpers.unzip_data_tar_gz(speech_cmd_zip, speech_data_dir)
	labels = [x for x in os.listdir(speech_data_dir) if os.path.isdir(os.path.join(speech_data_dir, x)) and x != "_background_noise_" and ((len(labels) > 0 and x in labels) or len(labels) == 0)]
	if using_set == 'validation':
		return len([x for x in open(os.path.join(speech_data_dir, "validation_list.txt")).read().splitlines() if x.split("/")[0] in labels])
	elif using_set == 'testing':
		return len([x for x in open(os.path.join(speech_data_dir, "testing_list.txt")).read().splitlines() if x.split("/")[0] in labels])

def get_batch_data(BATCH_SIZE=64, labels=[], using_set='training'):
	speech_cmd_zip = "{}/data_speech.tar.gz".format(os.path.dirname(os.path.realpath(__file__)))
	speech_data_dir = "{}/data_speech/".format(os.path.dirname(os.path.realpath(__file__)))
	if not os.path.isdir(speech_data_dir):
		helpers.unzip_data_tar_gz(speech_cmd_zip, speech_data_dir)
	labels = [x for x in os.listdir(speech_data_dir) if os.path.isdir(os.path.join(speech_data_dir, x)) and x != "_background_noise_" and ((len(labels) > 0 and x in labels) or len(labels) == 0)]
	filenames = []
	if using_set == 'testing':
		filenames = [os.path.join(speech_data_dir, x) for x in open(os.path.join(speech_data_dir, "testing_list.txt")).read().splitlines() if x.split("/")[0] in labels]
	elif using_set == 'validation':
		filenames = [os.path.join(speech_data_dir, x) for x in open(os.path.join(speech_data_dir, "validation_list.txt")).read().splitlines() if x.split("/")[0] in labels]
	else:
		for category in labels:
			filenames.extend([os.path.join(os.path.join(speech_data_dir, category), x) for x in os.listdir(os.path.join(speech_data_dir, category)) if x.endswith(".wav")])
	
	batch_items = []
	batch_labels = []
	shuffle(filenames)
	for file in filenames:
		if using_set == 'training':
			which_set = helpers.which_set(file, 10, 10)
			if which_set == 'training':
				label = file.split("/")[-2]
				wave, sr = librosa.load(file, mono=True)
				mfcc = librosa.feature.mfcc(wave, sr)
				mfcc = np.pad(mfcc,((0,0),(0,100-len(mfcc[0]))), mode='constant', constant_values=0)
				lbl = [1 if label == l else 0 for l in labels]
				batch_items.append(mfcc)
				batch_labels.append(lbl)
		else:
			label = file.split("/")[-2]
			wave, sr = librosa.load(file, mono=True)
			mfcc = librosa.feature.mfcc(wave, sr)
			mfcc = np.pad(mfcc,((0,0),(0,100-len(mfcc[0]))), mode='constant', constant_values=0)
			lbl = [1 if label == l else 0 for l in labels]
			batch_items.append(mfcc)
			batch_labels.append(lbl)
		
		if len(batch_items) == BATCH_SIZE:
			yield  batch_items, batch_labels
			batch_items = []
			batch_labels = []

def get_model(data, dropout):
	input_4d = tf.reshape(data, [-1, HEIGHT, WIDTH, 1])
	
	first_filter_width = 8
	first_filter_height = 20
	first_filter_count = 64
	w1 = tf.Variable(tf.truncated_normal([first_filter_height, first_filter_width, 1, first_filter_count], stddev=0.01))
	b1 = tf.Variable(tf.zeros([first_filter_count]))
	conv1 = tf.nn.conv2d(input_4d, w1, strides=[1, 1, 1, 1], padding='SAME')
	act1 = tf.nn.relu(conv1 + b1)
	dropout1 = tf.nn.dropout(act1, dropout)
	max_pool = tf.nn.max_pool(dropout1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	second_filter_width = 4
	second_filter_height = 10
	second_filter_count = 64
	w2 = tf.Variable(tf.truncated_normal([second_filter_height, second_filter_width, first_filter_count, second_filter_count], stddev=0.01))
	b2 = tf.Variable(tf.zeros([second_filter_count]))
	conv2 = tf.nn.conv2d(max_pool, w2, strides=[1, 1, 1, 1], padding='SAME')
	act2 = tf.nn.relu(conv2 + b2)
	dropout2 = tf.nn.dropout(act2, dropout)

	conv2shape = dropout2.get_shape()
	fc_count = int(conv2shape[1] * conv2shape[2] * conv2shape[3])
	flat_output = tf.reshape(dropout2, [-1, fc_count])

	w3 = tf.Variable(tf.truncated_normal([fc_count, n_classes], stddev=0.01))
	b3 = tf.Variable(tf.zeros([n_classes]))
	fc = tf.add(tf.matmul(flat_output, w3), b3)

	return fc

def train():
	tf.reset_default_graph()
	with tf.Session() as sess:

		x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH], name="x")
		y = tf.placeholder(tf.float32, shape=[None, n_classes], name="y")
		dropout = tf.placeholder(tf.float32, name='dropout')
		learning_rate = tf.placeholder(tf.float32, name='learning_rate_input')

		model = get_model(x, dropout)

		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
			tf.summary.scalar('loss', loss)

		with tf.name_scope('train'):
			train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

		with tf.name_scope('accuracy'):
			predicted = tf.argmax(model, 1)
			truth = tf.argmax(y, 1)
			correct_prediction = tf.equal(predicted, truth)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			confusion_matrix = tf.confusion_matrix(truth, predicted, num_classes=n_classes)
			tf.summary.scalar("accuracy", accuracy)

		# Tensorboard
		summ = tf.summary.merge_all()
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(LOGDIR+"/train_10000", sess.graph)
		validation_writer = tf.summary.FileWriter(VAL_LOGDIR+"/validate_10000")

		# Training Model
		print "Starting Training"
		for training_iter, LEARNING_RATE in zip(TRAINING_ITERS, LEARNING_RATES):
			batch = get_batch_data(labels = labels)
			val_i = 0
			for i in xrange(1, training_iter + 1):
				try:
					X, Y = next(batch)
				except StopIteration:
					batch = get_batch_data(labels=labels)
					X, Y = next(batch)
				train_summ, train_acc, train_loss, _ = sess.run([summ, accuracy, loss, train_step], feed_dict={x: X, y: Y, learning_rate: LEARNING_RATE, dropout:0.5})
				tf.logging.info('Step #{:d}: Loss {:f}, Accuracy {:.3f}%'.format(i, train_acc, train_loss))
				print 'Step #{:d}: Loss {:.3f}, Accuracy {:.2f}%'.format(i, train_loss, train_acc*100)
				writer.add_summary(train_summ, i)
				if i % eval_every == 0:
					total_acc = 0
					total_conf_matrix = None
					total_loss = 0
					val_size = get_size('validation', labels=labels)
					val_batch = get_batch_data(labels=labels, using_set='validation')
					j = 0
					for val_X, val_Y in val_batch:
						val_acc, val_loss, s, conf_matrix = sess.run([accuracy, loss, summ, confusion_matrix], feed_dict={x: val_X, y: val_Y, dropout: 1.0})
						v_batch_size = min(BATCH_SIZE, val_size - j)
						total_acc += (val_acc * v_batch_size) / val_size
						total_loss += (val_loss * v_batch_size) / val_size
						if total_conf_matrix is None:
							total_conf_matrix = conf_matrix
						else:
							total_conf_matrix += conf_matrix
					tf.logging.info("Validation:\t Val Loss: {:.3f}. Val Acc: {:.2f}%. Confusion Matrix:\n {}".format(total_loss, total_acc*100, total_conf_matrix))
					print "Validation:\t Val Loss: {:.3f}. Val Acc: {:.2f}%. Confusion Matrix:\n {}".format(total_loss, total_acc*100, total_conf_matrix)
					validation_writer.add_summary(s, val_i)
					val_i += 1
					j += BATCH_SIZE
			
			saver.save(sess, os.path.join(CURRENT_DIR, "cnn_model_10000.ckpt"), i)
		
		# Testing Model
		test_size = get_size('testing', labels=labels)
		total_acc = 0
		total_conf_matrix = None
		test_batch = get_batch_data(labels=labels, using_set='testing')
		i = 0
		for test_X, test_Y in test_batch:
			testing_acc, conf_matrix = sess.run([accuracy, confusion_matrix], feed_dict={x: test_X, y: test_Y, dropout: 1.0})
			t_batch_size = min(BATCH_SIZE, test_size - i)
			total_acc += (testing_acc * t_batch_size) / test_size
			if total_conf_matrix is None:
				total_conf_matrix = conf_matrix
			else:
				total_conf_matrix += conf_matrix
			i += BATCH_SIZE

		tf.logging.info('Confusion Matrix:\n {}'.format(total_conf_matrix))
		print 'Confusion Matrix:\n {}'.format(total_conf_matrix)
		tf.logging.info('Final Test Accuracy: {:.1f}%'.format(total_acc*100))
		print 'Final Test Accuracy: {:.1f}%'.format(total_acc*100)

def predict(filename, expected_label):
	tf.reset_default_graph()
	with tf.Session() as sess:
		wave, sr = librosa.load(filename, mono=True)
		mfcc = librosa.feature.mfcc(wave, sr)
		mfcc = np.pad(mfcc,((0,0),(0,100-len(mfcc[0]))), mode='constant', constant_values=0)
		
		test_X = np.reshape([mfcc], [-1, 20, 100])
		test_Y = np.reshape([[1 if expected_label == l else 0 for l in labels]], [-1, 10])
		
		x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH], name="x")
		y = tf.placeholder(tf.float32, shape=[None, n_classes], name="y")
		dropout = tf.placeholder(tf.float32, name='dropout')

		model = get_model(x, 1.0)
		prediction = tf.nn.softmax(model)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph('./nlp/cnn_model_10000.ckpt-3000.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./nlp/'))
		pred = sess.run(prediction, feed_dict={x: test_X, y: test_Y, dropout: 1.0})
		exp_ind = labels.index(expected_label)
		print "I am {:.2f}% sure that the word said is {}".format(pred[0][exp_ind]*100, expected_label)
		return pred[0][exp_ind]