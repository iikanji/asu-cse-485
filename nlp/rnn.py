import tensorflow as tf
import os
import helpers
import librosa
from random import shuffle
import numpy as np

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

epochs = 10
n_classes = len(labels)
n_features = 20
LEARNING_RATE = 0.0001
TRAINING_ITERS = 5000
utterance_length = 100
batch_size = 64
chunk_size = 10
n_chunks = 10
rnn_size = 128
LOGDIR = "/tmp/tboard_logs/"

def rnn(x):
	layer = {
		'weights' : tf.Variable(tf.random_normal([rnn_size, n_classes])),
		'biases' : tf.Variable(tf.random_normal([n_classes]))
	}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, x, n_chunks)

	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['baises']
	return output

def train():
	tf.reset_default_graph()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		x = tf.placeholder(tf.float32, [None, n_features, utterance_length], name='x')
		y = tf.placeholder(tf.float32, name='y')
		dropout = tf.placeholder(tf.float32, name='dropout')

		model = rnn(x)

		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
			tf.summary.scalar('loss', loss)

		with tf.name_scope('train'):
			learning_rate = tf.placeholder(tf.float32, name='learning_rate_input')
			train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

		with tf.name_scope('accuracy'):
			predicted = tf.argmax(model, 1)
			truth = tf.argmax(y, 1)
			correct_prediction = tf.equal(predicted, truth)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			tf.summary.scalar("accuracy", accuracy)

		# Tensorboard
		summ = tf.summary.merge_all()
		#saver = tf.train.Saver()
		writer = tf.summary.FileWriter(LOGDIR+"/train_rnn", sess.graph)
		#validation_writer = tf.summary.FileWriter(VAL_LOGDIR+"/validate_5000")

		step = 0
		for epoch in range(epochs):
			batch = get_batch_data(BATCH_SIZE=batch_size, labels=labels)
			for train_X, train_Y in batch:
				train_X = train_X.reshape([batch_size, n_chunks, chunk_size])
				_, train_loss, train_acc, s = sess.run([train_step, loss, accuracy, summ], feed_dict={x: train_X, y: train_Y, dropout: 0.8, learning_rate: LEARNING_RATE})
				writer.add_summary(s, step)
				step += 1

			print 'Step #{:d}: Loss {:.3f}, Accuracy {:.2f}%'.format(epoch, train_loss, train_acc*100)

if __name__ == '__main__':
	train()