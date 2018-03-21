import tflearn
import numpy as np
import os
import pickle
import wavio
import wave
from random import shuffle
import helpers
import librosa

class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self, learning_rate=0.001, difficulty_level=0):
		self.difficulty_level = difficulty_level
		self.labels = ["Pink", "Yellow", "Green"]
		self.learning_rate = learning_rate
		current_dir = os.path.dirname(os.path.realpath(__file__))
		num_classes = len(os.listdir("{}/Corpora".format(current_dir))) + len([x for x in os.listdir("{}/data_speech".format(current_dir)) if os.path.isdir(os.path.join("{}/data_speech".format(current_dir), x)) and x != "_background_noise_"])
		net = tflearn.input_data([None, 20, 500])
		net = tflearn.lstm(net, 1024, dropout=0.8)
		net = tflearn.fully_connected(net, 30, activation='softmax')
		net = tflearn.regression(net, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy')
		self.model = tflearn.DNN(net, tensorboard_verbose=0)
		# self.model.load('speech_commands.lstm.preprocess')
		
	def set_level(self, level):
		self.difficulty_level = level

	def train(self, training_iterations=100, replication=False):
		builder = DatasetBuilder()
		#speech_command_dataset = builder.build_speech_command_dataset()
		#trainX, trainY = speech_command_dataset.get_training_set()
		#testX, testY = speech_command_dataset.get_testing_set()
		#batch = builder.build_speech_command_dataset()
		trainX, trainY, testX, testY = builder.build_speech_command_dataset()
		net = tflearn.input_data([None, 20, 500])
		net = tflearn.lstm(net, 512, dropout=0.8)
		net = tflearn.fully_connected(net, len(trainY[0]), activation='softmax')
		net = tflearn.regression(net, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy')
		self.model = tflearn.DNN(net, tensorboard_verbose=0)

		remaining = training_iterations
		while remaining > 0:
			print "{} iterations remaining".format(remaining)
			self.model.fit(trainX, trainY, n_epoch=50, validation_set=(testX, testY), show_metric=True, batch_size=64)
			remaining -= 1

		self.model.save('speech_commands.lstm.preprocess')

	def add_to_model(self, items=[]):
		"""
		Input: items - list of label names to be added to model
		"""
		pass

	def predict(self, filename, expected_value):
		"""
		Input: filename - the temp file of the spoken word
		Output: bool - True if accuracy falls in range, False otherwise
		"""

		wave, sr = librosa.load(filename, mono=True)
		mfcc = librosa.feature.mfcc(wave, sr)
		print len(mfcc[0])
		exit()
		mfcc = np.pad(mfcc,((0,0),(0,500-len(mfcc[0]))), mode='constant', constant_values=0)
		labels = [
		"bed",
		"bird",
		"cat",
		"dog",
		"down",
		"eight",
		"five",
		"four",
		"go",
		"happy",
		"house",
		"left",
		"marvin",
		"nine",
		"no",
		"off",
		"on",
		"one",
		"right",
		"seven",
		"sheila",
		"six",
		"stop",
		"three",
		"tree",
		"two",
		"up",
		"wow",
		"yes",
		"zero"
		]
		test = np.reshape([mfcc], [-1, 20, 80])
		predicts = self.model.predict(test)[0]
		accuracy = predicts[np.argmax(predicts)]
		label = labels[np.argmax(predicts)]
		print "I am {}% sure the word passed is {}".format(accuracy*100, label)

		return False if label != expected_value else self.validate_accuracy(self.difficulty_level, accuracy)

	def validate_accuracy(self, level, accuracy):
		if level == 0:
			pass
		elif level == 1:
			pass
		elif level == 2:
			pass
		elif level == 3:
			pass
		elif level == 4:
			pass
		elif level == 5:
			pass

		return True
		

class Dataset(object):

	def __init__(self, chunk_size = 4096, has_testset=False):
		self.num_features = []
		self.labels = []
		self.wavs = []
		self.batch_size = 0
		self.CHUNK = chunk_size
		self.num_models = 0
		self.max_height = 0
		self.x = []
		self.y = []
		self.proportion = .75 if not has_testset else 0
		self.cutoff = 0
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None

	def setX_Y(self, x, y):
		self.x = x
		self.y = y
		self.cutoff = int(self.proportion * len(self.x))

	def set_labels(self, l):
		self.labels = l
		self.num_models = len(self.labels)

	def get_training_set(self):
		if self.proportion != 0:
			self.train_x = np.reshape(self.x[:self.cutoff], [-1, 20, 80])
			self.train_y = self.y[:self.cutoff]
		else:
			self.train_x = np.reshape(self.x, [-1, 20, 80])
			self.train_y = self.y
		return np.array(self.train_x), np.array(self.train_y)

	def get_testing_set(self):
		if self.proportion != 0:
			self.test_x = np.reshape(self.x[self.cutoff:], [-1, 20, 80])
			self.test_y = self.y[self.cutoff:]
		elif len(self.test_x) == 0 or len(self.test_y) == 0:
			raise Exception("Need to set test data before calling get_testing_set(). Use set_testing_data()")
		
		return np.array(self.test_x), np.array(self.test_y)

	def set_testing_data(self, x, y):
		self.test_x = np.reshape(x, [-1, 20, 80])
		self.test_y = y

class DatasetBuilder(object):

	def __init__(self, data_dir="{}/Corpora".format(os.path.dirname(os.path.realpath(__file__)))):
		self.CORPUS_DIR = data_dir

	def create_labels(self, dataset, exclusive_labels=None):
		categories = set([x for x in os.listdir(self.CORPUS_DIR) if os.path.isdir(os.path.join(self.CORPUS_DIR, x))])
		labels = []
		cats = []
		for label in [x for x in os.listdir(self.CORPUS_DIR) if os.path.isdir(os.path.join(self.CORPUS_DIR, x))]:
			if (exclusive_labels is not None and label in exclusive_labels) or exclusive_labels is None:
				labels.append(label)
		dataset.set_labels(labels)

	def build_dataset(self, labels=None):
		dataset = Dataset()
		corpora_zip = "./Corpora.zip"
		corpora_dir = "./Corpora/"
		if not os.path.isdir(corpora_dir):
			helpers.unzip_data_zip(corpora_zip, corpora_dir)
		self.create_labels(dataset, labels)

		batch_items = []
		for label in dataset.labels:
			label_dir = os.path.join(self.CORPUS_DIR, label)
			filenames = [x for x in os.listdir(label_dir) if x.endswith(".wav")]

			for file in filenames:
				wave, sr = librosa.load(os.path.join(label_dir, file), mono=True)
				# chunk = helpers.load_wav_file(os.path.join(label_dir, file), dataset.CHUNK)
				mfcc = librosa.feature.mfcc(wave, sr)
				mfcc = np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
				dataset.num_features = len(mfcc)
				lbl = [1 if label == l else 0 for l in dataset.labels]
				batch_items.append((mfcc, lbl))

		shuffle(batch_items)
		dataset.setX_Y([x[0] for x in batch_items], [x[1] for x in batch_items])

		return dataset

	def build_speech_command_dataset(self, batch_size=64):
		#dataset = Dataset(has_testset=True)
		speech_cmd_zip = "{}/data_speech.tar.gz".format(os.path.dirname(os.path.realpath(__file__)))
		speech_data_dir = "{}/data_speech/".format(os.path.dirname(os.path.realpath(__file__)))
		if not os.path.isdir(speech_data_dir):
			helpers.unzip_data_tar_gz(speech_cmd_zip, speech_data_dir)
		labels = [x for x in os.listdir(speech_data_dir) if os.path.isdir(os.path.join(speech_data_dir, x)) and x != "_background_noise_"]
		#dataset.set_labels(labels)
		filenames = []
		for category in labels:
			filenames.extend([os.path.join(os.path.join(speech_data_dir, category), x) for x in os.listdir(os.path.join(speech_data_dir, category)) if x.endswith(".wav")])
		batch_items = []
		batch_labels = []
		test_items = []
		test_labels = []
		shuffle(filenames)
		for file in filenames:
			label = file.split("/")[-2]
			wave, sr = librosa.load(file, mono=True)
			mfcc = librosa.feature.mfcc(wave, sr)
			mfcc = np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
			# dataset.num_features = len(mfcc))
			lbl = [1 if label == l else 0 for l in labels]
			which_set = helpers.which_set(file, 10, 10)
			if which_set == 'validation':
				test_items.append(mfcc)
				test_labels.append(lbl)
				#batch_items.append((mfcc, lbl))
			elif which_set == 'training':
				batch_items.append(mfcc)
				batch_labels.append(lbl)
				#batch_items.append((mfcc, lbl))

		return  batch_items, batch_labels, test_items, test_labels
		#batch_items = []
		#batch_labels = []
		#test_items = []
		#test_labels = []

		#shuffle(batch_items)
		#dataset.setX_Y([x[0] for x in batch_items], [x[1] for x in batch_items])

		#shuffle(test_items)
		#dataset.set_testing_data([x[0] for x in test_items], [x[1] for x in test_items])

		#return dataset
