import tflearn
import numpy as np
import os
import pickle
import wavio
import wave
from random import shuffle
import helpers

class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self, learning_rate=0.001, difficulty_level=0):
		self.difficulty_level = difficulty_level
		self.labels = []
		data_dir = "Corpora/"
		categories = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
		for category in categories:
			lbls = [x for x in os.listdir(os.path.join(data_dir, category)) if os.path.isdir(os.path.join(os.path.join(data_dir, category), x))]
			self.labels.extend(lbls)
		self.model = None
		self.learning_rate = learning_rate
		

	def set_level(self, level):
		self.difficulty_level = level

	def train(self, training_iterations=100, replication=False):
		if not os.path.isfile("speech_commands.lstm.preprocess"):
			# Build preprocess model

			builder = DatasetBuilder()
			speech_command_dataset = builder.build_speech_command_dataset()
			trainX, trainY = speech_command_dataset.get_training_set()
			testX, testY = speech_command_dataset.get_testing_set()
			print len(trainX)
			print len(testX)

			net = tflearn.input_data([None, 1, 8192])
			net = tflearn.lstm(net, 128, dropout=0.8)
			net = tflearn.fully_connected(net, speech_command_dataset.num_models, activation='softmax')
			net = tflearn.regression(net, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy')
		
			model = tflearn.DNN(net, tensorboard_verbose=0)
			remaining = training_iterations
			while remaining > 0:
				print "{} iterations remaining".format(training_iterations)
				model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=len(trainX), validation_batch_size=len(testX))
				training_iterations -= 1

			self.model.save('speech_commands.lstm.preprocess')

		"""
		dataset = builder.build_dataset()
		self.labels = dataset.labels
		trainX, trainY = dataset.get_training_set()
		testX, testY = dataset.get_testing_set()
		remaining = training_iterations
		while remaining > 0:
			print "{} iterations remaining".format(training_iterations)
			self.model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=dataset.batch_size)
			training_iterations -= 1

		self.model.save('speech_commands.lstm.postprocess')
		"""

	def add_to_model(self, items=[]):
		"""
		Input: items - list of label names to be added to model
		"""
		pass

	def predict(self, filename):
		"""
		Input: filename - the temp file of the spoken word
		Output: bool - True if accuracy falls in range, False otherwise
		"""

		if not self.model:
			net = tflearn.input_data([None, 1, 8192])
			net = tflearn.lstm(net, 128, dropout=0.8)
			net = tflearn.fully_connected(net, len(self.labels), activation='softmax')
			net = tflearn.regression(net, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy')
			self.model = tflearn.DNN(net, tensorboard_verbose=0)
		
		self.model.load('tflearn.lstm.model.large')
		chunk = helpers.load_wav_file(filename, 4096)
		test = np.reshape([chunk], [-1, 1, 8192])
		predicts = self.model.predict(test)[0]
		accuracy = predicts[np.argmax(predicts)]
		label = self.labels[np.argmax(predicts)]
		print "I am {}% sure the word passed is {}".format(accuracy*100, label)

		return self.validate_accuracy(self.difficulty_level)

	def validate_accuracy(self, level):
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
		self.categories = []
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
			self.train_x = np.reshape(self.x[:self.cutoff], [-1, 1, self.max_height])
			self.train_y = self.y[:self.cutoff]
		else:
			self.train_x = np.reshape(self.x, [-1, 1, self.max_height])
			self.train_y = self.y
		return self.train_x, self.test_y

	def get_testing_set(self):
		if self.proportion != 0:
			self.test_x = np.reshape(self.x[self.cutoff:], [-1, 1, self.max_height])
			self.test_y = self.y[self.cutoff:]
		elif len(self.test_x) == 0 or len(self.test_y) == 0:
			raise Exception("Need to set test data before calling get_testing_set(). Use set_testing_data()")
		
		return self.test_x, self.test_y

	def set_testing_data(self, x, y):
		self.test_x = np.reshape(x, [-1, 1, self.max_height])
		self.test_y = y

class DatasetBuilder(object):

	def __init__(self, data_dir="{}/Corpora".format(os.path.dirname(os.path.realpath(__file__)))):
		self.CORPUS_DIR = data_dir

	def create_labels(self, dataset):
		categories = [x for x in os.listdir(self.CORPUS_DIR) if os.path.isdir(os.path.join(self.CORPUS_DIR, x))]
		labels = []
		cats = []
		if not os.path.isfile("categories.txt"):
			with open("categories.txt", "w") as f:
				for category in categories:
					labels = [x for x in os.listdir(os.path.join(self.CORPUS_DIR, category)) if os.path.isdir(os.path.join(os.path.join(self.CORPUS_DIR, category), x))]
					labels.extend(labels)
					for label in labels:
						cat_lbl = "{}/{}".format(category, label)
						cats.append(cat_lbl)
						f.write(cat_lbl+"\n")

			f.close()
		else:
			categories = [x for x in open("categories.txt")]
			for category in categories:
				category.replace("\n","")
				labels = [x for x in os.listdir(os.path.join(self.CORPUS_DIR, category)) if os.path.isdir(os.path.join(os.path.join(self.CORPUS_DIR, category), x))]
				labels.extend(labels)
				for label in labels:
					cat_lbl = "{}/{}".format(category, label)
					cats.append(cat_lbl)

		dataset.categories = cats
		dataset.set_labels(labels)

	def build_dataset(self, replication):
		dataset = Dataset()
		self.create_labels(dataset)

		batch_waves = []
		batch_labels = []
		batch_items = []
		batch_size = 0
		for category in dataset.categories:
			label_dir = os.path.join(self.CORPUS_DIR, category)
			label = category.split("/")[1]
			filenames = [x for x in os.listdir(label_dir) if x.endswith(".wav")]

			for file in filenames:
				reps = 50 if replication else 1
				chunk = helpers.load_wav_file(os.path.join(label_dir, file), dataset.CHUNK)
				dataset.max_height = len(chunk)
				for i in range(reps):
					lbl = [1 if label == l else 0 for l in dataset.labels]
					batch_waves.append(chunk)
					batch_labels.append(lbl)
					batch_items.append((chunk, lbl))
					batch_size += 1

		shuffle(batch_items)
		dataset.setX_Y([x[0] for x in batch_items], [x[1] for x in batch_items])
		dataset.batch_size = batch_size

		return dataset

	def build_speech_command_dataset(self):
		dataset = Dataset(has_testset=True)
		speech_cmd_zip = "{}/data_speech.tar.gz".format(os.path.dirname(os.path.realpath(__file__)))
		speech_data_dir = "{}/data_speech/".format(os.path.dirname(os.path.realpath(__file__)))
		if not os.path.isdir(speech_data_dir):
			helpers.unzip_data_tar_gz(speech_cmd_zip, speech_data_dir)
		labels = [x for x in os.listdir(speech_data_dir) if os.path.isdir(os.path.join(speech_data_dir, x)) and x != "_background_noise_"]
		dataset.set_labels(labels)
		filenames = []
		for category in labels:
			filenames.extend([os.path.join(os.path.join(speech_data_dir, category), x) for x in os.listdir(os.path.join(speech_data_dir, category)) if x.endswith(".wav")])
		# filenames = [x for x in open(os.path.join(speech_data_dir, "validation_list.txt"))]
		batch_items = []
		batch_size = 0
		test_items = []
		for file in filenames:
			label = file.split("/")[-2]
			chunk = helpers.load_wav_file(file, dataset.CHUNK)
			dataset.max_height = len(chunk)
			lbl = [1 if label == l else 0 for l in dataset.labels]
			which_set = helpers.which_set(file, 10, 10)
			if which_set == 'validation':
				test_items.append((chunk, lbl))
			elif which_set == 'training':
				batch_items.append((chunk, lbl))
				batch_size += 1

		shuffle(batch_items)
		dataset.setX_Y([x[0] for x in batch_items], [x[1] for x in batch_items])
		dataset.batch_size = batch_size

		shuffle(test_items)
		dataset.set_testing_data([x[0] for x in test_items], [x[1] for x in test_items])

		return dataset
