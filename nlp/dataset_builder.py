import os
import numpy as np
import pickle
import wavio
import wave
from random import shuffle

class DatasetBuilder(object):
	def __init__(self, data_dir="{}/Corpora".format(os.path.dirname(os.path.realpath(__file__)))):
		self.CORPUS_DIR = data_dir
		self.num_models = 0
		self.batch_size = 0
		self.max_width = 0
		self.max_height = 0
		self.AVERAGE_SIZE = 163664
		self.CHUNK = 4096

	def load_wav_file(self, name):
		f = wave.open(name, "rb")
		# print("loading %s"%name)
		chunk = []
		data0 = f.readframes(self.CHUNK)
		while data0:  # f.getnframes()
			# data=numpy.fromstring(data0, dtype='float32')
			# data = numpy.fromstring(data0, dtype='uint16')
			data = np.fromstring(data0, dtype='uint8')
			data = (data + 128) / 255.  # 0-1 for Better convergence
			# chunks.append(data)
			chunk.extend(data)
			data0 = f.readframes(self.CHUNK)
		# finally trim:
		chunk = chunk[0:self.CHUNK * 2]  # should be enough for now -> cut
		chunk.extend(np.zeros(self.CHUNK * 2 - len(chunk)))  # fill with padding 0's
		# print("%s loaded"%name)
		return chunk

	def get_labels(self):
		categories = [x for x in os.listdir(self.CORPUS_DIR) if os.path.isdir(os.path.join(self.CORPUS_DIR, x))]
		all_labels = []
		for category in categories:
			all_labels.extend([x for x in os.listdir(os.path.join(self.CORPUS_DIR, category)) if os.path.isdir(os.path.join(os.path.join(self.CORPUS_DIR, category), x))])

		return all_labels

	def create_dataset(self):
		avg_len = 0
		if not os.path.exists(self.CORPUS_DIR):
			print "Cannot find Corpus folder. Exiting..."
			exit(1)
		else:
			print "Creating dataset from {} ......".format(self.CORPUS_DIR)
		
		categories = [x for x in os.listdir(self.CORPUS_DIR) if os.path.isdir(os.path.join(self.CORPUS_DIR, x))]
		batch_waves = []
		all_labels = []
		batch_labels = []
		for category in categories:
			print "Creating category {}.....".format(category)
			all_labels.extend([x for x in os.listdir(os.path.join(self.CORPUS_DIR, category)) if os.path.isdir(os.path.join(os.path.join(self.CORPUS_DIR, category), x))])
		
		self.num_models = len(all_labels)

		#self.CORPUS_DIR = "{}/data_speech".format(os.path.dirname(os.path.realpath(__file__)))

		#for category in categories:
		#labels = [x for x in os.listdir(os.path.join(self.CORPUS_DIR, category)) if os.path.isdir(os.path.join(os.path.join(self.CORPUS_DIR, category), x))]
		for label in all_labels:
			# label_dir = os.path.join(os.path.join(self.CORPUS_DIR, category), label)
			label_dir = os.path.join(self.CORPUS_DIR, label)
			filenames = [x for x in os.listdir(label_dir) if x.endswith(".wav")]
			for file in filenames:
				chunk = self.load_wav_file(os.path.join(label_dir, file))
				self.max_height = len(chunk)
				lbl = []
				lbl = [1 if label == l else 0 for l in all_labels]
				batch_waves.append(chunk)
				batch_labels.append(lbl)
				self.batch_size += 1

		yield batch_waves, batch_labels
		batch_waves = []
		batch_labels = []