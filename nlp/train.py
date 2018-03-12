"""
import tflearn
import numpy as np
from dataset_builder import DatasetBuilder
import os

learning_rate = 0.001
training_iterations = 100

builder = DatasetBuilder()
batch = word_batch = builder.create_dataset()
X, Y = next(batch)
net = tflearn.input_data([None, 1, builder.max_height])
X = np.reshape(X, [-1, 1, builder.max_height])
trainX, trainY = X, Y
testX, testY = X, Y
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, builder.num_models, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

while training_iterations > 0:
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=builder.batch_size)
	training_iterations -= 1

model.save('tflearn.lstm.model.large')

print "Done!"
"""

from classifier import DatasetBuilder
import tflearn
import numpy as np
import os
from classifier import Classifier

"""
builder = DatasetBuilder()
dataset = builder.build_dataset(replication=True)
trainX, trainY = dataset.get_training_set()
testX, testY = dataset.get_testing_set()

learning_rate = 0.001
training_iterations=100
labels = [x.split("/")[1] for x in open("categories.txt")] if os.path.isfile("categories.txt") else []
net = tflearn.input_data([None, 1, 8192])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, dataset.num_models, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
while training_iterations > 0:
	print "{} iterations remaining".format(training_iterations)
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=dataset.batch_size)
	training_iterations -= 1

model.save('tflearn.lstm.model.large')

"""

"""
builder = DatasetBuilder()
speech_command_dataset = builder.build_speech_command_dataset()
print speech_command_dataset.batch_size
print speech_command_dataset.num_models
trainX, trainY = speech_command_dataset.get_training_set()
testX, testY = speech_command_dataset.get_testing_set()

learning_rate = 0.001
training_iterations=1800
net = tflearn.input_data([None, 1, speech_command_dataset.max_height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, speech_command_dataset.num_models, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
while training_iterations > 0:
	print "{} iterations remaining".format(training_iterations)
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=speech_command_dataset.batch_size)
	training_iterations -= 1

model.save('speech_commands.lstm.preprocess')

dataset = builder.build_dataset()
trainX, trainY = dataset.get_training_set()
testX, testY = dataset.get_testing_set()
training_iterations = 100
while training_iterations > 0:
	print "{} iterations remaining".format(training_iterations)
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=dataset.batch_size)
	training_iterations -= 1
"""

c = Classifier(learning_rate=0.0001)
c.train(training_iterations=50)