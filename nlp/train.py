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

