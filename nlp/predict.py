import os
import tflearn
import numpy as np
from dataset_builder import DatasetBuilder
import time

"""
builder = DatasetBuilder()
learning_rate = 0.001
labels = builder.get_labels()
net = tflearn.input_data([None, 1, 8192])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, len(labels), activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('tflearn.lstm.model')

start = time.time() # Check time taken of 2 predictions

file = "00b01445_nohash_0.wav"
chunk = builder.load_wav_file("{}/{}".format(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_speech/eight"), file))
test = np.reshape([chunk], [-1, 1, 8192])
label = builder.get_labels()[model.predict_label(test)[0][0]]
predicts = model.predict(test)[0]
accuracy = predicts[np.argmax(predicts)]
label = builder.get_labels()[np.argmax(predicts)]
print "I am {}% sure the word passed is {}".format(accuracy*100, label)

######## Print Another To Check Prediction Latency ############
######## Checking Against Train Data To Validate Overfit #########

file = "ten02.wav"
chunk = builder.load_wav_file("{}/{}".format(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Corpora/Numbers/Ten"), file))
test = np.reshape([chunk], [-1, 1, 8192])
label = builder.get_labels()[model.predict_label(test)[0][0]]
predicts = model.predict(test)[0]
accuracy = predicts[np.argmax(predicts)]
label = builder.get_labels()[np.argmax(predicts)]
print "I am {}% sure the word passed is {}".format(accuracy*100, label)

end = time.time()

print "2 predictions took {} seconds".format(end - start)
"""

filename = "yellow_1.wav"
path_to_file = "{}/{}".format(os.path.dirname(os.path.realpath(__file__)), filename)

from classifier import Classifier

c = Classifier()

c.predict(path_to_file)