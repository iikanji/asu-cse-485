import pyaudio
import wave
import threading
from sys import byteorder
from array import array
from struct import pack
from classifier import Classifier
import os
import time

class Speech(object):

	"""
	Plan here is to create a Speech class which acts as a streaming bridge for I/O and NLP
	"""

	def __init__(self, difficulty):
		self.audio = pyaudio.PyAudio()
		self.THRESHOLD = 500
		self.CHUNK_SIZE = 2048
		self.FORMAT = pyaudio.paInt16
		self.RATE = 44100
		self.classifier = Classifier(difficulty_level=difficulty)
		self.output_path = "{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "tmp_outputfile.wav")

	def increase_difficulty(self, difficulty):
		self.classifier.set_level(difficulty)

	def record_and_validate(self, expected_word):
		self.record()
		start = time.time()
		result = self.classifier.predict(self.output_path, expected_word)
		os.remove(self.output_path)
		end = time.time()
		print "Prediction took {} seconds".format(end - start)
		return result

	def _is_silent(self, data):
		"""
		Returns Bool - True if sound is below the silence threshold
		"""
		return max(data) < self.THRESHOLD

	def _normalize(self, data):
		"""Average the volume out"""
		MAXIMUM = 16384
		times = float(MAXIMUM)/max(abs(i) for i in data)

		r = array('h')
		for i in data:
			r.append(int(i*times))
		return r

	def trim(self, data):
		"""Trim the blank spots at the start and end"""
		def _trim(snd_data):
			snd_started = False
			r = []

			for i in snd_data:
				if not snd_started and abs(i)>self.THRESHOLD:
					snd_started = True
					r.append(i)

				elif snd_started:
					r.append(i)
			return r

		# Trim to the left
		snd_data = _trim(data)

		# Trim to the right
		snd_data.reverse()
		snd_data = _trim(snd_data)
		snd_data.reverse()
		return snd_data

	def _add_silence(self, data, seconds):
		"""Add silence to the start and end of 'data' of length 'seconds' (float)"""
		r = [0 for i in xrange(int(seconds*self.RATE))]
		r.extend(data)
		r.extend([0 for i in xrange(int(seconds*self.RATE))])
		return r

	def record(self):
		"""
		Record a word or words from the microphone and 
		return the data as an array of signed shorts.

		Normalizes the audio, trims silence from the 
		start and end, and pads with 0.5 seconds of 
		blank sound.

		Saves to file.
		"""
		p = pyaudio.PyAudio()
		stream = p.open(format=self.FORMAT, channels=1, rate=self.RATE, 
			input=True, frames_per_buffer=self.CHUNK_SIZE)

		num_silent = 0
		snd_started = False

		r = []

		while 1:
			# little endian, signed short
			snd_data = array('h', stream.read(self.CHUNK_SIZE))
			if byteorder == 'big':
				snd_data.byteswap()
			silent = self._is_silent(snd_data)

			if silent and snd_started:
				num_silent += 1
			elif not snd_started:
				print "Started Recording"
				snd_started = True
				r.extend(snd_data)
			else:
				r.extend(snd_data)

			if snd_started and num_silent > 50:
				print "Done Recording"
				break

		sample_width = p.get_sample_size(self.FORMAT)
		stream.stop_stream()
		stream.close()
		p.terminate()

		r = self._normalize(r)
		r = self.trim(r)
		r = self._add_silence(r, 0.5)
		data = pack('<' + ('h'*len(r)), *r)

		wf = wave.open(self.output_path, 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(sample_width)
		wf.setframerate(self.RATE)
		wf.writeframes(data)
		wf.close()

if __name__ == "__main__":
	s = Speech(0)
	print s.record_and_validate("dog")
