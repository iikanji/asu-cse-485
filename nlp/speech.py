import pyaudio
import wave
import threading

class Speech(object):

	"""
	Plan here is to create a Speech class which acts as a streaming bridge for I/O and NLP
	"""

	CHUNK = 1024 # Read 1 byte at a time

	def __init__(self):
		# initialize things here
		self.audio = pyaudio.PyAudio()
		self.streaming = False

	def Start(self):
		# begin streaming info. Needs to be threaded.
		self.streaming = True
		self._stream()

	def Stop(self):
		# Tear down all running processes and close the stream
		self.streaming = False

	def _stream(self):
		# Start listening for I/O
		# stream = self.audio.open(data-goes-here)
		while self.streaming:
			pass
		# stream.close()