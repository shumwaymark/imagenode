import datetime
from collections import deque

class FPS:

	def __init__(self, history=160):  
		# default allows for 5 seconds of history at 32 images/sec 
		self._deque = deque(maxlen=history) 

	def update(self):
		# capture current timestamp
		self._deque.append(datetime.datetime.now())
	
	def fps(self):
		# calculate and return estimated frames/sec
		if len(self._deque) < 2:
			return 0
		else:
			return int(len(self._deque) / 
				(self._deque[-1] - self._deque[0]).total_seconds())
