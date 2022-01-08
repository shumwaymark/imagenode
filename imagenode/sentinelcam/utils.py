from datetime import datetime
from collections import deque

class FPS:

	def __init__(self, history=160) -> None:  
		# default allows for 5 seconds of history at 32 images/sec 
		self._deque = deque(maxlen=history) 

	def lastStamp(self) -> datetime:
		# return most recent timestamp, if any
		return(self._deque[-1]) if len(self._deque) > 0 else None

	def update(self) -> datetime:
		# capture current timestamp, and return it
		self._deque.append(datetime.utcnow())
		return self.lastStamp()
	
	def fps(self) -> float:
		# calculate and return estimated frames/sec
		if len(self._deque) < 2:
			return 0.0
		else:
			return len(self._deque) / (self._deque[-1] - self._deque[0]).total_seconds()
