# import the necessary packages
import numpy as np

class DirectionCounter:
	def __init__(self, directionMode, H, W):
		# initialize the height and width of the input image
		self.H = H
		self.W = W

		# initialize variables holding the direction of movement,
		# along with counters for each respective movement (i.e.,
		# left-to-right and top-to-bottom)
		self.directionMode = directionMode
		self.totalUp = 0
		self.totalDown = 0
		self.totalRight = 0
		self.totalLeft = 0

		# the direction the trackable object is moving in
		self.direction = ""

	def find_direction(self, to, centroid):
		# check to see if we are tracking horizontal movements
		if self.directionMode == "horizontal":
			# the difference between the x-coordinate of the
			# *current* centroid and the mean of *previous* centroids
			# will tell us in which direction the object is moving
			# (negative for 'left' and positive for 'right')
			x = [c[0] for c in to.centroids]
			delta = centroid[0] - np.mean(x)

			# determine the sign of the delta -- if it is negative,
			# the object is moving left
			if delta < 0:
				self.direction = "left"

			# otherwise if the sign is positive, the object is moving
			# right
			elif delta > 0:
				self.direction = "right"

		# otherwise we are tracking vertical movements
		elif self.directionMode == "vertical":
			# the difference between the y-coordinate of the
			# *current* centroid and the mean of *previous* centroids
			# will tell us in which direction the object is moving
			# (negative for 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			delta = centroid[1] - np.mean(y)

			# determine the sign of the delta -- if it is negative,
			# the object is moving up
			if delta < 0:
				self.direction = "up"

			# otherwise, if the sign of the delta is positive, the
			# object is moving down
			elif delta > 0:
				self.direction = "down"

	def count_object(self, to, centroid):
		# initialize the output list
		output = []

		# check if the direction of the movement is horizontal
		if self.directionMode == "horizontal":
			# if the object is currently left of center and is
			# moving left, count the object as moving left
			leftOfCenter = centroid[0] < self.W // 2
			if self.direction == "left" and leftOfCenter:
				self.totalLeft += 1
				to.counted = True

			# otherwise, if the object is right of center and moving
			# right, count the object as moving right
			elif self.direction == "right" and not leftOfCenter:
				self.totalRight += 1
				to.counted = True

			# construct a list of tuples with the count of objects
			# that have passed in the left and right direction
			output = [("Left", self.totalLeft),
				("Right",self.totalRight)]

		# otherwise the direction of movement is vertical
		elif self.directionMode == "vertical":
			# if the the centroid is above the middle and is moving
			# up, count the object as moving up
			aboveMiddle = centroid[1] < self.H // 2
			if self.direction == "up" and aboveMiddle:
				self.totalUp += 1
				to.counted = True

			# otherwise, if the object is moving down and is below
			# the middle, count the object as moving down
			elif self.direction == "down" and not aboveMiddle:
				self.totalDown += 1
				to.counted = True

			# return a list of tuples with the count of objects that
			# have passed in the up and down direction
			output = [("Up", self.totalUp), ("Down", self.totalDown)]

		# return the output list
		return output