import queue

class Memory():
	def __init__(self):
		self.short_term = queue.Queue(maxsize=15)
		self.long_term = []

	def commit_short_term(self, json):
		if self.short_term.full():
			self.short_term.get()
		self.short_term.put(json)

	def get_short_term(self):
		return list(self.short_term.queue)
