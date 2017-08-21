# Pedagogical Value-Aligned Crowdsourcing
# @Tony Runzhe Yang

from utils import *


def strong(H, belief, examples):
	ll = lambda hx, y: 1.0 / (
			1.0 + np.exp(-3 * (1 - 2 * np.abs(hx - y))))
	frac = [1.0] * len(H)
	nume = 0
	for i in xrange(len(H)):
		for eg in examples:
			x, y = eg
			frac[i] *= ll(H[i](x), y)
		nume += frac[i] * belief[i]
	return np.array(frac) * belief / nume


def median(H, belief, examples):
	ll = lambda hx, y: 1.0 / (
			1.0 + np.exp(-1.2 * (1 - 2 * np.abs(hx - y))))
	frac = [1.0] * len(H)
	nume = 0
	for i in xrange(len(H)):
		for eg in examples:
			x, y = eg
			frac[i] *= ll(H[i](x), y)
		nume += frac[i] * belief[i]
	return np.array(frac) * belief / nume


def weak(H, belief, examples):
	ll = lambda hx, y: 1.0 / (
			1.0 + np.exp(-0.4*(1 - 2 * np.abs(hx - y))))
	frac = [1.0] * len(H)
	nume = 0
	for i in xrange(len(H)):
		for eg in examples:
			x, y = eg
			frac[i] *= ll(H[i](x), y)
		nume += frac[i] * belief[i]
	return np.array(frac) * belief / nume


def random(H, __, ___,):
	return l1normalize(np.random.rand(len(H)))

		
policy = {
	"strong": 	strong,
	"median": 	median,
	"weak":		weak,
	"random": 	random
}


class Student:
	def __init__(self, H, stu_type="strong"):
		self.H = H
		self.belief = l1normalize(np.random.rand(len(H)))
		self.update = policy[stu_type]
	
	def reset(self, belief=None, stu_type=None):
		if belief is None:
			self.belief = l1normalize(np.random.rand(len(self.H)))
		else:
			self.belief = belief
		if stu_type is not None:
			self.update = policy[stu_type]


	def practice(self, queries):
		k = len(queries)
		h_ids = np.random.choice(len(self.H), size=k, p=self.belief)
		obs = [(queries[i], self.H[h_ids[i]](queries[i])) for i in xrange(k)]
		return obs


	def learn(self, examples):
		self.belief = self.update(self.H, self.belief, examples)


	def real_eta(self, G):
		return sum(
				[sum([(1 - np.abs(self.H[i](x) - y)) * self.belief[i] 
				for i in xrange(len(self.H))]) for (x, y) in G]) / len(G)