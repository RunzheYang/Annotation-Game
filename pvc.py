# Pedagogical Value-Aligned Crowdsourcing
# @Tony Runzhe Yang

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog


class PedagogicalReasoning:
	# Prepare the kernel of transformation B,
	# accuracy u, and learning mask psi.
	def __init__(self, Z, H, G, alpha = 2.0, gamma = 0.3):
		B = np.zeros((2*len(Z), len(H)))
		u = np.zeros(len(H))
		psi = {}
		summask = {}
		for i in xrange(len(H)):
			h = H[i]
			for r in xrange(len(Z)):
				x = Z[r]
				B[2 * r, i] = h(x)
				B[2 * r + 1, i] =  1 - h(x)
				xy = str((x, h(x)))
				if xy not in summask:
					summask[xy] = []
				summask[xy].append(i)
			for (x, y) in G:
				xy = str((x, y))
				u[i] = u[i] + 1.0 / len(G) * (1 - np.abs(h(x) - y))
				if xy not in psi:
					psi[xy] = np.zeros(len(H))
				psi[xy][i] = sigma(h(x), y, alpha)
		KB = Kernel(B)
		self.KB, self.u, self.psi = KB, u, psi
		self.gamma = gamma
		self.summask = summask

	# Belief Estimate, returns the whole equivalence class
	# and the estimated performance
	def belief_estimate(self, gt):
		rho = MLE(gt, len(self.u), self.summask)
		eta = self.u.dot(rho)
		return (rho, self.KB), eta


	# Teaching via giving examples
	def teach(self, ot, rhos, eta, G, limit=20, grain=0.001):
		at = []
		rho, region = rhos
		eta_prime = eta
		psi = np.ones(rho.shape)
		for xy in ot:
			xy = str(xy)
			psi = psi * self.psi[xy]
		delta = self.gamma * (1 - eta)
		eps = np.ones(rho.size) * grain
		A_ub = np.concatenate((region, -region), axis=0)
		b_ub = np.concatenate((np.ones(rho.size) - rho - eps, rho - eps))
		A_eq = np.ones(rho.size).dot(region).reshape(1, -1)
		b_eq = np.zeros(1)
		rst = (A_ub, b_ub, A_eq, b_eq)

		if len(ot) > 0:
			tmpA = psi * (1 - self.u)
			tmpC = (psi * (1 - self.u)).dot(region)
			eta_prime = 1 + LP(-tmpC, rst) - tmpA.dot(rho)
			beta = psi.dot(region.dot(LP(-tmpC, rst, "x")) + rho)
			delta = (1 - beta * (1 - self.gamma)) * (1 - eta)

		cnt = 0
		while eta_prime - eta < delta and cnt < limit:
			cnt += 1
			tmpA = psi * (1 - self.u)
			tmpB = tmpA.reshape(-1, 1) * region
			E = np.array([LP(-self.psi[str(xy)].dot(tmpB), rst) for xy in G])
			E = E + np.array(
				[1 - (self.psi[str(xy)] * tmpA).dot(rho) for xy in G])
			xy = G[E.argmax()]
			at.append(xy)
			psi = psi * self.psi[str(xy)]
			tmpC = (psi * (1 - self.u)).dot(region)
			eta_prime = np.max(E)
			beta = psi.dot(region.dot(LP(-tmpC, rst, "x")) + rho)
			delta = (1 - beta * (1 - self.gamma)) * (1 - eta)
		return at, (eta_prime / beta) - (1 - beta) / beta


# Noice-tolerant likelihood function
def sigma(hx, y, alpha):
	logit = np.exp(-alpha * (1 - 2 * np.abs(hx - y)))
	return 1.0 / (1.0 + logit)


# Calculate null space of the given matrix
def Kernel(A, atol=1e-13, rtol=0):
	A = np.atleast_2d(A)
	u, s, vh = np.linalg.svd(A)
	tol = max(atol, rtol * s[0])
	nnz = (s >= tol).sum()
	ns = vh[nnz:].conj().T
	return ns


# loglikelihood
def lnlikelihood(theta, gt, summask):
	theta = theta - theta.max()
	est_rho = np.exp(theta)
	partition = np.exp(theta).sum()
	est_rho = est_rho / partition
	confp = np.array([est_rho[summask[str(ans)]].sum() for ans in gt])
	lnp = np.log(confp).sum()
	return lnp


# Maximum Likelihood Solver
def MLE(gt, dim, summask):
	nll = lambda *args: -lnlikelihood(*args)
	result = minimize(nll, np.zeros(dim),
					  args=(gt, summask), tol=1e-6)
	theta = result["x"]
	theta = theta - theta.max()
	est_rho = np.exp(theta)
	partition = np.exp(theta).sum()
	return est_rho / partition


# Linear Programming for solving the worst case
def LP(c, rst, result="fun"):
	A_ub, b_ub, A_eq, b_eq = rst
	scalar = 1000.0 if np.linalg.norm(c) < 0.001 else 1.0
	res = linprog(c * scalar,
		A_ub = A_ub,
		b_ub = b_ub,
		A_eq = A_eq,
		b_eq = b_eq)
	if res["status"] == 2: # infeasible
		if result == "x":
			return np.zeros(c.size)
		else:
			return 0.0
	else:
		return res[result] / scalar
