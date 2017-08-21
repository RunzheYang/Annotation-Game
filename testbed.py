import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pvc import *
from utils import *
from simulation import *

import argparse
parser = argparse.ArgumentParser(
	description="Testbed for Pedagogical Value-Aligned Crowdsourcing")
parser.add_argument("--acc", action="store_true", default=False,
					help="the student will use the revealed examples cumulatively")
parser.add_argument("--pos", action="store_true", default=False,
					help="the student will receive a penalty")
# TODO: How robust is the teaching algorithm?

# Feature Space Z:
Z = np.array([[f1, f2, f3]
			 for f1 in xrange(2)
			 for f2 in xrange(2)
			 for f3 in xrange(2)])

# Hypothesis Space H:
H = {
		0: lambda t: t[0],	 # h1:  triangle
		1: lambda t: 1 - t[0], # h1': circle
		2: lambda t: t[1],	 # h2:  real
		3: lambda t: 1 - t[1], # h2': dotted
		4: lambda t: t[2],	 # h3:  pink
		5: lambda t: 1 - t[2]  # h3': bule
	}

# Ground Truth Set G:
X = Z[np.random.choice(len(Z), size=10)]
G = [(x, H[4](x)) for x in X]

# Define reward:
def rw(prev, cur, gamma, eps=0.07):
	if prev - eps > cur and args.pos:
		return 0.0
	elif cur - prev + eps > gamma * (1.0 - prev):
		return 1.0
	else:
		return (cur - prev + eps) / (gamma * (1.0 - prev) + eps)

# Define logs
curves = {"round": [], "type": [], "value": []}
rewards = {"round": [], "type": [], "value": []}
num_eg = {"round": [], "type": [], "value": []}
summary_pf = {"round":[], "stutype": [], "value": []}
summary_rw = {"round":[], "stutype": [], "value": []}
summary_eg = {"round":[], "stutype": [], "value": []}
curves_sum = {"round": [], "type": [], "value": []}
super_summary_rw = {"gamma": [], "stutype": [], "value": []}
super_summary_eg = {"gamma": [], "stutype": [], "value": []}

def insert_log(r, t, v):
	if t == "reward":
		rewards["round"].append(r)
		rewards["type"].append(t)
		rewards["value"].append(v)
	elif t == "#eg":
		num_eg["round"].append(r)
		num_eg["type"].append(t)
		num_eg["value"].append(v)
	else:
		curves["round"].append(r)
		curves["type"].append(t)
		curves["value"].append(v)

def insert_summary(r, t, v, e, rew):
	summary_pf["round"].append(r)
	summary_eg["round"].append(r)
	summary_rw["round"].append(r)
	summary_pf["stutype"].append(t)
	summary_eg["stutype"].append(t)
	summary_rw["stutype"].append(t)
	summary_pf["value"].append(v)
	summary_eg["value"].append(e)
	summary_rw["value"].append(rew)

def insert_super_summary(g, t, e, r):
	super_summary_rw["gamma"].append(g)
	super_summary_eg["gamma"].append(g)
	super_summary_rw["stutype"].append(t)
	super_summary_eg["stutype"].append(t)
	super_summary_rw["value"].append(r)
	super_summary_eg["value"].append(e)


if __name__ == '__main__':

	args = parser.parse_args()
	folder_name = "results/sim_res_"
	if args.pos:
		folder_name = folder_name + "pos_"
	else:
		folder_name = folder_name + "penalty_"

	if args.acc:
		folder_name = folder_name + "acc/"
	else:
		folder_name = folder_name + "ins/"

	# Run the simulation
	k = 10
	N = 5
	init_belief = l1normalize(np.random.rand(len(H)))

	for gammas in [0.3, 0.5, 0.7, 0.9]:

		model_curves = {"round": [], "type": [], "value": []}
		for rnd in xrange(N):
		    if rnd == 0:
		    	student = Student(H)
		        student.reset(belief=init_belief)
		        eta = student.real_eta(G)
		    model_curves["round"].append(rnd)
		    model_curves["type"].append("model_student")
		    model_curves["value"].append(eta)
		    eta = eta + gammas*(1 - eta)

		summary_pf = {"round":[], "stutype": [], "value": []}
		summary_eg = {"round":[], "stutype": [], "value": []}

		for student_types in ["random", "weak", "median", "strong"]:

			curves_sum = {"round": [], "type": [], "value": []}
			curves = {"round": [], "type": [], "value": []}
			rewards = {"round": [], "type": [], "value": []}
			num_eg = {"round": [], "type": [], "value": []}

			# Initialize the Teacher
			teacher = PedagogicalReasoning(Z, H, G, alpha=3, gamma=gammas)
			# Initialize the Student
			student = Student(H, stu_type=student_types)
			totrw = 0.0
			for repeat in xrange(10):
				ot, ot_prime = [], []
				sub_totrw = 0.0
				student.reset(belief=init_belief, stu_type=student_types)
				last_est_eta = 0.0
				for rnd in xrange(N):
					# Sample k questions
					queries = Z[np.random.choice(len(Z), size=k)]
					# Practice Phase
					answers = student.practice(queries)
					est_rhos, est_eta = teacher.belief_estimate(answers)
					insert_log(rnd, "real",	 student.real_eta(G))
					insert_log(rnd, "estimated", est_eta)
					if rnd > 0:
						reward = rw(last_est_eta, est_eta, gammas)
						insert_log(rnd, "reward", reward)
						sub_totrw += reward
					else:
						reward = 0.0
						insert_log(rnd, "reward", 0.0)
					if rnd < N-1: insert_log(rnd+1, "target", gammas*(1-est_eta)+est_eta)
					# Teaching Phase
					examples, tilde_eta = teacher.teach(ot, est_rhos, est_eta, G)
					insert_log(rnd, "#eg", len(examples))
					insert_summary(rnd, student_types, student.real_eta(G),
									len(examples), reward)
					ot_prime = ot_prime + examples
					if args.acc:
						student.learn(ot_prime)
					else:
						student.learn(examples)
					last_est_eta = est_eta
					ot = ot_prime if args.acc else []
				sub_totrw = 0.0 if sub_totrw < 0 else sub_totrw
				totrw += sub_totrw
				insert_super_summary(gammas, student_types, len(ot_prime), sub_totrw)
			totrw /= 10.0

			curves_sum["round"] = curves["round"] + model_curves["round"]
			curves_sum["type"] = curves["type"] + model_curves["type"]
			curves_sum["value"] = curves["value"] + model_curves["value"]

			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			ax1 = sns.pointplot(x="round", y="value", hue="type", data=curves_sum)
			ax2 = ax1.twinx()
			ax2 = sns.barplot(x="round", y="value", hue="type", palette="Reds_r", data=rewards)
			ax1.set_ylim(top=1.1)
			ax2.set(ylim=(0.0, 3))
			ax1.set_xlabel("round")
			ax1.set_ylabel("performance")
			ax2.set_ylabel("reward")
			ax1.legend(loc="upper left")
			ax2.legend(loc="upper right")
			fig.savefig(folder_name+"gamma-(%0.2f)_type-%s_reward-(%0.2f).png"%(gammas, student_types, totrw))

			fig2 = plt.figure()
			ax = sns.barplot(x="round", y="value", hue="type", palette="Blues_d", data=num_eg)
			ax.set_xlabel("round")
			ax.set_ylabel("number of examples")
			fig2.savefig(folder_name+"gamma-(%0.2f)_type-%s_examples.png"%(gammas, student_types))

			plt.close(fig)
			plt.close(fig2)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1 = sns.pointplot(x="round", y="value", hue="stutype", palette="Set1", data=summary_pf)
		ax2 = ax1.twinx()
		ax2 = sns.barplot(x="round", y="value", hue="stutype", palette="Set1", data=summary_rw)
		ax1.set(ylim=(0.0, 1.1))
		ax2.set(ylim=(0.0, 3))
		ax1.set_xlabel("round")
		ax1.set_ylabel("performance")
		ax2.set_ylabel("reward")
		ax1.legend().set_visible(False)
		ax2.legend(loc="upper left")
		fig.savefig(folder_name+"gamma-(%0.2f)_summary.png"%(gammas))
		plt.close(fig)

		fig2 = plt.figure()
		ax = sns.barplot(x="round", y="value", hue="stutype", palette="Set1", data=summary_eg)
		ax.set_xlabel("round")
		ax.set_ylabel("number of examples")
		fig2.savefig(folder_name+"gamma-(%0.2f)_examples_summary.png"%(gammas))

fig = plt.figure()
ax = sns.barplot(x="gamma", y="value", hue="stutype", palette="Set3", data=super_summary_eg)
ax.set_xlabel("gamma")
ax.set_ylabel("total number of examples")
fig.savefig(folder_name+"examples_summary.png")

fig2 = plt.figure()
ax = sns.barplot(x="gamma", y="value", hue="stutype", palette="Set3", data=super_summary_rw)
ax.set_xlabel("gamma")
ax.set_ylabel("total reward")
fig2.savefig(folder_name+"reward_summary.png")
