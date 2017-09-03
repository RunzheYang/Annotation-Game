import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pvc import *
from utils import *
from simulation import *
from matplotlib.backends.backend_pdf import PdfPages

import argparse
parser = argparse.ArgumentParser(
	description="Testbed for Pedagogical Value-Aligned Crowdsourcing")
parser.add_argument("--acc", action="store_true", default=False,
					help="the student will use the revealed examples cumulatively")
parser.add_argument("--pos", action="store_true", default=False,
					help="the student will receive a penalty")
parser.add_argument("--largeH", action="store_true", default=False,
					help="use a larger hypothesis space")
parser.add_argument("--diffprior", action="store_true", default=False,
					help="test students has different prior")
# TODO: How robust is the teaching algorithm?
args = parser.parse_args()


# Feature Space Z:
# Z = np.array([[f1, f2, f3]
# 			 for f1 in xrange(2)
# 			 for f2 in xrange(2)
# 			 for f3 in xrange(2)])

Z = np.array([[f1, f2, f3]
			 for f1 in xrange(5) # Blue / Red / Green / Orange / Pink
			 for f2 in xrange(3) # Triangle / Circle / Rectangle
			 for f3 in xrange(2)]) # Real / Dotted

# Hypothesis Space H:
# H = {
# 		0: lambda t: t[0],	 # h1:  triangle
# 		1: lambda t: 1 - t[0], # h1': circle
# 		2: lambda t: t[1],	 # h2:  real
# 		3: lambda t: 1 - t[1], # h2': dotted
# 		4: lambda t: t[2],	 # h3:  pink
# 		5: lambda t: 1 - t[2]  # h3': blue
# 	}

H = {
	0: lambda t: 1.0 if t[0] == 0 else 0.0,
	1: lambda t: 1.0 if t[0] == 1 else 0.0,
	2: lambda t: 1.0 if t[0] == 2 else 0.0,
	3: lambda t: 1.0 if t[0] == 3 else 0.0,
	4: lambda t: 1.0 if t[0] == 4 else 0.0,
	5: lambda t: 1.0 if t[1] == 0 else 0.0,
	6: lambda t: 1.0 if t[1] == 1 else 0.0,
	7: lambda t: 1.0 if t[1] == 2 else 0.0,
	8: lambda t: 1.0 if t[2] == 0 else 0.0,
	9: lambda t: 1.0 if t[2] == 1 else 0.0
}

# if args.largeH:
# 	H[6] = lambda t: t[0]*t[1]
# 	H[7] = lambda t: t[1]*t[2]
# 	H[8] = lambda t: t[2]*t[0]
# 	H[9] = lambda t: (1-t[0])*t[1]
# 	H[10] = lambda t: (1-t[1])*t[2]
# 	H[11] = lambda t: (1-t[2])*t[0]
# 	H[12] = lambda t: t[0]*(1-t[1])
# 	H[13] = lambda t: t[1]*(1-t[2])
# 	H[14] = lambda t: t[2]*(1-t[0])
# 	H[15] = lambda t: (1-t[0])*(1-t[1])
# 	H[16] = lambda t: (1-t[1])*(1-t[2])
# 	H[17] = lambda t: (1-t[2])*(1-t[0])

# Ground Truth Set G:
X = Z[np.random.choice(len(Z), size=50)]
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
super_summary_all = []

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
	folder_name = "results/sim_res_"
	if args.largeH:
		folder_name = folder_name + "largeH_"
	if args.diffprior:
		folder_name = folder_name + "diffprior_"
	if args.pos:
		folder_name = folder_name + "pos_"
	else:
		folder_name = folder_name + "penalty_"
	if args.acc:
		folder_name = folder_name + "acc/"
	else:
		folder_name = folder_name + "ins/"

	# Run the simulation
	k = 15
	N = 5
	REPEAT = 100
	init_belief = l1normalize(np.random.rand(len(H)))

	for gammas in [0.3, 0.5, 0.7, 0.9]:

		if args.diffprior: continue

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
		summary_rw = {"round":[], "stutype": [], "value": []}

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
			for repeat in xrange(REPEAT):
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
					real_eta = student.real_eta(G)
					insert_log(rnd, "real",	 real_eta)
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
					insert_summary(rnd, student_types, real_eta,
									len(examples), reward)
					ot_prime = ot_prime + examples
					if args.acc:
						student.learn(ot_prime)
					else:
						student.learn(examples)
					last_est_eta = est_eta
					ot = ot_prime if args.acc else []
					super_summary_all.append({
										"gamma": gammas,
										"stu": student_types,
										"rnd": rnd,
										"real": real_eta,
										"estimated": est_eta,
										"rwd": reward,
										"#eg": len(examples)})
				sub_totrw = 0.0 if sub_totrw < 0 else sub_totrw
				totrw += sub_totrw
				insert_super_summary(gammas, student_types, len(ot_prime), sub_totrw)
			totrw /= REPEAT * 1.0

			curves_sum["round"] = curves["round"] + model_curves["round"]
			curves_sum["type"] = curves["type"] + model_curves["type"]
			curves_sum["value"] = curves["value"] + model_curves["value"]

			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			ax1 = sns.pointplot(x="round",
								y="value",
								hue="type",
								data=curves_sum,
								markers=["o", "v", "s", "x"],
								linestyles=["-", "--", "-.", ":"])
			ax2 = ax1.twinx()
			ax2 = sns.barplot(x="round", y="value", hue="type", palette="Reds_r", data=rewards)
			# ax1.set_ylim(top=1.1)
			ax2.set(ylim=(0.0, 3))
			ax1.set_xlabel("round")
			ax1.set_ylabel("performance")
			ax2.set_ylabel("reward")
			ax1.legend(loc="upper left")
			ax2.legend(loc="upper right")
			pp = PdfPages(folder_name+"gamma-(%0.2f)_type-%s_reward-(%0.2f).pdf"%(gammas, student_types, totrw))
			fig.savefig(pp, format='pdf')
			pp.close()

			fig2 = plt.figure()
			ax = sns.barplot(x="round", y="value", hue="type", palette="Blues_d", data=num_eg)
			ax.set_xlabel("round")
			ax.set_ylabel("number of examples")
			pp = PdfPages(folder_name+"gamma-(%0.2f)_type-%s_examples.pdf"%(gammas, student_types))
			fig2.savefig(pp, format='pdf')
			pp.close()

			plt.close(fig)
			plt.close(fig2)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1 = sns.pointplot(x="round",
							y="value",
							hue="stutype",
							palette="Set3",
							data=summary_pf,
							markers=["o", "v", "s", "x"],
							linestyles=["-", "--", "-.", ":"])
		ax2 = ax1.twinx()
		ax2 = sns.barplot(x="round", y="value", hue="stutype", palette="Set3", data=summary_rw)
		ax1.set(ylim=(0.0, 1.1))
		ax2.set(ylim=(0.0, 3))
		ax1.set_xlabel("round")
		ax1.set_ylabel("performance")
		ax2.set_ylabel("reward")
		ax1.legend().set_visible(False)
		ax2.legend(loc="upper left")
		pp = PdfPages(folder_name+"gamma-(%0.2f)_summary.pdf"%(gammas))
		fig.savefig(pp, format='pdf')
		pp.close()
		plt.close(fig)

		fig2 = plt.figure()
		ax = sns.barplot(x="round", y="value", hue="stutype", palette="Set3", data=summary_eg)
		ax.set_xlabel("round")
		ax.set_ylabel("number of examples")
		ax.legend(loc="upper right")
		pp = PdfPages(folder_name+"gamma-(%0.2f)_examples_summary.pdf"%(gammas))
		fig2.savefig(pp, format='pdf')
		pp.close()


	for student_types in ["random", "weak", "median", "strong"]:
		if not args.diffprior: continue
		gammas = 0.7
		summary_pf = {"round":[], "stutype": [], "value": []}
		summary_eg = {"round":[], "stutype": [], "value": []}
		summary_rw = {"round":[], "stutype": [], "value": []}
		for init_belief in [
						np.array([0.1,0.0,0.1,0.0,0.8,0.0,0.0,0.0,0.0,0.0]),
						np.array([0.2,0.0,0.2,0.0,0.6,0.0,0.0,0.0,0.0,0.0]),
						np.array([0.2,0.1,0.2,0.1,0.4,0.0,0.0,0.0,0.0,0.0]),
						np.array([0.2,0.1,0.2,0.2,0.1,0.2,0.0,0.0,0.0,0.0])
					]:

			curves = {"round": [], "type": [], "value": []}
			rewards = {"round": [], "type": [], "value": []}
			num_eg = {"round": [], "type": [], "value": []}

			# Initialize the Teacher
			teacher = PedagogicalReasoning(Z, H, G, alpha=3, gamma=gammas)
			# Initialize the Student
			student = Student(H, stu_type=student_types)
			totrw = 0.0
			for repeat in xrange(REPEAT):
				ot, ot_prime = [], []
				sub_totrw = 0.0
				student.reset(belief=init_belief, stu_type=student_types)
				last_est_eta = 0.0
				init_perfomance = student.real_eta(G)
				for rnd in xrange(N):
					# Sample k questions
					queries = Z[np.random.choice(len(Z), size=k)]
					# Practice Phase
					answers = student.practice(queries)
					est_rhos, est_eta = teacher.belief_estimate(answers)
					real_eta = student.real_eta(G)
					insert_log(rnd, "real",	 real_eta)
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
					insert_summary(rnd, init_perfomance, real_eta,
									len(examples), reward)
					ot_prime = ot_prime + examples
					if args.acc:
						student.learn(ot_prime)
					else:
						student.learn(examples)
					last_est_eta = est_eta
					ot = ot_prime if args.acc else []
					super_summary_all.append({
										"stuprior": init_perfomance,
										"stutype": student_types,
										"rnd": rnd,
										"real": real_eta,
										"estimated": est_eta,
										"rwd": reward,
										"#eg": len(examples)})
				sub_totrw = 0.0 if sub_totrw < 0 else sub_totrw
				totrw += sub_totrw
				insert_super_summary(student_types, init_perfomance, len(ot_prime), sub_totrw)
			totrw /= REPEAT * 1.0

			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			ax1 = sns.pointplot(x="round",
								y="value",
								hue="type",
								data=curves,
								markers=["o", "v", "s", "x"],
								linestyles=["-", "--", "-.", ":"])
			ax2 = ax1.twinx()
			ax2 = sns.barplot(x="round", y="value", hue="type", palette="Reds_r", data=rewards)
			ax1.set_ylim(top=1.1)
			ax2.set(ylim=(0.0, 3))
			ax1.set_xlabel("round")
			ax1.set_ylabel("performance")
			ax2.set_ylabel("reward")
			ax1.legend(loc="upper left")
			ax2.legend(loc="upper right")
			pp = PdfPages(folder_name+"type-%s_perform-(%0.2f)_reward-(%0.2f).pdf"%(student_types, init_perfomance, totrw))
			fig.savefig(pp, format="pdf")
			pp.close()

			fig2 = plt.figure()
			ax = sns.barplot(x="round", y="value", hue="type", palette="Blues_d", data=num_eg)
			ax.set_xlabel("round")
			ax.set_ylabel("number of examples")
			pp = PdfPages(folder_name+"type-%s_perform-(%0.2f)_examples.pdf"%(student_types, init_perfomance))
			fig2.savefig(pp, format="pdf")
			pp.close()

			plt.close(fig)
			plt.close(fig2)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1 = sns.pointplot(x="round",
							y="value",
							hue="stutype",
							palette="Set3",
							data=summary_pf,
							markers=["o", "v", "s", "x"],
							linestyles=["-", "--", "-.", ":"])
		ax2 = ax1.twinx()
		ax2 = sns.barplot(x="round", y="value", hue="stutype", palette="Set3", data=summary_rw)
		# ax1.set(ylim=(0.0, 1.1))
		ax2.set(ylim=(0.0, 3))
		ax1.set_xlabel("round")
		ax1.set_ylabel("performance")
		ax2.set_ylabel("reward")
		ax1.legend().set_visible(False)
		ax2.legend(loc="upper left")
		pp = PdfPages(folder_name+"type-%s_summary.pdf"%(student_types))
		fig.savefig(pp, format="pdf")
		pp.close()
		plt.close(fig)

		fig2 = plt.figure()
		ax = sns.barplot(x="round", y="value", hue="stutype", palette="Set3", data=summary_eg)
		ax.set_xlabel("round")
		ax.set_ylabel("number of examples")
		ax.legend(loc="upper right")
		pp = PdfPages(folder_name+"type-%s_examples_summary.pdf"%(student_types))
		fig2.savefig(pp, format="pdf")
		pp.close()

	fig = plt.figure()
	ax = sns.barplot(x="gamma", y="value", hue="stutype", palette="Set3", data=super_summary_eg)
	if args.diffprior:
		ax.set_xlabel("learnability")
		ax.legend(loc="upper right")
	else:
		ax.set_xlabel("gamma")
		ax.legend(loc="upper left")
	ax.set_ylabel("total number of examples")
	pp = PdfPages(folder_name+"examples_summary.pdf")
	fig.savefig(pp, format="pdf")
	pp.close()

	fig2 = plt.figure()
	ax = sns.barplot(x="gamma", y="value", hue="stutype", palette="Set3", data=super_summary_rw)
	if args.diffprior:
		ax.set_xlabel("learnability")
	else:
		ax.set_xlabel("gamma")
	ax.set_ylabel("total reward")
	ax.set(ylim=(0.0, 4.0))
	ax.legend(loc="upper left")
	pp = PdfPages(folder_name+"reward_summary.pdf")
	fig2.savefig(pp, format="pdf")
	pp.close()

	f = open(folder_name + "log.json", 'w')
	json.dump(super_summary_all, f)
