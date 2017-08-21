import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pvc import *
from utils import *
from simulation import *

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
def rw(prev, cur, eps=0.001):
	if prev-eps > cur:
		return 0.0
	elif cur - prev + eps > 0.6 * (1.0 - prev):
		return 1.0
	else: 
		return (cur - prev + eps) / (0.6 * (1.0 - prev + eps))

# Define logs
curves = {"round": [], "type": [], "value": []}
rewards = {"round": [], "type": [], "value": []}
num_eg = {"round": [], "type": [], "value": []}

curves_sum = {"round": [], "type": [], "value": []}

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

# Run the simulation
k = 10
N = 5
init_belief = l1normalize(np.random.rand(len(H)))

model_curves = {"round": [], "type": [], "value": []}
for rnd in xrange(N):
    if rnd == 0:
    	student = Student(H)
        student.reset(belief=init_belief)
        eta = student.real_eta(G)
    model_curves["round"].append(rnd)
    model_curves["type"].append("model_student")
    model_curves["value"].append(eta)
    eta = eta + 0.6*(1 - eta)

for gammas in [0.3, 0.5, 0.7, 0.9]:
	for student_types in ["random", "weak", "median", "strong"]:
		
		curves_sum = {"round": [], "type": [], "value": []}
		curves = {"round": [], "type": [], "value": []}
		rewards = {"round": [], "type": [], "value": []}
		num_eg = {"round": [], "type": [], "value": []}
		
		# Initialize the Teacher
		teacher = PedagogicalReasoning(Z, H, G, alpha=3, gamma=gammas)
		# Initialize the Student
		student = Student(H, stu_type=student_types)
		for repeat in xrange(10):
			ot = []
			student.reset(belief=init_belief, stu_type=student_types)
			last_est_eta = 0.0
			totrw = 0.0
			for rnd in xrange(N):
				# Sample k questions
				queries = Z[np.random.choice(len(Z), size=k)]
				# Practice Phase
				answers = student.practice(queries)
				est_rhos, est_eta = teacher.belief_estimate(answers)
				insert_log(rnd, "real", student.real_eta(G))
				insert_log(rnd, "estimated", est_eta)
				reward = rw(last_est_eta, est_eta)
				totrw += reward / 10.0
				if rnd > 0: insert_log(rnd, "reward", reward)
				else: insert_log(rnd, "reward", 0.0)
				if rnd < N-1: insert_log(rnd+1, "target", 0.6*(1-est_eta)+est_eta)
				# Teaching Phase
				examples, tilde_eta = teacher.teach(ot, est_rhos, est_eta, G)
				insert_log(rnd, "#eg", len(examples))
				ot = ot + examples
				student.learn(ot)
				last_est_eta = est_eta

		curves_sum["round"] = curves["round"] + model_curves["round"]
		curves_sum["type"] = curves["type"] + model_curves["type"]
		curves_sum["value"] = curves["value"] + model_curves["value"]

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1 = sns.pointplot(x="round", y="value", hue="type", data=curves_sum)
		ax2 = ax1.twinx()
		ax2 = sns.barplot(x="round", y="value", hue="type", palette="Reds_r", data=rewards)
		ax2.set(ylim=(0.0, 3));
		ax1.set_xlabel("round");
		ax1.set_ylabel("performance");
		ax2.set_ylabel("reward");
		fig.savefig("sim_res/gamma-"+str(gammas)+"_type-"+str(student_types)+"_reward-"+str(totrw)+".png")


		fig2 = plt.figure()
		ax = sns.barplot(x="round", y="value", hue="type", palette="Blues_d", data=num_eg)
		ax.set_xlabel("round");		
		ax.set_xlabel("number of examples");
		fig2.savefig("sim_res/gamma-"+str(gammas)+"_type-"+str(student_types)+"_examples.png")

