import json
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
sns.set(style='ticks', palette='Set2')

def plot_curve_and_reward(curves, rewards, gammas, student_types):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1 = sns.pointplot(x="round",
						y="value",
						hue="type",
						data=curves_sum,
						ms=14,
						markeredgewidth=5,
						markers=["o", "v", "s", "x"],
						markerfacecolor="None",
						linestyles=["-", "--", "-.", ":"])
	ax2 = ax1.twinx()
	ax2 = sns.barplot(x="round",
					  y="value",
					  hue="type", 
					  palette="Reds_r", 
					  data=rewards)
	# ax1.set_ylim(top=1.1)
	ax2.set(ylim=(0.0, 3))
	ax1.set_xlabel("round")
	ax1.set_ylabel("performance")
	ax2.set_ylabel("reward")
	ax1.legend(loc="upper left")
	ax2.legend(loc="upper right")
	pp = PdfPages("rp-gamma-(%0.2f)_type-%s.pdf"%(gammas, student_types))
	fig.savefig(pp, format='pdf')
	pp.close()

file = open("log.json")
records = json.load(file)
file.close()

# initialize curves & rewards
curves_sum = {"round": [], "type": [], "value": []}
curves = {"round": [], "type": [], "value": []}
model_curves = {"round": [], "type": [], "value": []}
rewards = {"round": [], "type": [], "value": []}

# get corresponding records
for rec in records:
	if rec["gamma"] == 0.7 and rec["stu"] == "random":
		if rec["rnd"] == 0:
			init_perf = rec["real"]
		curves["round"].append(rec["rnd"])
		curves["round"].append(rec["rnd"])
		curves["type"].append("real")
		curves["type"].append("estimated")
		curves["value"].append(rec["real"])
		curves["value"].append(rec["estimated"])
		if rec["rnd"] < 4:
			curves["round"].append(rec["rnd"]+1)
			curves["type"].append("target")
			curves["value"].append(
				(1.0 - rec["estimated"]) * 0.7 + rec["estimated"])
		rewards["round"].append(rec["rnd"])
		rewards["type"].append("reward")
		rewards["value"].append(rec["rwd"])

for i in xrange(5):
	model_curves["round"].append(i)
	model_curves["type"].append("model_student")
	model_curves["value"].append(init_perf)
	init_perf = (1 - init_perf) * 0.7 + init_perf

curves_sum["round"] = curves["round"] + model_curves["round"]
curves_sum["type"]  = curves["type"]  + model_curves["type"]
curves_sum["value"] = curves["value"] + model_curves["value"]


plot_curve_and_reward(curves_sum, rewards, 0.7, "random")


