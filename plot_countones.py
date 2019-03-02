import matplotlib.pyplot as plt
import pandas as pd
from itertools import product


def all_trials_count_rhc(save=False):
	"""
	Plotting fitness vs iterations in all trials for the count ones problem using RHC
	"""
	gen_path = '.\COUNT_logs\COUNT_RHC_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_RHC.png'

	df = [0 for _ in range(0, 5)]
	for i in range(0, 5):
		filepath = gen_path.replace("@TRIAL@", str(i + 1))
		df[i] = pd.read_csv(filepath)

	plt.figure()
	plt.title("RHC for count ones problem")
	x = df[0].loc[:, 'iterations']
	for i in range(0, 5):
		y = df[i].loc[:, 'fitness']
		plt.plot(x, y)

	if save:
		plt.savefig(save_path, dpi=300)


def all_trials_count_sa(save=False):
	"""
	Plotting fitness vs iterations in all trials for the count ones problem using SA
	"""
	gen_path = '.\COUNT_logs\COUNT_SA@COOLING@_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_SA@COOLING@.png'

	for cooling_mult in [0.15, 0.35, 0.55, 0.75, 0.95]:
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@COOLING@", str(cooling_mult))
			df[i] = pd.read_csv(filepath)

		plt.figure()
		plt.title("SA with cooling multipler {} for count ones problem".format(cooling_mult))
		x = df[0].loc[:, 'iterations']
		for i in range(0, 5):
			y = df[i].loc[:, 'fitness']
			plt.plot(x, y)

		if save:
			plt.savefig(save_path.replace("@COOLING@", str(cooling_mult)), dpi=300)


def best_trials_count_sa(save=False):
	"""
	Plotting fitness vs iterations in the best trials for each cooling multiplier
	for the TSP problem using SA
	"""
	gen_path = '.\COUNT_logs\COUNT_SA@COOLING@_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_SA.png'

	best = []
	cooling_mult = [0.15, 0.35, 0.55, 0.75, 0.95]
	for k in range(0, len(cooling_mult)):
		df = [0 for _ in range(0, 5)]
		best_trial_ix = 0
		best_fitness = -1
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@COOLING@", str(cooling_mult[k]))
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_trial_ix = i
		best.append(df[best_trial_ix])

	plt.figure()
	plt.title("Best trial for each SA using different cooling multipliers for count ones problem")
	x = best[0].loc[:, 'iterations']
	for i in range(0, 5):
		y = best[i].loc[:, 'fitness']
		plt.plot(x, y, label=str(cooling_mult[i]))
	plt.legend()

	if save:
		plt.savefig(save_path, dpi=300)


def all_trials_count_ga(save=False):
	"""
	Plotting fitness vs iterations in all trials for the TSP problem using GA
	"""
	gen_path = '.\COUNT_logs\COUNT_GA@PAR@_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_GA@PAR@.png'

	for pop, mutate_frac in product([100, 500], [0.1, 0.3, 0.5]):
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", "{}_{}".format(pop, mutate_frac))
			df[i] = pd.read_csv(filepath)

		plt.figure()
		plt.title("GA with population {} and mutate {} for count ones problem".format(pop, mutate_frac))
		x = df[0].loc[:, 'iterations']
		for i in range(0, 5):
			y = df[i].loc[:, 'fitness']
			plt.plot(x, y)

		if save:
			plt.savefig(save_path.replace("@PAR@", "{}_{}".format(pop, mutate_frac)), dpi=300)


def best_trials_count_ga(save=False):
	"""
	Plotting fitness vs iterations in the best trials for each param set
	for the TSP problem using GA
	"""
	gen_path = '.\COUNT_logs\COUNT_GA@PAR@_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_GA.png'

	best = []
	param_str = []
	for pop, mutate_frac in product([100, 500], [0.1, 0.3, 0.5]):
		param_str.append("{}_{}".format(pop, mutate_frac))  # Used later for labeling too
		df = [0 for _ in range(0, 5)]
		best_trial_ix = 0
		best_fitness = -1
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", param_str[-1])
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_trial_ix = i
		best.append(df[best_trial_ix])

	plt.figure()
	plt.title("Best trial for each GA using different population/mate/mutate numbers for count ones problem")
	x = best[0].loc[:, 'iterations']
	for i in range(0, len(param_str)):
		y = best[i].loc[:, 'fitness']
		plt.plot(x, y, label=param_str[i])
	plt.legend()

	if save:
		plt.savefig(save_path, dpi=300)


def all_trials_count_mimic(save=False):
	"""
	Plotting fitness vs iterations in all trials for the TSP problem using MIMIC
	"""
	gen_path = '.\COUNT_logs\COUNT_MIMIC@PAR@_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_MIMIC@PAR@.png'

	for samples in [50, 150, 250]:
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", "{}".format(samples))
			df[i] = pd.read_csv(filepath)

		plt.figure()
		plt.title("MIMIC with population {} for count ones problem".format(samples))
		x = df[0].loc[:, 'iterations']
		for i in range(0, 5):
			y = df[i].loc[:, 'fitness']
			plt.plot(x, y)

		if save:
			plt.savefig(save_path.replace("@PAR@", "{}".format(samples)), dpi=300)


def best_trials_count_mimic(save=False):
	"""
	Plotting fitness vs iterations in the best trials for each param set
	for the TSP problem using MIMIC
	"""
	gen_path = '.\COUNT_logs\COUNT_MIMIC@PAR@_@TRIAL@_LOG.txt'
	save_path = '.\graphs\COUNT_MIMIC.png'

	best = []
	param_str = []
	for samples in [50, 150, 250]:
		param_str.append("{}".format(samples))  # Used later for labeling too
		df = [0 for _ in range(0, 5)]
		best_trial_ix = 0
		best_fitness = -1
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", param_str[-1])
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_trial_ix = i
		best.append(df[best_trial_ix])

	plt.figure()
	plt.title("Best trial for each MIMIC using different sample/keep numbers for count ones problem")
	x = best[0].loc[:, 'iterations']
	for i in range(0, len(param_str)):
		y = best[i].loc[:, 'fitness']
		plt.plot(x, y, label=param_str[i])
	plt.legend()

	if save:
		plt.savefig(save_path, dpi=300)


def best_count_opt(save=False):
	"""
	Plotting the best trials between all algorithms:
	fitness vs iterations
	function evaluations vs iterations
	time vs iterations
	"""
	# RHC, SA, GA, MIMIC
	best_df = [0, 0, 0, 0]
	labels = ['', '', '', '']

	# Best RHC
	gen_path = '.\COUNT_logs\COUNT_RHC_@TRIAL@_LOG.txt'
	best_fitness = -1
	df = [0 for _ in range(0, 5)]
	for i in range(0, 5):
		filepath = gen_path.replace("@TRIAL@", str(i + 1))
		df[i] = pd.read_csv(filepath)
		if df[i]['fitness'].iloc[-1] > best_fitness:
			best_fitness = df[i]['fitness'].iloc[-1]
			best_df[0] = df[i]
			labels[0] = "RHC"

	# Best SA
	gen_path = '.\COUNT_logs\COUNT_SA@COOLING@_@TRIAL@_LOG.txt'
	best_fitness = -1
	for cooling_mult in [0.15, 0.35, 0.55, 0.75, 0.95]:
		par = str(cooling_mult)
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@COOLING@", par)
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_df[1] = df[i]
				labels[1] = "SA_{}".format(par)

	# Best GA
	gen_path = '.\COUNT_logs\COUNT_GA@PAR@_@TRIAL@_LOG.txt'
	best_fitness = -1
	for pop, mutate_frac in product([100, 500], [0.1, 0.3, 0.5]):
		par = "{}_{}".format(pop, mutate_frac)
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", par)
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_df[2] = df[i]
				labels[2] = "GA_{}".format(par)

	# Best MIMIC
	gen_path = '.\COUNT_logs\COUNT_MIMIC@PAR@_@TRIAL@_LOG.txt'
	best_fitness = -1
	for samples in [50, 150, 250]:
		par = "{}".format(samples)
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", par)
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_df[3] = df[i]
				labels[3] = "MIMIC_{}".format(par)

	# Fitness vs Iteration
	plt.figure()
	plt.title("Fitness vs Iterations of each algorithm for count ones")
	for i in range(0, 4):
		x = best_df[i]['iterations']
		y = best_df[i]['fitness']
		plt.plot(x, y, label=labels[i])
	plt.legend()
	if save:
		plt.savefig("./graphs/COUNT_FIT_ITER.png", dpi=300)


	# Fitness vs Function Evaluations
	plt.figure()
	plt.title("Fitness vs Function Evaluations of each algorithm for count ones")
	x = best_df[0]['fevals']
	y = best_df[0]['fitness']
	plt.plot(x, y, label=labels[0])
	x = best_df[1]['fevals']
	y = best_df[1]['fitness']
	plt.plot(x, y, label=labels[1])
	x = best_df[2]['fevals'].iloc[:3]
	y = best_df[2]['fitness'].iloc[:3]
	plt.plot(x, y, label=labels[2])
	x = best_df[3]['fevals'].iloc[:10]
	y = best_df[3]['fitness'].iloc[:10]
	plt.plot(x, y, label=labels[3])
	plt.legend()
	if save:
		plt.savefig("./graphs/COUNT_FIT_EVALS.png", dpi=300)


	# Fitness vs Time
	plt.figure()
	plt.title("Fitness vs Time of each algorithm for count ones")
	x = best_df[0]['time']
	y = best_df[0]['fitness']
	plt.plot(x, y, label=labels[0])
	x = best_df[1]['time']
	y = best_df[1]['fitness']
	plt.plot(x, y, label=labels[1])
	x = best_df[2]['time'].iloc[:10]
	y = best_df[2]['fitness'].iloc[:10]
	plt.plot(x, y, label=labels[2])
	x = best_df[3]['time'].iloc[:3]
	y = best_df[3]['fitness'].iloc[:3]
	plt.plot(x, y, label=labels[3])
	plt.legend()
	if save:
		plt.savefig("./graphs/COUNT_FIT_TIME.png", dpi=300)


if __name__ == "__main__":
	save = True
	all_trials_count_rhc(save)
	best_trials_count_sa(save)
	all_trials_count_sa(save)
	all_trials_count_ga(save)
	best_trials_count_ga(save)
	all_trials_count_mimic(save)
	best_trials_count_mimic(save)
	best_count_opt(save)