import matplotlib.pyplot as plt
import pandas as pd
from itertools import product


def all_trials_tsp_rhc(save=False):
	"""
	Plotting fitness vs iterations in all trials for the TSP problem using RHC
	"""
	gen_path = '.\TSP_logs\TSP_RHC_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_RHC.png'

	df = [0 for _ in range(0, 5)]
	for i in range(0, 5):
		filepath = gen_path.replace("@TRIAL@", str(i + 1))
		df[i] = pd.read_csv(filepath)

	plt.figure()
	plt.title("RHC for TSP")
	x = df[0].loc[:, 'iterations']
	for i in range(0, 5):
		y = df[i].loc[:, 'fitness']
		plt.plot(x, y)

	if save:
		plt.savefig(save_path, dpi=300)


def all_trials_tsp_sa(save=False):
	"""
	Plotting fitness vs iterations in all trials for the TSP problem using SA
	"""
	gen_path = '.\TSP_logs\TSP_SA@COOLING@_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_@COOLING@.png'

	for cooling_mult in [0.15, 0.35, 0.55, 0.75, 0.95]:
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@COOLING@", str(cooling_mult))
			df[i] = pd.read_csv(filepath)

		plt.figure()
		plt.title("SA with cooling multipler {} for TSP".format(cooling_mult))
		x = df[0].loc[:, 'iterations']
		for i in range(0, 5):
			y = df[i].loc[:, 'fitness']
			plt.plot(x, y)

		if save:
			plt.savefig(save_path.replace("@COOLING@", str(cooling_mult)), dpi=300)


def best_trials_tsp_sa(save=False):
	"""
	Plotting fitness vs iterations in the best trials for each cooling multiplier
	for the TSP problem using SA
	"""
	gen_path = '.\TSP_logs\TSP_SA@COOLING@_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_SA.png'

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
	plt.title("Best trial for each SA using different cooling multipliers for TSP")
	x = best[0].loc[:, 'iterations']
	for i in range(0, 5):
		y = best[i].loc[:, 'fitness']
		plt.plot(x, y, label=str(cooling_mult[i]))
	plt.legend()

	if save:
		plt.savefig(save_path, dpi=300)


def all_trials_tsp_ga(save=False):
	"""
	Plotting fitness vs iterations in all trials for the TSP problem using GA
	"""
	gen_path = '.\TSP_logs\TSP_GA@PAR@_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_GA@PAR@.png'

	for pop in [100, 500, 1000]:
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", "{}".format(pop))
			df[i] = pd.read_csv(filepath)

		plt.figure()
		plt.title("GA with population {} for TSP".format(pop))
		x = df[0].loc[:, 'iterations']
		for i in range(0, 5):
			y = df[i].loc[:, 'fitness']
			plt.plot(x, y)

		if save:
			plt.savefig(save_path.replace("@PAR@", "{}".format(pop)), dpi=300)


def best_trials_tsp_ga(save=False):
	"""
	Plotting fitness vs iterations in the best trials for each param set
	for the TSP problem using GA
	"""
	gen_path = '.\TSP_logs\TSP_GA@PAR@_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_GA.png'

	best = []
	param_str = []
	for pop in [100, 500, 1000]:
		param_str.append("{}".format(pop))  # Used later for labeling too
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
	plt.title("Best trial for each GA using different population/mate/mutate numbers for TSP")
	x = best[0].loc[:, 'iterations']
	for i in range(0, len(param_str)):
		y = best[i].loc[:, 'fitness']
		plt.plot(x, y, label=param_str[i])
	plt.legend()

	if save:
		plt.savefig(save_path, dpi=300)


def all_trials_tsp_mimic(save=False):
	"""
	Plotting fitness vs iterations in all trials for the TSP problem using MIMIC
	"""
	gen_path = '.\TSP_logs\TSP_MIMIC@PAR@_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_MIMIC@PAR@.png'

	for samples in [300, 500, 700]:
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", "{}".format(samples))
			df[i] = pd.read_csv(filepath)

		plt.figure()
		plt.title("MIMIC with population {} for TSP".format(samples))
		x = df[0].loc[:, 'iterations']
		for i in range(0, 5):
			y = df[i].loc[:, 'fitness']
			plt.plot(x, y)

		if save:
			plt.savefig(save_path.replace("@PAR@", '{}'.format(samples)), dpi=300)


def best_trials_tsp_mimic(save=False):
	"""
	Plotting fitness vs iterations in the best trials for each param set
	for the TSP problem using MIMIC
	"""
	gen_path = '.\TSP_logs\TSP_MIMIC@PAR@_@TRIAL@_LOG.txt'
	save_path = './graphs/TSP_MIMIC.png'

	best = []
	param_str = []
	for samples in [300, 500, 700]:
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
	plt.title("Best trial for each MIMIC using different sample/keep numbers for TSP")
	x = best[0].loc[:, 'iterations']
	for i in range(0, len(param_str)):
		y = best[i].loc[:, 'fitness']
		plt.plot(x, y, label=param_str[i])
	plt.legend()

	if save:
		plt.savefig(save_path, dpi=300)


def best_tsp_opt(save=False):
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
	gen_path = '.\TSP_logs\TSP_RHC_@TRIAL@_LOG.txt'
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
	gen_path = '.\TSP_logs\TSP_SA@COOLING@_@TRIAL@_LOG.txt'
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
	gen_path = '.\TSP_logs\TSP_GA@PAR@_@TRIAL@_LOG.txt'
	best_fitness = -1
	for pop in [100, 500, 1000]:
		mate = 0.5 * pop
		mutate = 0.1 * pop
		par = "{}".format(pop)
		df = [0 for _ in range(0, 5)]
		for i in range(0, 5):
			filepath = gen_path.replace("@TRIAL@", str(i + 1)).replace("@PAR@", par)
			df[i] = pd.read_csv(filepath)
			if df[i]['fitness'].iloc[-1] > best_fitness:
				best_fitness = df[i]['fitness'].iloc[-1]
				best_df[2] = df[i]
				labels[2] = "GA_{}".format(par)

	# Best MIMIC
	gen_path = '.\TSP_logs\TSP_MIMIC@PAR@_@TRIAL@_LOG.txt'
	best_fitness = -1
	for samples in [300, 500, 700]:
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
	plt.title("Fitness vs Iterations of each algorithm for TSP")
	for i in range(0, 4):
		x = best_df[i]['iterations']
		y = best_df[i]['fitness']
		plt.plot(x, y, label=labels[i])
	plt.legend()
	if save:
		plt.savefig("./graphs/TSP_FIT_ITER.png", dpi=300)

	# Fitness vs Function Evaluations
	plt.figure()
	plt.title("Fitness vs Function Evaluations of each algorithm for TSP")
	x = best_df[0]['fevals']
	y = best_df[0]['fitness']
	plt.plot(x, y, label=labels[0])
	x = best_df[1]['fevals']
	y = best_df[1]['fitness']
	plt.plot(x, y, label=labels[1])
	x = best_df[2]['fevals'].iloc[:10]
	y = best_df[2]['fitness'].iloc[:10]
	plt.plot(x, y, label=labels[2])
	x = best_df[3]['fevals'].iloc[:10]
	y = best_df[3]['fitness'].iloc[:10]
	plt.plot(x, y, label=labels[3])
	plt.legend()
	if save:
		plt.savefig("./graphs/TSP_FIT_EVALS.png", dpi=300)

	# Fitness vs Time
	plt.figure()
	plt.title("Fitness vs Time of each algorithm for TSP")
	x = best_df[0]['time']
	y = best_df[0]['fitness']
	plt.plot(x, y, label=labels[0])
	x = best_df[1]['time']
	y = best_df[1]['fitness']
	plt.plot(x, y, label=labels[1])
	x = best_df[2]['time'].iloc[:150]
	y = best_df[2]['fitness'].iloc[:150]
	plt.plot(x, y, label=labels[2])
	x = best_df[3]['time'].iloc[:2]
	y = best_df[3]['fitness'].iloc[:2]
	plt.plot(x, y, label=labels[3])
	plt.legend()
	if save:
		plt.savefig("./graphs/TSP_FIT_TIME.png", dpi=300)


if __name__ == "__main__":
	save = True
	all_trials_tsp_rhc(save)
	all_trials_tsp_sa(save)
	best_trials_tsp_sa(save)
	all_trials_tsp_ga(save)
	best_trials_tsp_ga(save)
	all_trials_tsp_mimic(save)
	best_trials_tsp_mimic(save)
	best_tsp_opt(save)