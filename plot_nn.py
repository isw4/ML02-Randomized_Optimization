import matplotlib.pyplot as plt
import pandas as pd

def plot_rhc(save=False):
	gen_path = './NN_logs/RHC_LOG_@TRIAL@.txt'

	dfs = []
	for trial in range(0, 5):
		filepath = gen_path.replace("@TRIAL@", str(trial))
		dfs.append(pd.read_csv(filepath))

	# Finding the average of each metric
	train_acc = pd.DataFrame(dfs[0]['iteration'])
	test_acc  = pd.DataFrame(dfs[0]['iteration'])
	train_MSE = pd.DataFrame(dfs[0]['iteration'])
	test_MSE  = pd.DataFrame(dfs[0]['iteration'])
	for i in range(0, len(dfs)):
		train_acc[i] = dfs[i]['acc_trg']
		test_acc[i]  = dfs[i]['acc_tst']
		train_MSE[i] = dfs[i]['MSE_trg']
		test_MSE[i]  = dfs[i]['MSE_tst']
	train_acc['average'] = train_acc.iloc[:, 1:len(dfs)+1].mean(axis=1)
	test_acc['average']  = test_acc.iloc[:, 1:len(dfs) + 1].mean(axis=1)
	train_MSE['average'] = train_MSE.iloc[:, 1:len(dfs) + 1].mean(axis=1)
	test_MSE['average']  = test_MSE.iloc[:, 1:len(dfs) + 1].mean(axis=1)

	# Plot
	plt.figure()
	plt.title("Accuracy using RHC for Neural Network Training")
	x = train_acc['iteration']
	train_y = train_acc['average']
	test_y = test_acc['average']
	plt.plot(x, train_y, label="Training accuracy")
	plt.plot(x, test_y, label="Testing accuracy")
	plt.legend()
	if save:
		plt.savefig('./graphs/RHC_NN_ACC.png', dpi=300)

	plt.figure()
	plt.title("MSE using RHC for Neural Network Training")
	x = train_MSE['iteration']
	train_y = train_MSE['average']
	test_y = test_MSE['average']
	plt.plot(x, train_y, label="Training accuracy")
	plt.plot(x, test_y, label="Testing accuracy")
	plt.legend()
	if save:
		plt.savefig('./graphs/RHC_NN_MSE.png', dpi=300)


def plot_sa(save=False):
	gen_path = './NN_logs/SA@PAR@_LOG_@TRIAL@.txt'

	CE = [0.15, 0.35, 0.55, 0.75, 0.95]
	averages = []
	for cooling_mult in CE:
		dfs = []
		for trial in range(0, 5):
			filepath = gen_path.replace("@PAR@", str(cooling_mult)).replace("@TRIAL@", str(trial))
			dfs.append(pd.read_csv(filepath))

		# Finding the average of each metric
		train_acc = pd.DataFrame(dfs[0]['iteration'])
		test_acc = pd.DataFrame(dfs[0]['iteration'])
		train_MSE = pd.DataFrame(dfs[0]['iteration'])
		test_MSE = pd.DataFrame(dfs[0]['iteration'])
		for i in range(0, len(dfs)):
			train_acc[i] = dfs[i]['acc_trg']
			test_acc[i] = dfs[i]['acc_tst']
			train_MSE[i] = dfs[i]['MSE_trg']
			test_MSE[i] = dfs[i]['MSE_tst']
		avg_df = pd.DataFrame(dfs[0]['iteration'])
		avg_df['train_acc'] = train_acc.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		avg_df['test_acc']  = test_acc.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		avg_df['train_MSE'] = train_MSE.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		avg_df['test_MSE']  = test_MSE.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		averages.append(avg_df)

	# Plot
	plt.figure()
	plt.title("Train Accuracy using SA for Neural Network Training")
	x = averages[0]['iteration']
	for i in range(0, len(CE)):
		y = averages[i]['train_acc']
		plt.plot(x, y, label=str(CE[i]))
	plt.legend()
	if save:
		plt.savefig('./graphs/SA_NN_ACC_TRAIN.png', dpi=300)

	plt.figure()
	plt.title("Test Accuracy using SA for Neural Network Training")
	x = averages[0]['iteration']
	for i in range(0, len(CE)):
		y = averages[i]['test_acc']
		plt.plot(x, y, label=str(CE[i]))
	plt.legend()
	if save:
		plt.savefig('./graphs/SA_NN_ACC_TEST.png', dpi=300)


def plot_ga(save=False):
	gen_path = './NN_logs/GA@PAR@_LOG_@TRIAL@.txt'

	pop = [100, 300, 500, 700, 1000]
	averages = []
	for p in pop:
		dfs = []
		for trial in range(0, 5):
			filepath = gen_path.replace("@PAR@", str(p)).replace("@TRIAL@", str(trial))
			dfs.append(pd.read_csv(filepath))

		# Finding the average of each metric
		train_acc = pd.DataFrame(dfs[0]['iteration'])
		test_acc = pd.DataFrame(dfs[0]['iteration'])
		train_MSE = pd.DataFrame(dfs[0]['iteration'])
		test_MSE = pd.DataFrame(dfs[0]['iteration'])
		for i in range(0, len(dfs)):
			train_acc[i] = dfs[i]['acc_trg']
			test_acc[i] = dfs[i]['acc_tst']
			train_MSE[i] = dfs[i]['MSE_trg']
			test_MSE[i] = dfs[i]['MSE_tst']
		avg_df = pd.DataFrame(dfs[0]['iteration'])
		avg_df['train_acc'] = train_acc.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		avg_df['test_acc'] = test_acc.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		avg_df['train_MSE'] = train_MSE.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		avg_df['test_MSE'] = test_MSE.iloc[:, 1:len(dfs) + 1].mean(axis=1)
		averages.append(avg_df)

	# Plot
	plt.figure()
	plt.title("Train Accuracy using GA for Neural Network Training")
	x = averages[0]['iteration']
	for i in range(0, len(pop)):
		y = averages[i]['train_acc']
		plt.plot(x, y, label=str(pop[i]))
	plt.legend()
	if save:
		plt.savefig('./graphs/GA_NN_ACC_TRAIN.png', dpi=300)

	plt.figure()
	plt.title("Test Accuracy using GA for Neural Network Training")
	x = averages[0]['iteration']
	for i in range(0, len(pop)):
		y = averages[i]['test_acc']
		plt.plot(x, y, label=str(pop[i]))
	plt.legend()
	if save:
		plt.savefig('./graphs/GA_NN_ACC_TEST.png', dpi=300)


if __name__ == "__main__":
	save = True
	# plot_rhc(save)
	plot_sa(save)
	plot_ga(save)
	plt.show()