"""
RHC NN training on Madelon data (Feature selection complete)

"""
import os
import csv
import time
import sys
import random

with open("ABAGAIL_absolute_path.txt") as f:    # Importing ABAGAIL.jar
	abagail_filepath = f.readline()
sys.path.append(abagail_filepath)

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU


def initialize_instances(infile):
	"""Read the  CSV data into a list of instances."""
	instances = []

	# Read in the CSV file
	with open(infile, "r") as dat:
		reader = csv.reader(dat)

		is_header = True
		for row in reader:
			if is_header:   # Skips the first line (the header rows)
				is_header = False
				continue
			instance = Instance([float(value) for value in row[:-1]])
			instance.setLabel(Instance(int(row[-1])))
			instances.append(instance)

	return instances
	

def errorOnDataSet(network, ds, measure):
	N = len(ds)
	error = 0.
	correct = 0
	incorrect = 0
	for instance in ds:
		# Predicting on instance
		network.setInputValues(instance.getData())
		network.run()
		actual = instance.getLabel().getContinuous()
		confidence = network.getOutputValues().get(0)
		if confidence >= 0.5:
			predicted = 1
		else:
			predicted = 0

		# Counting accuracy
		if abs(predicted - actual) < 0.1:
			correct += 1
		else:
			incorrect += 1

		# Error?
		output = instance.getLabel()
		output_values = network.getOutputValues()
		example = Instance(output_values, Instance(output_values.get(0)))
		error += measure.value(output, example)

	MSE = error/float(N)
	acc = correct/float(correct+incorrect)
	return MSE, acc
	
	
def train(oa, network, train_instances, test_instances, measure, filepath, max_iter):
	"""Train a given network on a set of instances.
	"""
	with open(filepath, 'w') as f:
		f.write('{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'time'))

	cur_elapsed = 0
	for iteration in xrange(max_iter + 1):
		start = time.clock()
		oa.train()
		elapsed = time.clock()-start
		cur_elapsed += elapsed

		if iteration % 10 == 0:
			MSE_trg, acc_trg = errorOnDataSet(network, train_instances, measure)
			MSE_tst, acc_tst = errorOnDataSet(network, test_instances, measure)
			txt = '{},{},{},{},{},{}\n'.format(iteration, MSE_trg, MSE_tst, acc_trg, acc_tst, cur_elapsed)
			print txt
			with open(filepath, 'a+') as f:
				f.write(txt)


def optimize_rhc(data_path, base_architecture, n_trials=5, max_iter=5000):
	gen_path = './NN_logs/RHC_LOG_@TRIAL@.txt'
	for trial in xrange(n_trials):
		filepath = gen_path.replace("@TRIAL@", str(trial))

		# Getting all data and shuffling them into train and test set
		instances = initialize_instances(data_path)
		random.shuffle(instances)
		cutoff = int(0.1 * len(instances))
		train_inst = instances[:cutoff]
		test_inst = instances[cutoff:]

		# Setting up model
		train_dataset = DataSet(train_inst)
		factory = BackPropagationNetworkFactory()
		relu = RELU()
		measure = SumOfSquaresError()
		classification_network = factory.createClassificationNetwork(base_architecture, relu)
		nnop = NeuralNetworkOptimizationProblem(train_dataset, classification_network, measure)
		oa = RandomizedHillClimbing(nnop)

		# Training the model
		train(oa, classification_network, train_inst, test_inst, measure, filepath, max_iter)


def optimize_sa(data_path, base_architecture, n_trials=5, max_iter=5000):
	gen_path = './NN_logs/SA@PAR@_LOG_@TRIAL@.txt'
	CE = [0.15, 0.35, 0.55, 0.75, 0.95]
	for cooling_mult in CE:
		for trial in xrange(n_trials):
			filepath = gen_path.replace("@PAR@", str(cooling_mult)).replace("@TRIAL@", str(trial))

			# Getting all data and shuffling them into train and test set
			instances = initialize_instances(data_path)
			random.shuffle(instances)
			cutoff = int(0.1 * len(instances))
			train_inst = instances[:cutoff]
			test_inst = instances[cutoff:]

			# Setting up model
			train_dataset = DataSet(train_inst)
			factory = BackPropagationNetworkFactory()
			relu = RELU()
			measure = SumOfSquaresError()
			classification_network = factory.createClassificationNetwork(base_architecture, relu)
			nnop = NeuralNetworkOptimizationProblem(train_dataset, classification_network, measure)
			oa = SimulatedAnnealing(1E10, cooling_mult, nnop)

			# Training the model
			train(oa, classification_network, train_inst, test_inst, measure, filepath, max_iter)


def optimize_ga(data_path, base_architecture, n_trials=5, max_iter=2000):
	gen_path = './NN_logs/GA@PAR@_LOG_@TRIAL@.txt'
	for pop in [100, 300, 500, 700, 1000]:
		mate = int(0.5 * pop)
		mutate = int(0.5 * pop)
		for trial in xrange(n_trials):
			filepath = gen_path.replace("@PAR@", str(pop)).replace("@TRIAL@", str(trial))

			# Getting all data and shuffling them into train and test set
			instances = initialize_instances(data_path)
			random.shuffle(instances)
			cutoff = int(0.1 * len(instances))
			train_inst = instances[:cutoff]
			test_inst = instances[cutoff:]

			# Setting up model
			train_dataset = DataSet(train_inst)
			factory = BackPropagationNetworkFactory()
			relu = RELU()
			measure = SumOfSquaresError()
			classification_network = factory.createClassificationNetwork(base_architecture, relu)
			nnop = NeuralNetworkOptimizationProblem(train_dataset, classification_network, measure)
			oa = StandardGeneticAlgorithm(pop, mate, mutate, nnop)

			# Training the model
			train(oa, classification_network, train_inst, test_inst, measure, filepath, max_iter)


if __name__ == "__main__":
	# Pre-processed data path
	data_path = './data/processed-wine-equality-red.csv'

	# Network parameters found "optimal" in Assignment 1
	INPUT_LAYER = 11
	HIDDEN_LAYER1 = 25
	HIDDEN_LAYER2 = 25
	HIDDEN_LAYER3 = 25
	HIDDEN_LAYER4 = 25
	OUTPUT_LAYER = 1
	MODEL_ARCHITECTURE = [INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, HIDDEN_LAYER3, HIDDEN_LAYER4, OUTPUT_LAYER]

	# Building the architecture
	# optimize_rhc(data_path, MODEL_ARCHITECTURE)
	# optimize_sa(data_path, MODEL_ARCHITECTURE)
	optimize_ga(data_path, MODEL_ARCHITECTURE)