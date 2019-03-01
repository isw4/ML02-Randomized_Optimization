import sys
import os
import time
from time import clock
from itertools import product
from array import array

with open("ABAGAIL_absolute_path.txt") as f:    # Importing ABAGAIL.jar
	abagail_filepath = f.readline()
sys.path.append(abagail_filepath)

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction


def optimize(n_vertices=100):
	random = Random()
	# set N value.  This is the number of points
	N = n_vertices

	# Number of trials. If the number is changed, the plotting function needs to be updated to take in the right
	# number of csv files
	num_trials = 5

	# Filepath where the logs are written
	outfile = './TSP_logs/TSP_@ALG@_@N@_LOG.txt'

	# Initializes a 2D array of shape 100 x 2
	# The cities are on a 2D map, and the points define the x and y coordinates of the cities
	points = [[0 for _ in xrange(2)] for _ in xrange(N)]
	for i in range(0, len(points)):
		points[i][0] = random.nextDouble()
		points[i][1] = random.nextDouble()

	ef = TravelingSalesmanRouteEvaluationFunction(points)
	odd = DiscretePermutationDistribution(N)
	nf = SwapNeighbor()
	mf = SwapMutation()
	cf = TravelingSalesmanCrossOver(ef)
	gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)


	# Randomized Hill Climbing
	# print("Running hill climbing...")
	# max_iters = 40000
	# for t in range(num_trials):
	# 	# For each trial, open a file to log the data
	# 	fname = outfile.replace('@ALG@', 'RHC').replace('@N@', str(t+1))
	# 	with open(fname, 'w') as f:
	# 		f.write('iterations,fitness,time,fevals\n')
	#
	# 	# Reinitialize problems per trial
	# 	ef = TravelingSalesmanRouteEvaluationFunction(points)
	# 	odd = DiscretePermutationDistribution(N)
	# 	nf = SwapNeighbor()
	# 	hcp = GenericHillClimbingProblem(ef, odd, nf)
	# 	rhc = RandomizedHillClimbing(hcp)
	#
	# 	# Iterate, logging every 10 steps
	# 	fit = FixedIterationTrainer(rhc, 10)
	# 	cum_elapsed = 0
	# 	for i in range(0, max_iters + 1, 10):
	# 		# Train 10 iterations and clock time
	# 		start = clock()
	# 		fit.train()
	# 		elapsed = time.clock() - start
	# 		cum_elapsed += elapsed
	#
	# 		fevals = ef.fevals                  # Number of function evaluations
	# 		score = ef.value(rhc.getOptimal())  # Fitness score
	# 		ef.fevals -= 1                      # Reducing by one because getting the score counts as an eval
	#
	# 		# Logging
	# 		st = '{},{},{},{}\n'.format(i, score, cum_elapsed, fevals)
	# 		# print st
	# 		with open(fname, 'a') as f:
	# 			f.write(st)


	# Simulated Annealing
	# print("Running simulated annealing...")
	# max_iters = 40000
	# for t in range(num_trials):
	# 	# Each iteration, the temp is cooled by the eqn T *= cooling_mult
	# 	for cooling_mult in [0.15, 0.35, 0.55, 0.75, 0.95]:
	# 		# For each trial, open a file to log the data
	# 		fname = outfile.replace('@ALG@','SA{}'.format(cooling_mult)).replace('@N@',str(t+1))
	# 		with open(fname,'w') as f:
	# 			f.write('iterations,fitness,time,fevals\n')
	#
	# 		# Reinitialize problems per trial
	# 		ef = TravelingSalesmanRouteEvaluationFunction(points)
	# 		odd = DiscretePermutationDistribution(N)
	# 		nf = SwapNeighbor()
	# 		hcp = GenericHillClimbingProblem(ef, odd, nf)
	# 		sa = SimulatedAnnealing(1E10, cooling_mult, hcp)
	#
	# 		# Iterate, logging every 10 steps
	# 		fit = FixedIterationTrainer(sa, 10)
	# 		cum_elapsed = 0
	# 		for i in range(0, max_iters + 1, 10):
	# 			# Train 10 iterations and clock time
	# 			start = clock()
	# 			fit.train()
	# 			elapsed = time.clock()-start
	# 			cum_elapsed += elapsed
	#
	# 			fevals = ef.fevals  # Number of function evaluations
	# 			score = ef.value(sa.getOptimal())  # Fitness score
	# 			ef.fevals -= 1  # Reducing by one because getting the score counts as an eval
	#
	# 			# Logging
	# 			st = '{},{},{},{}\n'.format(i, score, cum_elapsed, fevals)
	# 			# print st
	# 			with open(fname, 'a') as f:
	# 				f.write(st)


	# Genetic Algorithms
	# print("Running genetic algorithms...")
	# max_iters = 10000
	# for t in range(num_trials):
	# 	# pop: number in population
	# 	# mate: number in population to mate
	# 	# mutate: number in population to mutate
	# 	for pop in [100, 500, 1000]:
	# 		mate = int(0.5 * pop)
	# 		mutate = int(0.1 * pop)
	# 		# For each trial, open a file to log the data
	# 		fname = outfile.replace('@ALG@','GA{}'.format(pop)).replace('@N@', str(t+1))
	# 		with open(fname, 'w') as f:
	# 			f.write('iterations,fitness,time,fevals\n')
	#
	# 		# Reinitialize problems per trial
	# 		ef = TravelingSalesmanRouteEvaluationFunction(points)
	# 		odd = DiscretePermutationDistribution(N)
	# 		mf = SwapMutation()
	# 		cf = TravelingSalesmanCrossOver(ef)
	# 		gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
	# 		ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
	#
	# 		# Iterate, logging every 10 steps
	# 		fit = FixedIterationTrainer(ga, 10)
	# 		cum_elapsed = 0
	# 		for i in range(0, max_iters + 1, 10):
	# 			# Train 10 iterations and clock time
	# 			start = clock()
	# 			fit.train()
	# 			elapsed = time.clock()-start
	# 			cum_elapsed += elapsed
	#
	# 			fevals = ef.fevals                  # Number of function evaluations
	# 			score = ef.value(ga.getOptimal())   # Fitness score
	# 			ef.fevals -= 1                      # Reducing by one because getting the score counts as an eval
	#
	# 			# Logging
	# 			st = '{},{},{},{}\n'.format(i, score, cum_elapsed, fevals)
	# 			# print st
	# 			with open(fname, 'a') as f:
	# 				f.write(st)


	#MIMIC
	print("Running MIMIC...")
	fill = [N] * N
	ranges = array('i', fill)
	max_iters = 10000
	for t in range(num_trials):
		for samples in [300, 500, 700]:
			keep = int(0.5 * samples)
			# For each trial, open a file to log the data
			fname = outfile.replace('@ALG@', 'MIMIC{}'.format(samples)).replace('@N@', str(t+1))
			with open(fname, 'w') as f:
				f.write('iterations,fitness,time,fevals\n')

			# Reinitialize problems per trial
			ef = TravelingSalesmanSortEvaluationFunction(points)
			odd = DiscreteUniformDistribution(ranges)
			df = DiscreteDependencyTree(0.1, ranges)
			pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
			mimic = MIMIC(samples, keep, pop)

			# Iterate, logging every 10 steps
			fit = FixedIterationTrainer(mimic, 10)
			cum_elapsed = 0
			for i in range(0, max_iters, 10):
				# Train 10 iterations and clock time
				start = clock()
				fit.train()
				elapsed = time.clock()-start
				cum_elapsed += elapsed

				fevals = ef.fevals                      # Number of function evaluations
				score = ef.value(mimic.getOptimal())    # Fitness score
				ef.fevals -= 1                          # Reducing by one because getting the score counts as an eval

				# Logging
				st = '{},{},{},{}\n'.format(i, score, cum_elapsed, fevals)
				# print st
				with open(fname, 'a') as f:
					f.write(st)
