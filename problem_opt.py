import sys
import itertools
import problem_tsp
import problem_fourpeaks
import problem_knapsack

"""
Command line arguments:
tsp         Traveling Salesman Problem
peaks       4 Peaks Problem
knapsack    Knapsack Problem
"""
if __name__ == '__main__':
	command = sys.argv[1]

	if command == 'tsp':
		problem_tsp.optimize()

	elif command == 'peaks':
		pass

	elif command == 'knapsack':
		problem_knapsack.optimize()

	else:
		print("Invalid command")

