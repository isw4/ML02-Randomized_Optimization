import sys
import problem_tsp
import problem_countones
import problem_knapsack

"""
Command line arguments:
tsp         Traveling Salesman Problem
countones   Count Ones Problem
knapsack    Knapsack Problem
"""
if __name__ == '__main__':
	command = sys.argv[1]

	if command == 'tsp':
		problem_tsp.optimize()

	elif command == 'countones':
		problem_countones.optimize()

	elif command == 'knapsack':
		problem_knapsack.optimize()

	else:
		print("Invalid command")

