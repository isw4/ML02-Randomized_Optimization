import pandas as pd
import numpy as np


def aggregate_wine_labels(Y):
	new_Y = Y.copy()
	new_Y[Y >= 6] = 1  # Good wines
	new_Y[Y < 6] = 0  # Bad wines
	return new_Y


def normalize(X):
	# Mean normalization
	return (X - X.mean()) / X.std()


def process(filepath_input, filepath_output):
	"""
	reads csv file and removes data instances with empty features, normalizes, and aggregates labels
	then outputs them into a csv file with ',' delimiter

	:param filepath_input: str, filepath of the input csv file
	:param filepath_output: str, filepath of the output csv file after cleaning
	"""
	df = pd.read_csv(filepath_input, sep=';')

	# Removes all rows that have empty cells
	row_no_nan = df.notna().all(axis=1)
	df = df[row_no_nan]

	df.iloc[:, :-1] = normalize(df.iloc[:, :-1])
	df.iloc[:, -1] = aggregate_wine_labels(df.iloc[:, -1])

	df.to_csv(filepath_output, index=False)


if __name__ == "__main__":
	raw_data_path = './data/raw-winequality-red.csv'
	data_path = './data/processed-wine-equality-red.csv'
	process(raw_data_path, data_path)