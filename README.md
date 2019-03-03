### CS 7641 Assignment 2 - Randomized Optimization

Code and data can be found at: https://github.com/isw4/ML02-Randomized_Optimization

ABAGAIL library modified and built from: https://github.com/pushkar/ABAGAIL  
Code adapted from: https://github.com/JonathanTay/CS-7641-assignment-2

## Directories
data/       Contains the raw and cleaned csv data files, along with some description of each set  
graphs/     Contains graphs that are output by the various experiments  
ABAGAIL/    Contains the ABAGAIL src and jar files (minor edits made from the original source)  
COUNT_logs/ Contains the csv logs of the count ones problem optimizations  
KNAP_logs/  Contains the csv logs of the knapsack problem optimizations  
TSP_logs/   Contains the csv logs of the traveling salesman problem optimizations  
NN_logs/    Contains the csv logs of the neural network optimization  


## Setup to run the experiments
Experiments are run on jython, which is a Java implementation of Python 2.7

1)  Make sure to have JDK installed (jdk11: https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html)

2)  Make sure to have jython installed (https://www.jython.org/)

3)  Add the *absolute* path to the ABAGAIL jar file to "ABAGAIL_absolute_path.txt". e.g.
    ~~~
    C:/Documents/ML02-Randomized_Optimization/ABAGAIL/ABAGAIL.jar
    ~~~


## Setup to plot the data from the logs
Data processing and plotting is run on Python 3.6, since matplotlib does not have support for Python 2

1)  Make sure to have Conda installed

2)  Install the conda environment:
    ~~~
    conda env create -f environment.yml
    ~~~

3)  Activate the environment:
    If using Windows, open the Anaconda prompt and enter:
    ~~~
    activate Homework2
    ~~~

    If using Mac or Linux, open the terminal and enter:
    ~~~
    source activate Homework2
    ~~~


## Running the datasets
#### Data processing:
In Conda env (Python 3):
~~~
python data_processing.py
~~~

#### Running Experiments:
In Jython (Python 2):  
To run the neural network training experiments:
~~~
jython train_nn.py
~~~

To run the 3 optimization problems:
~~~
jython problem_countones.py
jython problem_knapsack.py
jython problem_tsp.py
~~~

#### Plotting
In Conda env (Python 3):
Does not automatically show the plots, but will save to the graphs folder
~~~
python plot_nn.py
python plot_countones.py
python plot_knapsack.py
python plot_tsp.py
~~~