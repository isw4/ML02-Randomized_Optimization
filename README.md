### CS 7641 Assignment 2 - Randomized Optimization

Code and data can be found at: https://github.com/isw4/ML02-Randomized_Optimization

ABAGAIL library modified and built from: https://github.com/pushkar/ABAGAIL

## Directories
data/       Contains the raw and cleaned csv data files, along with some description of each set  
graphs/     Contains graphs that are output by the various experiments  
ABAGAIL/    Contains the ABAGAIL src and jar files (minor edits made from the original source)
COUNT_logs/ Contains the logs of the count ones problem optimizations in csv  
KNAP_logs/  Contains the logs of the knapsack problem optimizations in csv  
TSP_logs/   Contains the logs of the traveling salesman problem optimizations in csv  


## Setup to run the experiments

1)  Make sure to have JDK installed (jdk11: https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html)

2)  Make sure to have jython installed (https://www.jython.org/)


## Setup to plot the data from the logs

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
