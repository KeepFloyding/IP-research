# Research work on IP platform

Python code and notebooks for IP team. 

Content includes

* Code to download clickstream data from database and manipulate into a readable format.
* Match with external data for each school. 
* Conduct statistical analysis on the data. 

## Getting Started

### Prerequisites

You will require git which can be downloaded from https://git-scm.com/. 
Clone the repository into a destination of your choice with:

```
git clone git@github.com:KeepFloyding/IP-research.git
```

You will also need the following python packages which automatically come with Anaconda (https://anaconda.org/anaconda/python)

* sklearn
* pandas
* numpy
* scipy
* matplotlib
* Jupyter notebook

In addition to this, you will need the original raw csv data which will be provided to you. Simply put this in the data/raw folder. 

### Testing 

To check if you have properly installed all the required repositories, open the notebook "test_nb" in the test folder and run. If there are no errors, then all packages have been correctly installed. 

## Deploying

After the raw csv files have been provided, make sure to copy them into data/raw. The following operations can be done in sequence:

1. Extracting and manipulating clickstream data (if raw csv data needs to be changed) 
2. Organising external data (extracting and creating a dataframe for external data)
3. Combining external data with internal data and performing statistical analysis. 

Alternatively, anyone of the notebooks can be run independantly provided that the required csv files are present. 

### 1. Extracting Clickstream Data

Extracting clickstream data from the IP database with a SQL query. This has to be done in the IP server and requires access to the database. 

The result are 2 csv files that contain the clickstream data for teachers and students. This is then cleaned, joined and grouped by school in the notebook X. 

### 2. Organising External Data

The external data that is provided needs to be combined. This operation is performed by the X python notebook. 

### Combining External Data with Clickstream Data

This one is done in the python notebook X.py. 

Here we also conduct the main analysis.

