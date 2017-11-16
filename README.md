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

The clickstream data can be extracted from the IP database. This will be documented later. 

The raw clickstream data is comprised of the clickstream data of each student and teacher within a certain period of time. Each of these users are linked to a school as given in a seperate csv:

* teacher_group.csv (teacher clickstream data)
* user_group.csv (student clickstream data)
* user_details (linking each user (student or teacher) to a particular school). 

The python notebook 

```
1_combine_clk_data.ipynb 
```

combines the data from these CSVs and cleans them in suitable format for analysis. The final dataframe is saved as 'school_clk_data.csv'.


### 2. Organising External Data

The external raw data is comprised of 2 seperate csv files

* exam results for each school (A2 results 2014-16.csv)
* external indicators for each school (external_outer_indicators.csv)

These 2 csv files are cleaned, combined and saved as a seperate csv (school_ext_data.csv) in the python notebook:

```
2_extract_ext_data.ipynb
```

### 3. Combining External Data with Clickstream Data

The 'all_data_joined.ipynb' notebook does the following:

* Combines external and internal dataframes by school
* Create new features and indices
* Perform statistical analysis

Any additional indices can be incoporated and tested. 
