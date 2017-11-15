# Research work on IP platform

## Getting Started

### Prerequisites

You will require git which can be downloaded from https://git-scm.com/downloads. 

Afterwards, clone the repository into a destination of your choice.

```
git clone git@github.com:KeepFloyding/IP-research.git
```
You will need the following python packages which automatically come with Anaconda (https://anaconda.org/anaconda/python)

* sklearn
* pandas
* numpy
* scipy
* matplotlib

You will also need to use Jupyter to read the IPython notebook. 

Further to this you will also need the csv that contains the data. When this has been provided to you, please put it in the data folder of this repository after you have clone it. 

## Deploying

After you have been provided with the relevant csv files, you can begin running the python notebooks.

### Extracting Clickstream Data

For this you will need access to the IP database. This can be entered into the config file in settings. After this, the file X.py can be run on the server to extract data with a SQL query and manipulate it into a usable format. 

### Cleaning Clickstream Data

Since the files retrieve the actions done by each user split according to teacher and student roles, a python notebook is provided that joins the 2 dataframes together to create a clean dataframe.

### Combining External Data

The external data that is provided needs to be combined. This operation is performed by the X python notebook. 

### Combining External Data with Clickstream Data

This one is done in the python notebook X.py. 

Here we also conduct the main analysis.

