# Statistical Analysis on IP data 

Content includes

* Code to download clickstream data from database and manipulate into a readable format.
* Match with external data for each school. 
* Conduct statistical analysis on the data. 

## Deploying

After the raw csv files have been provided, make sure to copy them into data/raw. The following operations can be done in sequence:

1. Extracting and manipulating clickstream data (if raw csv data needs to be changed) 
2. Organising external data (extracting and creating a dataframe for external data)
3. Combining external data with internal data and performing statistical analysis. 

Alternatively, anyone of the notebooks can be run independantly provided that the required csv files are present. 

### 1. Extracting Clickstream Data

The clickstream data can be extracted from the IP database by running 

```
python retrieve_data.py
```

in the IP server. This file requires SQL_helper.py and configDB.py where the latter needs to be completed with the user_id, database name and password. The file saves the csv files 

* teacher_group.csv (teacher clickstream data)
* user_group.csv (student clickstream data)
* user_details.csv (linking each user (student or teacher) to a particular school). 

This data is comprised of the clickstream data of each student and teacher within a certain period of time. Each of these users are linked to a school as given by 'user_details.csc':

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

Any additional indices can be incorporated and tested. 

```
all_data_joined.ipynb
```


