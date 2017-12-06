"""
Python libraries for SQL commands. 
The following packages are needed:
- pandas
- sqlalchemy
- psycopg2 (python wrapper for PostGreSQL)

Script should be run in conjunction with Python 2.7

Created on Wednesday 20th September 2017
@author: Andris Piebalgs 
Last updated: -
"""

import sqlalchemy
import pandas as pd
import psycopg2
import time
import numpy as np
from configDB import config

# Read a SQL command provided as a string and return a datafram with the results


def read_SQL(q):

    # Connecting to database
    engine = sqlalchemy.create_engine("postgresql+psycopg2://",connect_args={"database": config['database'],"user": config['user'], "host": "localhost","password": config['password']})
    con = engine.connect()

    # Reading result into pandas dataframe
    start_time = time.time()
    df = pd.read_sql(q,con)
    print "Time taken is {0} minutes".format(round((time.time()-start_time)/60,2))

    return df

# Generate string from SQL file
def create_string(file_name):

    with open(file_name) as myfile:
        q = myfile.read()

    return q

def prepare_df(df,date_feature, group_array,start_date,end_date):

    # Convert to datetime format
    df[date_feature] = pd.to_datetime(df[date_feature])

    # Select data from dates according to start_date and end_date
    df = df[df[date_feature]>= pd.to_datetime(start_date)]
    df = df[df[date_feature]<= pd.to_datetime(end_date)]

    # Group according to categories
    df_grp = df.groupby(group_array).count();

    # Unpack the indices into columns
    df_grp = df_grp.xs(date_feature, axis=1, drop_level=True)
    df_grp = df_grp.unstack().fillna(0)
  
    # Return the dataframe
    return df_grp

