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

def prepare_df(df,start_date, dt, date_feature, group_array):

    df[date_feature] = pd.to_datetime(df[date_feature])
    df[start_date] = pd.to_datetime(df[start_date])
    df['event_leg'] = np.floor((df[date_feature] - df[start_date]).dt.total_seconds()/dt)

    #df.drop('date_created',axis=1)
    df_grp = df.groupby(group_array).count();
    df_grp = df_grp.xs(date_feature, axis=1, drop_level=True)

    df_grp = df_grp.unstack().fillna(0)
    #df_grp = df_grp.reset_index()

    return df_grp

def prepare_df_new(df,start_date, dt, date_feature, group_array):

    df[date_feature] = pd.to_datetime(df[date_feature])
    df[start_date] = pd.to_datetime(df[start_date])
    df['last_seen'] = pd.to_datetime(df['last_seen'])
    df['event_leg'] = np.floor((df[date_feature] - df[start_date]).dt.total_seconds()/dt)
    df['churn_leg'] = np.floor((df['last_seen'] - df[start_date]).dt.total_seconds()/dt)

    # Finding churn leg for each user
    df_churn = df.groupby('user_id')['churn_leg'].max()
    
    #df.drop('date_created',axis=1)
    df_grp = df.groupby(group_array).count();
    df_grp = df_grp.xs(date_feature, axis=1, drop_level=True)

    df_grp = df_grp.unstack().fillna(0)
    #df_grp = df_grp.reset_index()

    return df_grp, df_churn

def prepare_df_all(df,date_feature, group_array,cut_off):

    df[date_feature] = pd.to_datetime(df[date_feature])
    df = df[df[date_feature]> pd.to_datetime(cut_off)]

    df_grp = df.groupby(group_array).count();
    df_grp = df_grp.xs(date_feature, axis=1, drop_level=True)
    df_grp = df_grp.unstack().fillna(0)
  

    return df_grp

