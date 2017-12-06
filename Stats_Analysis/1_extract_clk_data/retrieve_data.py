# Extract the clickstream data from database
# The python file along with the SQL_helper and configDB file needs to be run in the IP server.

from SQL_helper import *

# Selecting the dates to look at
start_date = '2014-08-01'
end_date   = '2016-06-01'

# Find the user clickstream data
save_list = ['user_group.csv','teacher_group.csv']
sql_list = ['SQL_commands/find_user_data.sql','SQL_commands/find_teacher_data.sql']

for sql_dir, save_dir in zip(sql_list,save_list):

	# Print status update
	print('Performing query in',sql_dir)

	# Reading the SQL file
	q = create_string(sql_dir)
	df = read_SQL(q)

	# Manipulating the dataframe
	df_grp = prepare_df(df,'date_of_event', ['user_id','event_type'], start_date, end_date)

	# Saving file to csv
	df_grp.to_csv(save_dir)
