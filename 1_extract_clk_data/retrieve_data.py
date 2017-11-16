# Extract the clickstream data from database
# Simply run the python file in the right location

from SQL_helper import *

cut_off = '2017-09-01'

count = 0;
save_list = ['user_group.csv','teacher_group.csv']

for str_dir in ['SQL_commands/find_user_data.sql','SQL_commands/find_teacher_data.sql'] :

	q = read_SQL(str_dir)
	df_grp = prepare_df_all(df,'date_of_event', ['id','event_type'],cut_off)

	df_grp.to_csv(save_list[count])
	count += 1
