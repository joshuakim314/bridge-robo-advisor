import numpy as np
import pandas as pd
import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)


def convert_db_fetch_to_df(fetched, column_names=None):
    """
    This method converts the cursor.fetchall() output of SELECT query into a Pandas dataframe.
    :param fetched: the output of SELECT query
    :type fetched: list of row tuples
    :param column_names: column names to use for the dataframe
    :type column_names: list of column names
    :return: converted dataframe
    :rtype: Pandas dataframe
    """
    return pd.DataFrame(fetched, columns=column_names)


conn = psycopg2.connect(
    host='database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    port=5432,
    user='postgres',
    password='capstone',
    database='can2_etfs'
)
print('connected to postgres db')

conn.autocommit = True
cursor = conn.cursor()
sql = '''SELECT * FROM prices ORDER BY date ASC, ticker ASC LIMIT 10'''
cursor.execute(sql)
selected = cursor.fetchall()
print(convert_db_fetch_to_df(selected))
cursor.close()
print("end")
