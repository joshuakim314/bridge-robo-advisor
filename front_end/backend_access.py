import numpy as np
import pandas as pd
from datetime import datetime
import time

import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

conn = psycopg2.connect(
    host='database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    port=5432,
    user='postgres',
    password='capstone',
    database='can2_etfs'
)


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

def push_new_user(connection, user_array):

    #print('connected to postgres db ...')

    connection.autocommit = True
    cursor = connection.cursor()

    #print('connected')

    #print('formatting dict to_record')

    sql = '''INSERT INTO clients (first, last, email, password, risk, control)
    VALUES(%s, %s, %s, %s, %s, %s)'''
    cursor.execute(sql, user_array)

    #print("end")

    return True

def pull_user_data(connection, email, pswrd):
    connection.autocommit = True
    cursor = connection.cursor()
    sql = f'''SELECT * FROM clients WHERE (email like '{email}') AND (password like '{pswrd}')'''
    cursor.execute(sql)
    selected = cursor.fetchall()
    cursor.close()
    #print("end")
    df = convert_db_fetch_to_df(selected, column_names=['First', 'Last', 'Email', 'Password', 'Risk', 'Control'])
    dct = {
        'First': df['First'][0],
        'Last': df['Last'][0],
        'Email': df['Email'][0],
        'Password': df['Password'][0],
        'Risk': df['Risk'][0],
        'Control': df['Control'][0]
    }
    return dct

def update_risk_control(connection, email, pswrd, risk, control):
    connection.autocommit = True
    cursor = connection.cursor()

    sql = f'''UPDATE clients SET risk = {risk}, control = {control} WHERE (email like'{email}') AND (password like '{pswrd}')'''

    cursor.execute(sql)

    # print("end")

    return True

def add_transaction(connection, email, asset, ammount):
    now = time.localtime()
    now = time.strftime('%Y-%m-%d %H:%M:%S', now)
    transact_row = np.array([now, email, asset, ammount])

    connection.autocommit = True
    cursor = connection.cursor()

    sql = '''INSERT INTO transactions (datetime, email, ticker, amount) VALUES(%s, %s, %s, %s)'''
    cursor.execute(sql, transact_row)

    return True

def wipe_table(connection, table):
    connection.autocommit = True
    cursor = connection.cursor()

    sql = f'''TRUNCATE TABLE {table}'''
    cursor.execute(sql)

    return True

def get_portfolio_data(connection, email):
    connection.autocommit = True
    cursor = connection.cursor()

    sql = f'''SELECT ticker, SUM(amount) FROM transactions WHERE (email like '{email}') AND (ticker like 'cash') GROUP BY ticker'''
    cursor.execute(sql)
    selected = cursor.fetchall()
    cursor.close()
    # print("end")
    df = convert_db_fetch_to_df(selected, column_names=['Ticker', 'Amount'])
    ret_dict = df.to_dict('records')
    return ret_dict

def get_past_deposits(connection, email):
    connection.autocommit = True
    cursor = connection.cursor()
    sql = f'''SELECT * FROM transactions WHERE (email like '{email}') AND (ticker like 'cash') AND (NOT amount = 0)'''

    cursor.execute(sql)
    selected = cursor.fetchall()
    cursor.close()
    # print("end")
    df = convert_db_fetch_to_df(selected, column_names=['Date Time', 'Email', 'Ticker', 'Amount'])
    df = df[['Date Time', 'Amount']]
    df['Amount'] = df['Amount'].div(100)
    df = df.sort_values(by = 'Date Time', ascending = False)
    df["Date Time"] = df["Date Time"].dt.strftime("%A %B %d, %Y - %H:%M:%S")
    print(df)

    print([{"name": i, "id": i} for i in df.columns])
    ret_dict = df.to_dict('records')
    return ret_dict

#get_portfolio_data(conn, 'aidanjones55@gmail.com')

#wipe_table(conn, 'clients')
#wipe_table(conn, 'transactions')
#add_transaction(conn, 'aidanjones55@gmail.com', 'cash', 1000)

