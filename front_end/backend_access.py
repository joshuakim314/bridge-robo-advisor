import numpy as np
import pandas as pd
from datetime import datetime
import time
import plotly.express as px

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

    sql = '''INSERT INTO clients (first, last, email, password, risk, control, horizon, return, max_card)
    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
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
    df = convert_db_fetch_to_df(selected, column_names=['First', 'Last', 'Email', 'Password', 'Risk', 'Control', 'Horizon', 'Return', 'Max'])
    dct = {
        'First': df['First'][0],
        'Last': df['Last'][0],
        'Email': df['Email'][0],
        'Password': df['Password'][0],
        'Risk': df['Risk'][0],
        'Control': df['Control'][0],
        'Horizon': df['Horizon'][0],
        'Return': df['Return'][0],
        'Max': df['Max'][0]
    }
    return dct

def update_risk_control(connection, email, pswrd, risk, control, horizon, ret, max_card):
    connection.autocommit = True
    cursor = connection.cursor()

    sql = f'''UPDATE clients SET risk = {risk}, control = {control}, horizon={horizon}, return={ret}, max_card={max_card} WHERE (email like'{email}') AND (password like '{pswrd}')'''

    cursor.execute(sql)

    # print("end")

    return True

def add_transaction(connection, email, asset, ammount, comment=''):
    now = time.localtime()
    date = time.strftime('%Y-%m-%d', now)

    now = time.strftime('%Y-%m-%d %H:%M:%S', now)
    transact_row = np.array([now, email, asset, ammount, date, comment])

    connection.autocommit = True
    cursor = connection.cursor()

    sql = '''INSERT INTO transactions (datetime, email, ticker, amount, date, comment) VALUES(%s, %s, %s, %s, %s, %s)'''
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

def get_past_deposits_brief(connection, email):
    connection.autocommit = True
    cursor = connection.cursor()
    sql = f'''SELECT * FROM transactions WHERE (email like '{email}') AND (ticker like 'cash')'''

    cursor.execute(sql)
    selected = cursor.fetchall()
    cursor.close()
    # print("end")
    df = convert_db_fetch_to_df(selected, column_names=['Date Time', 'Email', 'Ticker', 'Amount', 'Date', 'Comment'])
    df = df[['Date Time', 'Amount']]
    df['Amount'] = df['Amount'].div(100)
    df = df.sort_values(by='Date Time', ascending=False)
    df["Date Time"] = df["Date Time"].dt.strftime("%D")
    #print(df)
    df = df.groupby(["Date Time"]).Amount.sum().reset_index()
    # print(df)
    # print([{"name": i, "id": i} for i in df.columns])
    ret_dict = df.to_dict('records')
    #print(ret_dict)
    return ret_dict

def get_past_deposits(connection, email):
    connection.autocommit = True
    cursor = connection.cursor()
    sql = f'''SELECT * FROM transactions WHERE (email like '{email}') AND (ticker like 'cash')'''

    cursor.execute(sql)
    selected = cursor.fetchall()
    cursor.close()
    # print("end")
    df = convert_db_fetch_to_df(selected, column_names=['Date Time', 'Email', 'Ticker', 'Amount', 'Date', 'Comment'])
    df = df[['Date Time', 'Amount', 'Comment']]
    df['Amount'] = df['Amount'].div(100)
    df = df.sort_values(by = 'Date Time', ascending = False)
    #print(df)
    df["Date Time"] = df["Date Time"].dt.strftime("%A %B %d, %Y - %H:%M:%S")
    #print(df)
    #print([{"name": i, "id": i} for i in df.columns])
    ret_dict = df.to_dict('records')
    return ret_dict

def transact_stock(connection, email, stock, ammount):
    now = time.localtime()
    now = time.strftime('%Y-%m-%d %H:%M:%S', now)

    connection.autocommit = True
    cursor = connection.cursor()
    #Grab Max Date
    sql = f"""
        SELECT MAX(date) FROM prices WHERE ticker like '{stock}'
        """
    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['date'])
    max_date = df['date'][0]
    #print(max_date)

    #Calculate Stocks Cost and store as value_of_shares
    sql = f"""
    SELECT adj_close FROM prices WHERE date = '{max_date}' AND ticker like '{stock}'
    """

    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['adj_close'])
    stock_value = int(100*ammount*df['adj_close'][0])+1
    #print(stock_value)
    if ammount > 0:
        transact_cash_statement = f'Bought {ammount} shares of {stock} for ${stock_value/100:,.2f}'
        transact_stock_statement = f'Bought {ammount} shares of {stock}'
    else:
        transact_cash_statement = f'Sold {ammount} shares of {stock} ${stock_value/100:,.2f}'
        transact_stock_statement = f'Sold {ammount} shares of {stock}'
    #Transact Cash
    add_transaction(connection, email, 'cash', -1*stock_value, transact_cash_statement)

    #Transact Stock
    add_transaction(connection, email, stock, ammount, transact_stock_statement)

    return True
    #Transact Stock

def get_portfolio_value_by_date(connection, email):
    connection.autocommit = True
    cursor = connection.cursor()
    # Grab Start Date of User Profile
    sql = f"""
            SELECT MIN(date) FROM transactions WHERE email like '{email}'
            """
    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['date'])
    min_account_date = df['date'][0]

    sql = f'''
    DROP VIEW IF EXISTS data5;
    DROP VIEW IF EXISTS data4;
    DROP VIEW IF EXISTS data3;
    DROP VIEW IF EXISTS data2;
    
    CREATE VIEW data2 AS
        SELECT * FROM transactions WHERE email like '{email}';
    
    CREATE VIEW data3 AS
        SELECT date, ticker, sum(amount) 
        OVER (PARTITION BY ticker ORDER BY date) AS cum_amt
        FROM data2
        ORDER BY date, ticker;
        
    CREATE VIEW data4 AS
        SELECT date, ticker, cum_amt FROM data3 GROUP BY date, ticker, cum_amt;

    SELECT * FROM data4
    '''
#
    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'cumAmount'])

    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    list_tickers = list(set(df['ticker'].to_list()))
    #print(list_tickers)
    if len(list_tickers) > 1:
        string_tickers = '('
        for tick in list_tickers[:-1]:
            string_tickers = string_tickers + (f"'{tick}', ")
        string_tickers = string_tickers + f"'{list_tickers[-1]}')"
    else:
        string_tickers = f"('{list_tickers[0]}')"


    #Get price info for unique tickers, between min and max date


    ### Start hitting prices tables ###
    sql = f'''
        SELECT date, ticker, adj_close FROM canadianetfs WHERE (date BETWEEN '{min_date}' AND '{max_date}') AND (ticker IN {string_tickers})
        '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    cadetfs = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
            SELECT date, ticker, adj_close FROM americanetfs WHERE (date BETWEEN '{min_date}' AND '{max_date}') AND (ticker IN {string_tickers})
            '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    ametfs = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
            SELECT date, ticker, adj_close FROM spy WHERE (date BETWEEN '{min_date}' AND '{max_date}') AND (ticker IN {string_tickers})
            '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    spy = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
            SELECT date, ticker, adj_close FROM tsx60 WHERE (date BETWEEN '{min_date}' AND '{max_date}') AND (ticker IN {string_tickers})
            '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    tsx60 = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    #print(tsx60)

    dfpx = cadetfs.append(ametfs, ignore_index=True)
    dfpx = dfpx.append(spy, ignore_index=True)
    dfpx = dfpx.append(tsx60, ignore_index=True)
    #print(dfpx)

    cursor.close()

    dfpx['date'] = pd.to_datetime(dfpx['date'])
    df.set_index(['date', 'ticker'], inplace=True)

    unique_tickers = df.index.unique(level='ticker')
    date_range = pd.DatetimeIndex(pd.date_range(start=min_date, end=max_date, freq="D"))

    ticker_date_index = (
        pd.MultiIndex
            .from_product(
            iterables=[date_range, unique_tickers],
            names=['date', 'ticker']
        )
    )

    df = df.reindex(ticker_date_index)
    df = df.sort_index(level=1).ffill().reindex(df.index)

    df = pd.merge(
        df,
        dfpx,
        how="left",
        on=['date', 'ticker'],
    )

    mask = (df['ticker'] == 'cash')
    df['adj_close'][mask] = 0.01

    df = df.set_index(['date', 'ticker'])
    df = df.sort_index(level=1).ffill().reindex(df.index)
    df.fillna(0, inplace=True)

    df['value'] = df['cumAmount']*df['adj_close']
    #print(df)
    df = df.round(2)

    df = df.reset_index(level=1)
    df = df.groupby(['date']).sum().reset_index()[['date','value']]

    final_value = {'portfolio_value': round(df['value'].to_list()[-1], 2)}

    data = df.round(2).to_json(date_format='iso', orient='split')

    return data, final_value

def get_all_tickers(connection):
    connection.autocommit = True
    cursor = connection.cursor()

    sql = f'''SELECT DISTINCT(ticker) FROM spy'''

    cursor.execute(sql)
    selected = cursor.fetchall()
    spy = convert_db_fetch_to_df(selected, column_names=['ticker'])

    sql = f'''SELECT DISTINCT(ticker) FROM tsx60'''

    cursor.execute(sql)
    selected = cursor.fetchall()
    tsx60 = convert_db_fetch_to_df(selected, column_names=['ticker'])

    dfpx = spy.append(tsx60, ignore_index=True)
    cursor.close()

    ret_dict = {'stocks': dfpx['ticker'].to_list()}
    # print("end")
    return ret_dict

#print(get_all_tickers(conn))


#get_portfolio_value_by_date(conn, 'aidanjones55@gmail.com')

#transact_stock(conn, 'aidanjones55@gmail.com', 'XIU.TO', 2)
#wipe_table(conn, 'clients')
#wipe_table(conn, 'transactions')
#add_transaction(conn, 'aidanjones55@gmail.com', 'cash', 1000)


#add_transaction(conn, 'aidanjones55@gmail.com', 'cash', 0, 'Initialize Account')
#get_past_deposits(conn, 'aidanjones55@gmail.com')
#print(pull_user_data(conn, 'aidanjones55@gmail.com', 'admin'))
