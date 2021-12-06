import numpy as np
import pandas as pd
from datetime import datetime
import time
import plotly.express as px

import psycopg2.extensions
from dateutil.relativedelta import relativedelta

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

    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
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
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    #print('STOCK TIME')
    #print(now)
    connection.autocommit = True
    cursor = connection.cursor()
    #Grab Max Date
    sql = f"""
        SELECT MAX(date) FROM americanetfs
        """
    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['date'])
    #print(df)
    max_date = df['date'][0]
    #print(max_date)

    #Calculate Stocks Cost and store as value_of_shares
    sql = f"""SELECT adj_close FROM americanetfs WHERE (date = '{max_date}') AND (ticker like '{stock}')
    """

    cursor.execute(sql)
    selected = cursor.fetchall()
    cadetfs = convert_db_fetch_to_df(selected, column_names=['adj_close'])

    sql = f"""SELECT adj_close FROM spy WHERE (date = '{max_date}') AND (ticker like '{stock}')
    """

    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['adj_close'])

    df = df.append(cadetfs, ignore_index=True)

    stock_value = int(100*ammount*df['adj_close'][0])+1
    #print(stock_value)
    if ammount > 0:
        transact_cash_statement = f'Bought {ammount} shares of {stock} for ${stock_value/100:,.2f}'
        transact_stock_statement = f'Bought {ammount} shares of {stock}'
    else:
        transact_cash_statement = f'Sold {abs(ammount)} shares of {stock} for ${stock_value/100:,.2f}'
        transact_stock_statement = f'Sold {abs(ammount)} shares of {stock}'
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
    #print(df)

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
    df_start = df.loc[slice(min_date), slice(None), :].fillna({'cumAmount':0})
    df.update(df_start)

    df = df.sort_index(level=1).ffill().reindex(df.index)

    #print(df.tail(20))

    #print(dfpx.tail(20))
    #print(df.reset_index().tail(20))
    df = pd.merge(
        df,
        dfpx,
        how="left",
        on=['date', 'ticker'],
    )
    mask = (df['ticker'] == 'cash')
    df['adj_close'][mask] = 0.01
    #print(df.tail(20))
    #print(df.reindex(ticker_date_index).tail(20))
    df = df.set_index(['date', 'ticker'])
    #print(df.tail(20))
    df = df.sort_index(level=1).ffill().reindex(df.index)
    #print(df.tail(20))
    df.fillna(0, inplace=True)

    df['value'] = df['cumAmount']*df['adj_close']
    #print(df)
    df = df.round(2)

    port_comp = df.loc[slice(max_date, max_date), slice(None), :].reset_index()[['ticker', 'value']]
    port_comp = port_comp[port_comp['value'] != 0]
    port_comp.reset_index(inplace=True)
    port_comp_dict = {}
    for index, row in port_comp.iterrows():
        port_comp_dict[row['ticker']] = row['value']
    if port_comp_dict == {}:
        port_comp_dict = {'No Assets': 1}

    df = df.reset_index(level=1)
    #print(df.tail(20))
    df = df.groupby(['date']).sum().reset_index()[['date','value']]
    final_value = {'portfolio_value': round(df['value'].to_list()[-1], 2)}
    initial_value = round(df['value'].to_list()[0], 2)

    if int(final_value['portfolio_value']) == 0:
        port_returns = {'port_returns': 0}

    else:
        time_difference = relativedelta(datetime.strptime(max_date, "%Y-%m-%d").date(), datetime.strptime(min_date, "%Y-%m-%d").date())
        difference_in_years = time_difference.years + time_difference.months*(1/12) + time_difference.days*(1/365)

        port_returns = (((final_value['portfolio_value'] - initial_value)/initial_value) + 1)**(1/difference_in_years) - 1
        port_returns = {'port_returns': round(port_returns,3)}

    data = df.round(2).to_json(date_format='iso', orient='split')

    return data, final_value, port_returns, port_comp_dict

def get_all_tickers(connection):
    connection.autocommit = True
    cursor = connection.cursor()

    #sql = f'''SELECT DISTINCT(ticker) FROM spy'''

    #cursor.execute(sql)
    #selected = cursor.fetchall()
    #spy = convert_db_fetch_to_df(selected, column_names=['ticker'])

    sql = f'''SELECT DISTINCT(ticker) FROM spy'''

    cursor.execute(sql)
    selected = cursor.fetchall()
    tsx60 = convert_db_fetch_to_df(selected, column_names=['ticker'])

    #dfpx = spy.append(tsx60, ignore_index=True)
    dfpx = tsx60
    cursor.close()

    ret_dict = {'stocks': dfpx['ticker'].to_list()}
    # print("end")
    return ret_dict

def get_portfolio_weights(stock_dict, account_info, ):
    optimal_portfolio = {}

    numStocks = len(stock_dict.keys())

    for i in range(0, numStocks):
        optimal_portfolio[stock_dict[i]['stock']] = (1/numStocks)

    expected_monthly_returns = {
        'exp': 1.015,
        'ub': 1.02,
        'lb': 1.01
    }
    return optimal_portfolio, expected_monthly_returns

def get_trades(connection, email, opt, rets):
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
    # print(list_tickers)
    if len(list_tickers) > 1:
        string_tickers = '('
        for tick in list_tickers[:-1]:
            string_tickers = string_tickers + (f"'{tick}', ")
        string_tickers = string_tickers + f"'{list_tickers[-1]}')"
    else:
        string_tickers = f"('{list_tickers[0]}')"

    # Get price info for unique tickers, between min and max date

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

    # print(tsx60)

    dfpx = cadetfs.append(ametfs, ignore_index=True)
    dfpx = dfpx.append(spy, ignore_index=True)
    dfpx = dfpx.append(tsx60, ignore_index=True)
    # print(dfpx)

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

    df['value'] = df['cumAmount'] * df['adj_close']
    # print(df)
    df = df.round(2)
    max_port_date = df.index.max()[0]
    df = df.loc[max_port_date]
    port_total_value = sum(df['value'].to_list())
    #print(port_total_value)

    #Calculate # of new stocks to by/old to sell
    curr_stocks = df.reset_index()[df.reset_index()['ticker'] != 'cash']
    old_portfolio_values = df.reset_index()[['ticker', 'value']]
    #print('old_portfolio_values')
    #print(old_portfolio_values)

    #Get list of stocks from optimal dict
    list_tickers = list(opt.keys())

    # print(list_tickers)
    if len(list_tickers) > 1:
        string_tickers = '('
        for tick in list_tickers[:-1]:
            string_tickers = string_tickers + (f"'{tick}', ")
        string_tickers = string_tickers + f"'{list_tickers[-1]}')"
    else:
        string_tickers = f"('{list_tickers[0]}')"

    # Get price info for unique tickers, between min and max date
    sql = f"""
            SELECT MAX(date) FROM americanetfs
            """
    cursor.execute(sql)
    selected = cursor.fetchall()
    df = convert_db_fetch_to_df(selected, column_names=['date'])
    max_date = df['date'][0]

    ### Start hitting prices tables ###
    sql = f'''
                SELECT date, ticker, adj_close FROM canadianetfs WHERE (date = '{max_date}') AND (ticker IN {string_tickers})
                '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    cadetfs = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
                    SELECT date, ticker, adj_close FROM americanetfs WHERE (date = '{max_date}') AND (ticker IN {string_tickers})
                    '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    ametfs = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
                    SELECT date, ticker, adj_close FROM spy WHERE (date = '{max_date}') AND (ticker IN {string_tickers})
                    '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    spy = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
                    SELECT date, ticker, adj_close FROM tsx60 WHERE (date = '{max_date}') AND (ticker IN {string_tickers})
                    '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    tsx60 = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    # print(tsx60)

    dfpx = cadetfs.append(ametfs, ignore_index=True)
    dfpx = dfpx.append(spy, ignore_index=True)
    dfpx = dfpx.append(tsx60, ignore_index=True)

    dfpx['new_weight'] = dfpx['ticker'].map(opt)
    dfpx['port_value'] = port_total_value
    dfpx['new_amount'] = (dfpx['new_weight'] * dfpx['port_value'])/dfpx['adj_close']
    dfpx['new_amount'] = dfpx['new_amount'].astype(int)
    dfpx['new_value'] = dfpx['new_amount'] * dfpx['adj_close']

    new_portfolio_cash_component = (int(100*port_total_value) - int(100*sum(dfpx['new_value'])+1))/100

    new_portfolio_values = dfpx[['ticker', 'new_value']]
    new_portfolio_values.loc[-1] = ['cash', new_portfolio_cash_component]
    new_portfolio_values = new_portfolio_values.reset_index(drop=True)
    new_portfolio_values = new_portfolio_values.rename(columns={'new_value': 'value'})
    #print('new_portfolio_values')
    #print(new_portfolio_values.round(2))
    #print(f'New cash: {new_portfolio_cash_component}')

    dfpx = dfpx[['ticker', 'new_amount']]

    # print(dfpx)
    curr_stocks = curr_stocks[['ticker', 'cumAmount']]
    curr_stocks = curr_stocks.rename(columns={'cumAmount':'old_amount'})

    dfpx = pd.merge(
        dfpx,
        curr_stocks,
        how="outer",
        on=['ticker'],
    )

    dfpx.fillna(0, inplace=True)
    dfpx['delta'] = dfpx['new_amount'] - dfpx['old_amount']
    tradeDf = dfpx.copy()

    #Handle Hypothetical Back Test
    back_test_min_date = max_date - relativedelta(years=5)

    sql = f'''
                    SELECT date, ticker, adj_close FROM americanetfs WHERE (date BETWEEN '{back_test_min_date}' AND '{max_date}') AND (ticker IN {string_tickers})
                    '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    cadetfs = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    sql = f'''
                        SELECT date, ticker, adj_close FROM spy WHERE (date BETWEEN '{back_test_min_date}' AND '{max_date}') AND (ticker IN {string_tickers})
                        '''

    cursor.execute(sql)
    selected = cursor.fetchall()
    tsx60 = convert_db_fetch_to_df(selected, column_names=['date', 'ticker', 'adj_close'])

    # print(tsx60)

    dfpx = cadetfs.append(tsx60, ignore_index=True)

    dfpx = pd.merge(
        dfpx,
        tradeDf[['ticker', 'new_amount']],
        how="left",
        on=['ticker'],
    )

    dfpx['value'] = dfpx['adj_close'] * dfpx['new_amount']


    ########

    dfpx = dfpx.reset_index(drop=True)
    #print(dfpx)
    dfpx = dfpx.set_index(['date', 'ticker'])
    #print(dfpx.dtypes)

    dfpx = dfpx.sort_index(level=1).ffill().reindex(dfpx.index)
    dfpx.fillna(0, inplace=True)

    # print(df)
    dfpx = dfpx.round(2)

    dfpx = dfpx.reset_index(level=1)
    #print(dfpx)
    dfpx = dfpx.groupby(['date']).sum().reset_index()
    #print(dfpx)
    dfpx = dfpx.reset_index()[['date', 'value']]
    dfpx['value'] = dfpx['value'] + new_portfolio_cash_component

    df_future_up = dfpx.copy()[dfpx['date'] == max_date].reset_index(drop=True)
    df_future_mid = dfpx.copy()[dfpx['date'] == max_date].reset_index(drop=True)
    df_future_down = dfpx.copy()[dfpx['date'] == max_date].reset_index(drop=True)


    for key in rets.keys():
        future_ret_date = max_date + relativedelta(months=key)

        df_future_up.loc[-1] = [future_ret_date, df_future_up['value'][0]*rets[key]]
        df_future_up = df_future_up.reset_index(drop=True)

        df_future_mid.loc[-1] = [future_ret_date, df_future_mid['value'][0] * rets[key]]
        df_future_mid = df_future_mid.reset_index(drop=True)

        df_future_down.loc[-1] = [future_ret_date, df_future_down['value'][0] * rets[key]]
        df_future_down = df_future_down.reset_index(drop=True)

    print(df_future_up)

    return dfpx, df_future_up, df_future_mid, df_future_down, tradeDf[['ticker', 'delta']].to_dict('records'), old_portfolio_values.round(2), new_portfolio_values.round(2)



#test = {0: {'stock': 'L.TO', 'range': [0, 100]}, 1: {'stock': 'SAP.TO', 'range': [0, 100]}, 2: {'stock': 'WCN.TO', 'range': [0, 100]}}
#opt, rets = get_portfolio_weights(test, True)
#bt, fu, fm, fd, trades = get_trades(conn, 'aidanjones55@gmail.com', opt, rets)
#print(trades)

#print(opt)
#print(rets)

#transact_stock(conn, 'aidanjones55@gmail.com', 'RY.TO', -40)
#get_portfolio_value_by_date(conn, 'aidanjones55@gmail.com')

#print(get_all_tickers(conn))


#get_portfolio_value_by_date(conn, 'aidanjones55@gmail.com')

#transact_stock(conn, 'aidanjones55@gmail.com', 'AAPL', 100)
#wipe_table(conn, 'clients')
#wipe_table(conn, 'transactions')
#add_transaction(conn, 'aidanjones55@gmail.com', 'cash', 1000)


#add_transaction(conn, 'aidanjones55@gmail.com', 'cash', 0, 'Initialize Account')
#get_past_deposits(conn, 'aidanjones55@gmail.com')
#print(pull_user_data(conn, 'aidanjones55@gmail.com', 'admin'))
