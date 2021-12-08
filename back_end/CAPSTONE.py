import psycopg2
import yfinance as yf
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from yfinance import ticker
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web 
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Create all the tables used in this database
"""
conn = psycopg2.connect(
    host = 'database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    database = 'can2_etfs',
    user = 'postgres',
    password = 'capstone',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

cur.execute('''CREATE TABLE currency2
                (
                Date DATE NOT NULL,
                Open FLOAT NOT NULL,
                High FLOAT NOT NULL,
                Low FLOAT NOT NULL,
                Close FLOAT NOT NULL,
                Volume FLOAT,
                Dividends FLOAT,
                Stock_Splits  FLOAT,
                PRIMARY KEY(date)
                );''')
print('table created successfully')
"""

"""
cur.execute('''CREATE TABLE fama
                (
                date INT NOT NULL,
                excess FLOAT NOT NULL,
                smb FLOAT NOT NULL,
                hml FLOAT NOT NULL,
                rmw FLOAT NOT NULL,
                cma FLOAT NOT NULL,
                riskfree FLOAT NOT NULL,
                PRIMARY KEY(date)
                );''')
print('table created successfully')
conn.close()


cur.execute(''' CREATE TABLE transactions
                (
                datetime timestamp NOT NULL,
                email VARCHAR(255) NOT NULL,
                ticker VARCHAR(255) NOT NULL,
                amount INT NOT NULL,
                PRIMARY KEY(datetime,email)
                );''')

cur.execute(''' CREATE TABLE preferences
                (
                email VARCHAR(255) NOT NULL,
                ticker VARCHAR(255) NOT NULL,
                min INT NOT NULL,
                max INT NOT NULL,
                PRIMARY KEY(email,ticker)
                );''')
print('tables created successfully')
conn.close()


cur.execute('''CREATE TABLE clients
               (
                first VARCHAR(255) NOT NULL,
                last VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL,
                risk INT NOT NULL,
                control INT NOT NULL,
                PRIMARY KEY (email)
                );''')

print('Table created successfully')
conn.close()

"""
#------------------------------------------------------------------------------------------------------------------------------------------------------
#Collect all the required security tickers and format properly as list
can_etf_ticker_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_Canadian_exchange-traded_funds')
can_etf_tickers = can_etf_ticker_data[0].Symbol  # Get the ticker symbols

tsx_tickers = ['AEM.TO', 'AQN.TO', 'ATD.B.TO', 'BCE.TO', 'BMO.TO', 'BNS.TO', 'ABX.TO', 'BHC.TO', 'BAM-A.TO', 'BIP-UN.TO', 'BPY-UN.TO', 'CCL-B.TO', 'GIB-A.TO', 'CCO.TO', 'CAR-UN.TO', 'CM.TO', 'CNR.TO', 'CNQ.TO', 'CP.TO', 'CTC-A.TO', 'WEED.TO', 'CVE.TO', 'CSU.TO', 'DOL.TO', 'EMA.TO', 'ENB.TO', 'FM.TO', 'FSV.TO', 'FTS.TO', 'FNV.TO', 'WN.TO', 'GIL.TO', 'IMO.TO', 'K.TO', 'KL.TO', 'L.TO', 'MG.TO', 'MFC.TO', 'MRU.TO', 'NTR.TO', 'OTEX.TO', 'PPL.TO', 'POW.TO', 'QSR.TO', 'RCI-B.TO', 'RY.TO', 'SNC.TO', 'SAP.TO', 'SJR-B.TO', 'SHOP.TO', 'SLF.TO', 'SU.TO', 'TRP.TO', 'TECK-B.TO', 'T.TO', 'TRI.TO', 'TD.TO', 'WCN.TO', 'WPM.TO']


sp500_tickers_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500_tickers = sp500_tickers_data[0].Symbol
sp500_tickers = sp500_tickers.tolist()

usa_etf_ticker_data = pd.read_csv(r'C:\Users\James\Desktop\data\us_etf_tickers.csv')
usa_etf_tickers = usa_etf_ticker_data.Symbol
usa_etf_tickers = usa_etf_tickers.tolist()

tickers_list = can_etf_tickers.to_list()   # Converts to list to use next
for i in range(len(tickers_list)):
    tickers_list[i] =  tickers_list[i][5:]
    tickers_list[i] = tickers_list[i].replace('.','-')  
    tickers_list[i] = tickers_list[i] + '.TO'

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Connect to cloud database and update candadian etf prices

conn = psycopg2.connect(
    host = 'database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    database = 'can2_etfs',
    user = 'postgres',
    password = 'capstone',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

    
for i in range(len(tickers_list)):
    try:
        ticker = tickers_list[i]
        stock_data = yf.download(ticker,period='1d')
    
        stock_data.index = np.datetime_as_string(stock_data.index, unit='D')
        stock_data['Ticker'] = ticker
        stock_data = stock_data.rename(columns={'Adj Close':'Adj_Close'})
        records = stock_data.to_records(index=True)
    
        query = '''INSERT INTO canadianetfs (Date, Open, High, Low, Close, Adj_Close, Volume, Ticker)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
        cur = conn.cursor()
        cur.executemany(query, records)
    except:
        print('execption')
print("Data Insert Successfully")
conn.close()

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Connect to cloud databse and update american etf table
conn = psycopg2.connect(
    host = 'database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    database = 'can2_etfs',
    user = 'postgres',
    password = 'capstone',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

for i in range(len(usa_etf_tickers)):
    try:
        ticker = usa_etf_tickers[i]
        stock_data = yf.download(ticker,period='1d')
    
        stock_data.index = np.datetime_as_string(stock_data.index, unit='D')
        stock_data['Ticker'] = ticker
        stock_data = stock_data.rename(columns={'Adj Close':'Adj_Close'})
        records = stock_data.to_records(index=True)
    
        query = '''INSERT INTO americanetfs (Date, Open, High, Low, Close, Adj_Close, Volume, Ticker)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
        cur = conn.cursor()
        cur.executemany(query, records)
    except:
        print('execption')
print("Data Insert Successfully")
conn.close()

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Connect to cloud databse and and update tsx/60 stock prices
conn = psycopg2.connect(
    host = 'database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    database = 'can2_etfs',
    user = 'postgres',
    password = 'capstone',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

    
for i in range(len(tsx_tickers)):
    try:
        ticker = tsx_tickers[i]
        stock_data = yf.download(ticker,period='1d')
    
        stock_data.index = np.datetime_as_string(stock_data.index, unit='D')
        stock_data['Ticker'] = ticker
        stock_data = stock_data.rename(columns={'Adj Close':'Adj_Close'})
        records = stock_data.to_records(index=True)
    
        query = '''INSERT INTO tsx60 (Date, Open, High, Low, Close, Adj_Close, Volume, Ticker)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
        cur = conn.cursor()
        cur.executemany(query, records)
    except:
        print('execption')
print("Data Insert Successfully")
conn.close()

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Connect to cloud database and update sp500 stock prices
conn = psycopg2.connect(
    host = 'database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    database = 'can2_etfs',
    user = 'postgres',
    password = 'capstone',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

for i in range(len(sp500_tickers)):
    try:
        ticker = sp500_tickers[i]
        stock_data = yf.download(ticker,period='1')
    
        stock_data.index = np.datetime_as_string(stock_data.index, unit='D')
        stock_data['Ticker'] = ticker
        stock_data = stock_data.rename(columns={'Adj Close':'Adj_Close'})
        records = stock_data.to_records(index=True)
    
        query = '''INSERT INTO spy (Date, Open, High, Low, Close, Adj_Close, Volume, Ticker)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
        cur = conn.cursor()
        cur.executemany(query, records)
    except:
        print('execption')
print("Data Insert Successfully")
conn.close()


"""

conn = psycopg2.connect(
    host = 'localhost',
    user = 'postgres',
    password = 'carsoneagles',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

"""

"""

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Create tables to be filled first on local databse 
conn = psycopg2.connect(
    host = 'localhost',
    database = 'securities',
    user = 'postgres',
    password = 'carsoneagles',
    port = '5432'
    )

conn.autocommit=True

cur = conn.cursor()

cur.execute('''CREATE TABLE canadianetfs
               (
                Date DATE NOT NULL,
                Open FLOAT NOT NULL,
                High FLOAT NOT NULL,
                Low FLOAT NOT NULL,
                Close FLOAT NOT NULL,
                Adj_Close FLOAT NOT NULL,
                Volume BIGINT NOT NULL,
                Ticker VARCHAR(255) NOT NULL
                
                
                );''')

print('Table created successfully')
conn.close()

"""
"""
conn = psycopg2.connect(
    database="securities",
    user='postgres', 
    password='carsoneagles', 
    host='localhost', 
    port= '5432'
)
conn.autocommit = True
cur = conn.cursor()
"""
"""
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Download the bulk of the data to local database then migrate this data via a backup .sql file

for i in range(len(tickers_list)):
    try:
        ticker = tickers_list[i]
        stock_data = yf.download(ticker,start='2000-01-01',end='2021-11-15')
    
        stock_data.index = np.datetime_as_string(stock_data.index, unit='D')
        stock_data['Ticker'] = ticker
        stock_data = stock_data.rename(columns={'Adj Close':'Adj_Close'})
        records = stock_data.to_records(index=True)
    
        query = '''INSERT INTO spy (Date, Open, High, Low, Close, Adj_Close, Volume, Ticker)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
        cur = conn.cursor()
        cur.executemany(query, records)
    except:
        print('execption')
print("Data Insert Successfully")
conn.close()
"""
"""
sql2 = '''ALTER TABLE canadianetfs
ADD PRIMARY KEY (Date,Ticker)'''
cur =conn.cursor()
cur.execute(sql2)
conn.close()
"""


#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Update Currency conversion table for CAD in USD
#Not currently functinal only using american securities
"""
conn = psycopg2.connect(
    host = 'database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    database = 'can2_etfs',
    user = 'postgres',
    password = 'capstone',
    port = '5432'
   )
print('Connected to postgr db')

#USD_cad = yf.Ticker("USDCAD=X")
hist = yf.download("CADUSD=X",period='1d')

hist.index = np.datetime_as_string(hist.index, unit='D')
hist = hist.rename(columns={'Stock Split':'Stock_Split'})
currency_records = hist.to_records(index=True)
query = '''INSERT INTO currency2 (Date, Open, High, Low, Close, Volume, Dividends,Stock_Splits)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
cur = conn.cursor()
cur.executemany(query, currency_records)
"""

