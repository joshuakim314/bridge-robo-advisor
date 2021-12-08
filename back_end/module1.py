
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
import pandas as pd

conn = psycopg2.connect(
    host = 'localhost',
    database = 'can2_etfs'
    user = 'postgres',
    password = 'carsoneagles',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()

cur.execute('''CREATE TABLE clients
               (
                PRIMARY KEY VARCHAR(255),
                first VARCHAR(255) NOT NULL,
                last VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL,
                portfolio_id VARCHAR(255) NOT NULL
                );''')

print('Table created successfully')
conn.close()

conn = psycopg2.connect(
    host = 'localhost',
    database = 'can2_etfs'
    user = 'postgres',
    password = 'carsoneagles',
    port = '5432'
   )
print('Connected to postgr db')

conn.autocommit=True 
cur = conn.cursor()
"""
cur.execute('''CREATE TABLE portfolios
               (
                portfolio_id VARCHAR(255) NOT NULL,
                first VARCHAR(255),
                last VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL,
                portfolio_id VARCHAR(255) NOT NULL
                );''')

print('Table created successfully')
conn.close()
"""