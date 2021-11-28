import numpy as np
import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

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
print(selected)
cursor.close()
print("end")
