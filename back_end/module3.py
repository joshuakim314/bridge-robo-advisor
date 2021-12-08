
dfs = []
for chunk in pandas.read_sql_query(sql_query, con=cnx, chunksize=n):
	dfs.append(chunk)
df = pd.concat(dfs)


