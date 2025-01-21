import pandas as pd

df1 = pd.read_csv('data/LegoStatsMain.csv')
df2 = pd.read_csv('data/LegoStatsV2.csv')

df1['set_id'] = df1['set_id'].astype(str)
df2['set_id'] = df2['set_id'].astype(str)

dfMain = pd.merge(df1, df2, on='set_id',how='left')

dfMain.to_csv('LegoStatsMainV2.csv', index=False)