import pandas as pd

df1 = pd.read_csv('data/LegoStatsV1.csv')
df2 = pd.read_csv('data/LegoStatsV3.csv')

dfMain = pd.merge(df1, df2, on='set_id', how='inner')

dfMain.to_csv('LegoStatsMain.csv', index=False)