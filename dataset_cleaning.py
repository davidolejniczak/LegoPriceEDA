import kagglehub
import pandas as pd

path = kagglehub.dataset_download("alexracape/lego-sets-and-prices-over-time")

df = pd.read_csv("sets.csv")
df = df.drop(["Owned","Rating","Total_Quantity","Availability","Subtheme","Category","Packaging","Num_Instructions","Subtheme"],axis=1)
df = df.dropna(subset=["Current_Price","USD_MSRP","Minifigures"],axis=0,how='any')
df.to_csv("sets.csv",index=False)

