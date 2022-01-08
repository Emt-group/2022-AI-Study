
import pandas as pd


df = pd.read_csv("./ch02/insurance.csv")



print(df)

print( df[df["age"] < 20] )

print( df.iloc[2:5,:] )

